"""YOLOv8-pose on a Hailo-8 / Hailo-8L (Raspberry Pi AI HAT+ / AI Kit).

The model is loaded onto the chip once, at start-up. Each frame is
letterboxed to the model input (640x640 for the model-zoo yolov8s_pose),
and the raw output tensors are decoded here.

The model-zoo HEF has no on-chip post-processing. Per stride (8, 16, 32)
it has three outputs, told apart by channel count:
  64 = box distances (not needed here), 1 = person score, 51 = 17 keypoints x (x, y, conf).
We only want the most confident person, so this skips box decoding and NMS
(overlap removal) and just takes the grid cell with the highest score.
"""
import math
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from .base import PoseEstimator
from .preprocess import letterbox
from .types import COCO_KEYPOINT_NAMES, Keypoint, Pose

NUM_KEYPOINTS = len(COCO_KEYPOINT_NAMES)
INFER_TIMEOUT_MS = 1000


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_best_pose(outputs: Iterable[np.ndarray], input_w: int, input_h: int
                     ) -> Optional[Tuple[float, np.ndarray]]:
    """Find the most confident person in raw YOLOv8-pose outputs.

    Returns (score, keypoints) with keypoints shaped (17, 3) as
    (x, y, confidence), x/y in model-input pixels; or None if there are no
    outputs to decode.
    """
    groups: Dict[int, Dict[str, np.ndarray]] = {}
    for arr in outputs:
        a = np.asarray(arr, dtype=np.float32)
        if a.shape[-2:] == (NUM_KEYPOINTS, 3):
            a = a.reshape(a.shape[:-2] + (NUM_KEYPOINTS * 3,))
        channels = a.shape[-1]
        if channels == 1:
            kind = "score"
        elif channels == NUM_KEYPOINTS * 3:
            kind = "kpts"
        else:
            continue  # box distances
        flat = a.reshape(-1, channels)
        groups.setdefault(flat.shape[0], {})[kind] = flat

    best: Optional[Tuple[float, np.ndarray]] = None
    for cells, group in groups.items():
        if "score" not in group or "kpts" not in group:
            continue
        stride = math.sqrt(input_w * input_h / cells)
        cols = round(input_w / stride)
        scores = group["score"][:, 0]
        if scores.min() < 0.0 or scores.max() > 1.0:  # logits rather than probabilities
            scores = _sigmoid(scores)
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if best is not None and score <= best[0]:
            continue
        row, col = divmod(idx, cols)
        raw = group["kpts"][idx].reshape(NUM_KEYPOINTS, 3)
        kpts = np.empty_like(raw)
        kpts[:, 0] = (raw[:, 0] * 2.0 + col) * stride
        kpts[:, 1] = (raw[:, 1] * 2.0 + row) * stride
        kpts[:, 2] = _sigmoid(raw[:, 2])
        best = (score, kpts)
    return best


class _HailoRunner:
    """Keeps the HEF configured on the device and runs one frame at a time."""

    def __init__(self, model_path: str) -> None:
        try:
            from hailo_platform import (  # type: ignore[import-untyped]
                FormatType, HailoSchedulingAlgorithm, VDevice,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "Install hailort to use the Hailo backend (scripts/setup_pi.sh --hailo, "
                "which installs the hailo-all package)"
            ) from exc

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._device = VDevice(params)
        self._model = self._device.create_infer_model(model_path)
        self._model.set_batch_size(1)
        self._model.input().set_format_type(FormatType.UINT8)
        self._names = list(self._model.output_names)
        for name in self._names:
            self._model.output(name).set_format_type(FormatType.FLOAT32)
        self._ctx = self._model.configure()
        self._configured = self._ctx.__enter__()
        self._buffers = {
            name: np.empty(self._model.output(name).shape, dtype=np.float32) for name in self._names
        }
        self.input_shape: Tuple[int, ...] = tuple(self._model.input().shape)  # (H, W, C)

    def infer(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        bindings = self._configured.create_bindings(output_buffers=self._buffers)
        bindings.input().set_buffer(np.ascontiguousarray(image))
        self._configured.run([bindings], INFER_TIMEOUT_MS)
        return {name: bindings.output(name).get_buffer() for name in self._names}

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)
        self._device.release()


class HailoPoseEstimator(PoseEstimator):
    def __init__(self, model_path: str, min_confidence: float = 0.5, runner: Any = None) -> None:
        self._runner = runner or _HailoRunner(model_path)
        self._in_h, self._in_w = (int(v) for v in self._runner.input_shape[:2])
        self._min_confidence = min_confidence

    def estimate(self, frame: np.ndarray) -> Optional[Pose]:
        image, box = letterbox(frame, self._in_w, self._in_h)
        outputs = self._runner.infer(image)
        best = decode_best_pose(outputs.values(), self._in_w, self._in_h)
        if best is None:
            shapes = {name: np.shape(out) for name, out in outputs.items()}
            raise RuntimeError(f"Unexpected Hailo model outputs {shapes}; expected a YOLOv8-pose HEF")
        score, kpts = best
        if score < self._min_confidence:
            return None

        keypoints = {}
        for idx, name in enumerate(COCO_KEYPOINT_NAMES):
            x, y = box.to_frame(float(kpts[idx, 0]), float(kpts[idx, 1]))
            keypoints[name] = Keypoint(x=x, y=y, confidence=float(kpts[idx, 2]))
        return Pose(keypoints=keypoints, timestamp=time.monotonic())

    def close(self) -> None:
        close = getattr(self._runner, "close", None)
        if close:
            close()
