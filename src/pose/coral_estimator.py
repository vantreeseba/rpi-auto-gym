"""MoveNet single-pose on a Google Coral Edge TPU.

Uses ai-edge-litert (or the older tflite-runtime) with the Edge TPU plugin
(libedgetpu, from Google's apt repo), not pycoral: pycoral stops at
Python 3.9 and Pi OS Bookworm ships 3.11.
"""
import time
from typing import Any, Dict, Optional

import numpy as np

from .base import PoseEstimator
from .preprocess import Letterbox, letterbox
from .types import COCO_KEYPOINT_NAMES, Keypoint, Pose

EDGETPU_LIB = "libedgetpu.so.1"


def _make_interpreter(model_path: str) -> Any:
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate  # type: ignore[import]
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate  # type: ignore[import]
        except ImportError as exc:
            raise NotImplementedError(
                "Install ai-edge-litert to use the Coral backend "
                "(scripts/setup_pi.sh --coral, or pip install -e .[coral])"
            ) from exc
    try:
        delegate = load_delegate(EDGETPU_LIB)
    except (ValueError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not load the Edge TPU library ({EDGETPU_LIB}). Is libedgetpu1-std "
            "installed and the Coral plugged in? (USB: unplug and replug after installing.)"
        ) from exc
    return Interpreter(model_path=model_path, experimental_delegates=[delegate])


class CoralPoseEstimator(PoseEstimator):
    def __init__(self, model_path: str, min_confidence: float = 0.5,
                 interpreter: Any = None) -> None:
        self._interpreter = interpreter or _make_interpreter(model_path)
        self._interpreter.allocate_tensors()
        self._min_confidence = min_confidence
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        _, self._in_h, self._in_w, _ = (int(v) for v in self._input["shape"])

    def estimate(self, frame: np.ndarray) -> Optional[Pose]:
        image, box = letterbox(frame, self._in_w, self._in_h)
        self._interpreter.set_tensor(self._input["index"], self._to_input(image))
        self._interpreter.invoke()

        # MoveNet output: [1, 1, 17, 3], each row (y, x, score), normalised to the input
        keypoint_data = self._dequantize(self._interpreter.get_tensor(self._output["index"]))[0][0]
        keypoints = self._build_keypoints(keypoint_data, box)

        if all(kp.confidence < self._min_confidence for kp in keypoints.values()):
            return None
        return Pose(keypoints=keypoints, timestamp=time.monotonic())

    def _to_input(self, image: np.ndarray) -> np.ndarray:
        dtype = self._input["dtype"]
        if dtype == np.float32:  # float (non-TPU) MoveNet takes 0..255 floats
            return image[np.newaxis].astype(np.float32)
        return image[np.newaxis].astype(dtype)

    def _dequantize(self, data: np.ndarray) -> np.ndarray:
        if data.dtype == np.float32:
            return data
        scale, zero_point = self._output.get("quantization", (0.0, 0))
        if not scale:
            return data.astype(np.float32)
        return (data.astype(np.float32) - zero_point) * scale

    def _build_keypoints(self, keypoint_data: np.ndarray, box: Letterbox) -> Dict[str, Keypoint]:
        keypoints = {}
        for idx, name in enumerate(COCO_KEYPOINT_NAMES):
            y, x, score = (float(v) for v in keypoint_data[idx])
            fx, fy = box.to_frame(x * self._in_w, y * self._in_h)
            keypoints[name] = Keypoint(x=fx, y=fy, confidence=score)
        return keypoints
