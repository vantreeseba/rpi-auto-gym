import time

import numpy as np

from .base import PoseEstimator
from .types import COCO_KEYPOINT_NAMES, Keypoint, Pose

# MoveNet 17-keypoint order matches COCO index order exactly
MOVENET_KEYPOINT_ORDER = COCO_KEYPOINT_NAMES


class CoralPoseEstimator(PoseEstimator):
    def __init__(self, model_path: str, min_confidence: float = 0.5) -> None:
        try:
            from pycoral.utils.edgetpu import make_interpreter
            from pycoral.adapters import common
        except ImportError as exc:
            raise NotImplementedError(
                "Install pycoral/hailort to use this backend"
            ) from exc

        self._interpreter = make_interpreter(model_path)
        self._interpreter.allocate_tensors()
        self._min_confidence = min_confidence
        self._common = common

    def estimate(self, frame: np.ndarray) -> Pose | None:
        try:
            from ai_edge_litert.interpreter import Interpreter as _
        except ImportError as exc:
            raise NotImplementedError(
                "Install pycoral/hailort to use this backend"
            ) from exc

        input_details = self._interpreter.get_input_details()
        input_shape = input_details[0]["shape"]
        height, width = input_shape[1], input_shape[2]

        resized = np.expand_dims(
            np.array(
                self._resize_frame(frame, height, width),
                dtype=np.uint8,
            ),
            axis=0,
        )
        self._common.set_input(self._interpreter, resized)
        self._interpreter.invoke()

        # Output shape: [1, 1, 17, 3] — values are (y, x, score)
        output = self._interpreter.get_tensor(
            self._interpreter.get_output_details()[0]["index"]
        )
        keypoint_data = output[0][0]  # shape: [17, 3]

        keypoints = self._build_keypoints(keypoint_data)

        all_below_threshold = all(
            kp.confidence < self._min_confidence for kp in keypoints.values()
        )
        if all_below_threshold:
            return None

        return Pose(keypoints=keypoints, timestamp=time.monotonic())

    def _resize_frame(
        self, frame: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        import cv2  # type: ignore[import-untyped]
        return cv2.resize(frame, (width, height))

    def _build_keypoints(self, keypoint_data: np.ndarray) -> dict[str, Keypoint]:
        # MoveNet outputs (y, x, score) — swap to (x, y) for Keypoint
        return {
            name: Keypoint(
                x=float(keypoint_data[idx][1]),
                y=float(keypoint_data[idx][0]),
                confidence=float(keypoint_data[idx][2]),
            )
            for idx, name in enumerate(MOVENET_KEYPOINT_ORDER)
        }
