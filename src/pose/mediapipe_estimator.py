import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from .base import PoseEstimator
from .types import COCO_KEYPOINT_NAMES, Keypoint, Pose

if TYPE_CHECKING:
    from mediapipe.tasks.python import vision as _vision

BLAZEPOSE_TO_COCO: dict[str, int] = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class MediaPipePoseEstimator(PoseEstimator):
    def __init__(self, model_path: str, min_confidence: float = 0.5) -> None:
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for MediaPipePoseEstimator. "
                "Install it with: pip install mediapipe"
            ) from exc

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._vision = vision
        self._min_confidence = min_confidence

    def estimate(self, frame: np.ndarray) -> Optional[Pose]:
        mp_image = self._vision.Image(
            image_format=self._vision.ImageFormat.SRGB,
            data=frame,
        )
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        # Single-person MVP: always use first detected person (index 0)
        landmarks = result.pose_landmarks[0]

        keypoints = self._build_keypoints(landmarks)

        all_below_threshold = all(
            kp.confidence < self._min_confidence for kp in keypoints.values()
        )
        if all_below_threshold:
            return None

        return Pose(keypoints=keypoints, timestamp=time.monotonic())

    def _build_keypoints(self, landmarks: list) -> dict[str, Keypoint]:
        return {
            name: Keypoint(
                x=landmarks[idx].x,
                y=landmarks[idx].y,
                confidence=landmarks[idx].visibility,
            )
            for name, idx in BLAZEPOSE_TO_COCO.items()
        }
