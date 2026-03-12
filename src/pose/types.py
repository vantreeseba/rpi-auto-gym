from dataclasses import dataclass

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass
class Keypoint:
    x: float           # normalized 0.0–1.0 (left→right)
    y: float           # normalized 0.0–1.0 (top→bottom)
    confidence: float  # 0.0–1.0


@dataclass
class Pose:
    keypoints: dict[str, Keypoint]  # COCO keypoint names as keys
    timestamp: float                # time.monotonic()
