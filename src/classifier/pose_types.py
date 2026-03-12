from dataclasses import dataclass


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class Pose:
    keypoints: dict[str, Keypoint]
    timestamp: float
