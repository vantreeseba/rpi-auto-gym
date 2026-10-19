"""Pose skeleton drawing shared by the debug window and the web interface."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class Pose:
    keypoints: dict[str, Keypoint]
    timestamp: float


SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

_KEYPOINT_COLOUR = (0, 255, 0)     # green in BGR
_SKELETON_COLOUR = (0, 200, 255)   # yellow-ish in BGR
_KEYPOINT_RADIUS = 5
_SKELETON_THICKNESS = 2
_MIN_CONFIDENCE = 0.3


def _draw_skeleton(frame: "np.ndarray", pose: Pose) -> "np.ndarray":
    """Return a copy of frame with pose skeleton overlaid."""
    canvas = frame.copy()

    for name, kp in pose.keypoints.items():
        if kp.confidence < _MIN_CONFIDENCE:
            continue
        cx = int(kp.x * canvas.shape[1])
        cy = int(kp.y * canvas.shape[0])
        cv2.circle(canvas, (cx, cy), _KEYPOINT_RADIUS, _KEYPOINT_COLOUR, -1)

    for start_name, end_name in SKELETON_CONNECTIONS:
        start_kp = pose.keypoints.get(start_name)
        end_kp = pose.keypoints.get(end_name)

        if start_kp is None or end_kp is None:
            continue
        if start_kp.confidence < _MIN_CONFIDENCE or end_kp.confidence < _MIN_CONFIDENCE:
            continue

        sx = int(start_kp.x * canvas.shape[1])
        sy = int(start_kp.y * canvas.shape[0])
        ex = int(end_kp.x * canvas.shape[1])
        ey = int(end_kp.y * canvas.shape[0])
        cv2.line(canvas, (sx, sy), (ex, ey), _SKELETON_COLOUR, _SKELETON_THICKNESS)

    return canvas


def draw_skeleton(frame: "np.ndarray", pose) -> "np.ndarray":
    """Public alias: frame copy with the pose drawn (pose may be None)."""
    return _draw_skeleton(frame, pose) if pose is not None else frame.copy()
