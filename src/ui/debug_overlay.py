from dataclasses import dataclass
from typing import Callable

import tkinter as tk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as exc:
    raise ImportError(
        "DebugOverlay requires opencv-python, numpy, and Pillow. "
        "Install them with: pip install opencv-python numpy Pillow"
    ) from exc


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


class DebugOverlay(tk.Toplevel):
    """Floating window showing camera feed with pose skeleton drawn on top."""

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.title("Debug — Camera Feed")
        self.configure(bg="#000000")
        self.resizable(True, True)

        self._canvas = tk.Canvas(self, bg="#000000", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Keep a reference to prevent garbage collection
        self._photo: "ImageTk.PhotoImage | None" = None

    def update_frame(self, frame: "np.ndarray", pose: "Pose | None") -> None:
        """Draw frame + skeleton and update the canvas. frame is RGB uint8."""
        rendered = _draw_skeleton(frame, pose) if pose is not None else frame.copy()

        pil_image = Image.fromarray(rendered)
        photo = ImageTk.PhotoImage(pil_image)

        canvas_width = pil_image.width
        canvas_height = pil_image.height
        self._canvas.config(width=canvas_width, height=canvas_height)

        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        # Hold reference so GC doesn't collect the image
        self._photo = photo
