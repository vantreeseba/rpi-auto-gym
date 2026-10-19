"""Image prep shared by the accelerator backends (Coral, Hailo)."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Letterbox:
    """How a frame was fitted into the model input, so points can be mapped back."""
    scale: float
    pad_x: int
    pad_y: int
    frame_w: int
    frame_h: int

    def to_frame(self, x: float, y: float) -> Tuple[float, float]:
        """Model-input pixel coords -> normalised 0..1 frame coords."""
        fx = (x - self.pad_x) / self.scale / self.frame_w
        fy = (y - self.pad_y) / self.scale / self.frame_h
        return fx, fy


def letterbox(frame: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, Letterbox]:
    """Resize ``frame`` to fit ``width`` x ``height`` without stretching, padding with black."""
    import cv2  # type: ignore[import-untyped]

    frame_h, frame_w = frame.shape[:2]
    scale = min(width / frame_w, height / frame_h)
    new_w, new_h = max(1, round(frame_w * scale)), max(1, round(frame_h * scale))
    pad_x, pad_y = (width - new_w) // 2, (height - new_h) // 2

    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(frame, (new_w, new_h))
    return out, Letterbox(scale, pad_x, pad_y, frame_w, frame_h)
