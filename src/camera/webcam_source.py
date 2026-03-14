import cv2
import numpy as np
from typing import Optional

from .base import CameraSource


class WebcamSource(CameraSource):
    """Reads frames from a USB webcam, returning last good frame on miss."""

    def __init__(self, index: int = 0) -> None:
        self._index = index
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._index)

    def read(self) -> np.ndarray:
        if self._cap is None or not self._cap.isOpened():
            return self._last_frame

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return self._last_frame

        self._last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._last_frame

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
