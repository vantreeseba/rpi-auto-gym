import cv2
import numpy as np

from .base import CameraSource


class FileCameraSource(CameraSource):
    """Reads frames from an image or video file, looping video at end."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._cap: cv2.VideoCapture | None = None
        self._black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._path)

    def read(self) -> np.ndarray:
        if self._cap is None or not self._cap.isOpened():
            return self._black_frame

        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()

        if not ok or frame is None:
            return self._black_frame

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
