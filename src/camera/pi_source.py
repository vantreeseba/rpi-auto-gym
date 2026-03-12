import numpy as np

from .base import CameraSource


class PiCameraSource(CameraSource):
    """Reads frames from the Raspberry Pi Camera Module via picamera2.

    picamera2 is imported at call-time so this file can be imported safely
    on non-RPi machines.
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._width = width
        self._height = height
        self._camera = None

    def start(self) -> None:
        from picamera2 import Picamera2  # type: ignore[import]

        self._camera = Picamera2()
        config = self._camera.create_preview_configuration(
            main={"size": (self._width, self._height), "format": "RGB888"}
        )
        self._camera.configure(config)
        self._camera.start()

    def read(self) -> np.ndarray:
        if self._camera is None:
            raise RuntimeError("PiCameraSource.start() must be called before read()")

        frame = self._camera.capture_array()
        return frame.astype(np.uint8)

    def stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
