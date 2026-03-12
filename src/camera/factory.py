from .base import CameraSource
from .file_source import FileCameraSource
from .pi_source import PiCameraSource
from .webcam_source import WebcamSource

_SOURCES = {
    "file": FileCameraSource,
    "webcam": WebcamSource,
    "picamera": PiCameraSource,
}


def make_camera(source: str, **kwargs) -> CameraSource:
    """Instantiate a camera source by name.

    Args:
        source: One of 'file', 'webcam', 'picamera'.
        **kwargs: Passed directly to the source constructor
                  (e.g. path=, index=, width=, height=).

    Raises:
        ValueError: If source name is not recognised.
    """
    source_class = _SOURCES.get(source)
    if source_class is None:
        valid = ", ".join(f"'{k}'" for k in _SOURCES)
        raise ValueError(f"Unknown camera source '{source}'. Valid options: {valid}")

    return source_class(**kwargs)
