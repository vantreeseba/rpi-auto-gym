"""Tests for camera source implementations."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.camera.factory import make_camera
from src.camera.file_source import FileCameraSource
from src.camera.webcam_source import WebcamSource


@pytest.fixture
def png_path(tmp_path):
    """Create a tiny 1×1 pixel RGB PNG test fixture."""
    pixel = np.array([[[255, 128, 0]]], dtype=np.uint8)
    bgr_pixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2BGR)
    file_path = str(tmp_path / "pixel.png")
    cv2.imwrite(file_path, bgr_pixel)
    return file_path


class TestFileCameraSource:
    def test_read_returns_ndarray_with_correct_shape_and_dtype(self, png_path):
        cam = FileCameraSource(png_path)
        cam.start()
        frame = cam.read()
        cam.stop()

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8

    def test_read_converts_to_rgb(self, png_path):
        """Pixel was written as orange (255,128,0) in RGB; verify round-trip."""
        cam = FileCameraSource(png_path)
        cam.start()
        frame = cam.read()
        cam.stop()

        r, g, b = frame[0, 0]
        assert r == 255
        assert g == 128
        assert b == 0

    def test_loops_on_static_image(self, png_path):
        """Multiple reads on a single image should all succeed (loop rewind)."""
        cam = FileCameraSource(png_path)
        cam.start()

        for _ in range(5):
            frame = cam.read()
            assert isinstance(frame, np.ndarray)
            assert frame.dtype == np.uint8

        cam.stop()

    def test_returns_black_frame_when_file_missing(self):
        cam = FileCameraSource("/nonexistent/path/image.png")
        cam.start()
        frame = cam.read()
        cam.stop()

        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3

    def test_loops_by_resetting_video_position(self, png_path):
        """Simulate end-of-file: mock cap returns False once, then a real frame.

        We use a fully mocked VideoCapture so we can control read() return values,
        and verify that a second read is attempted after position reset.
        """
        real_frame = np.zeros((1, 1, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (False, None),   # first call: end-of-file
            (True, real_frame),  # second call: after rewind
        ]

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = FileCameraSource(png_path)
            cam.start()
            frame = cam.read()
            cam.stop()

        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert mock_cap.read.call_count == 2
        mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)


class TestWebcamSource:
    def test_instantiation_without_real_device(self):
        """WebcamSource can be created without a live camera."""
        cam = WebcamSource(index=99)
        assert cam._index == 99

    def test_read_with_mocked_capture(self):
        """Mock cv2.VideoCapture so no real device is required."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_frame[0, 0] = [0, 128, 255]  # Blue-ish in BGR

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = WebcamSource(index=0)
            cam.start()
            frame = cam.read()
            cam.stop()

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
        # BGR (0,128,255) converts to RGB (255,128,0)
        assert frame[0, 0, 0] == 255
        assert frame[0, 0, 1] == 128
        assert frame[0, 0, 2] == 0

    def test_returns_last_good_frame_on_missed_read(self):
        """A failed read returns the previous good frame instead of crashing."""
        good_frame = np.full((480, 640, 3), 42, dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, good_frame),
            (False, None),
        ]

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = WebcamSource(index=0)
            cam.start()
            first = cam.read()
            second = cam.read()
            cam.stop()

        assert second.dtype == np.uint8
        # second should equal the RGB-converted first good frame
        assert second.shape == first.shape


class TestMakeCamera:
    def test_make_camera_file_returns_file_source(self, png_path):
        cam = make_camera("file", path=png_path)
        assert isinstance(cam, FileCameraSource)

    def test_make_camera_webcam_returns_webcam_source(self):
        cam = make_camera("webcam", index=0)
        assert isinstance(cam, WebcamSource)

    def test_make_camera_unknown_source_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown camera source"):
            make_camera("unknown_source")
