"""Tests for the pose estimation module."""
import sys
import types
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Stub out mediapipe before importing our module so tests run without the lib
# ---------------------------------------------------------------------------

def _build_mediapipe_stub() -> None:
    """Register a minimal mediapipe stub in sys.modules."""
    if "mediapipe" in sys.modules:
        return  # already stubbed or installed

    mediapipe_mod = types.ModuleType("mediapipe")
    tasks_mod = types.ModuleType("mediapipe.tasks")
    python_mod = types.ModuleType("mediapipe.tasks.python")
    vision_mod = types.ModuleType("mediapipe.tasks.python.vision")

    # BaseOptions stub
    class BaseOptions:
        def __init__(self, model_asset_path: str) -> None:
            self.model_asset_path = model_asset_path

    # ImageFormat stub
    class ImageFormat:
        SRGB = "SRGB"

    # Image stub
    class Image:
        def __init__(self, image_format: object, data: object) -> None:
            self.image_format = image_format
            self.data = data

    # PoseLandmarkerOptions stub
    class PoseLandmarkerOptions:
        def __init__(
            self,
            base_options: object,
            output_segmentation_masks: bool = False,
        ) -> None:
            pass

    # PoseLandmarker stub — create_from_options returns a mock by default
    class PoseLandmarker:
        @staticmethod
        def create_from_options(options: object) -> "PoseLandmarker":
            return PoseLandmarker()

        def detect(self, image: object) -> MagicMock:
            return MagicMock(pose_landmarks=[])

    python_mod.BaseOptions = BaseOptions  # type: ignore[attr-defined]
    vision_mod.Image = Image  # type: ignore[attr-defined]
    vision_mod.ImageFormat = ImageFormat  # type: ignore[attr-defined]
    vision_mod.PoseLandmarkerOptions = PoseLandmarkerOptions  # type: ignore[attr-defined]
    vision_mod.PoseLandmarker = PoseLandmarker  # type: ignore[attr-defined]

    tasks_mod.python = python_mod  # type: ignore[attr-defined]
    mediapipe_mod.tasks = tasks_mod  # type: ignore[attr-defined]

    sys.modules["mediapipe"] = mediapipe_mod
    sys.modules["mediapipe.tasks"] = tasks_mod
    sys.modules["mediapipe.tasks.python"] = python_mod
    sys.modules["mediapipe.tasks.python.vision"] = vision_mod


_build_mediapipe_stub()

# Now safe to import our modules
from src.pose.types import COCO_KEYPOINT_NAMES, Keypoint, Pose  # noqa: E402
from src.pose.mediapipe_estimator import (  # noqa: E402
    BLAZEPOSE_TO_COCO,
    MediaPipePoseEstimator,
)
from src.pose.factory import make_estimator  # noqa: E402


# ---------------------------------------------------------------------------
# Tests: Keypoint and Pose dataclasses
# ---------------------------------------------------------------------------

class TestKeypoint:
    def test_instantiates_with_named_fields(self) -> None:
        kp = Keypoint(x=0.5, y=0.3, confidence=0.9)
        assert kp.x == 0.5
        assert kp.y == 0.3
        assert kp.confidence == 0.9

    def test_accepts_boundary_values(self) -> None:
        kp_zero = Keypoint(x=0.0, y=0.0, confidence=0.0)
        kp_one = Keypoint(x=1.0, y=1.0, confidence=1.0)
        assert kp_zero.confidence == 0.0
        assert kp_one.confidence == 1.0


class TestPose:
    def _make_full_pose(self) -> Pose:
        keypoints = {
            name: Keypoint(x=0.5, y=0.5, confidence=0.8)
            for name in COCO_KEYPOINT_NAMES
        }
        return Pose(keypoints=keypoints, timestamp=time.monotonic())

    def test_instantiates_with_keypoints_and_timestamp(self) -> None:
        pose = self._make_full_pose()
        assert isinstance(pose.keypoints, dict)
        assert isinstance(pose.timestamp, float)

    def test_contains_all_17_coco_keypoint_names(self) -> None:
        pose = self._make_full_pose()
        assert set(pose.keypoints.keys()) == set(COCO_KEYPOINT_NAMES)

    def test_coco_keypoint_names_has_exactly_17_entries(self) -> None:
        assert len(COCO_KEYPOINT_NAMES) == 17


# ---------------------------------------------------------------------------
# Tests: MediaPipePoseEstimator
# ---------------------------------------------------------------------------

def _make_landmark(x: float, y: float, visibility: float) -> MagicMock:
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.visibility = visibility
    return lm


def _make_full_landmark_list() -> list:
    """33 BlazePose landmarks, all with high visibility."""
    return [_make_landmark(0.5, 0.5, 0.9) for _ in range(33)]


class TestMediaPipePoseEstimator:
    def _make_estimator(self) -> MediaPipePoseEstimator:
        # create_from_options is already mocked in the stub
        return MediaPipePoseEstimator(model_path="fake.task", min_confidence=0.5)

    def test_returns_none_when_no_landmarks_detected(self) -> None:
        estimator = self._make_estimator()
        # Default stub returns empty pose_landmarks
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame)
        assert result is None

    def test_returns_pose_when_landmarks_present(self) -> None:
        estimator = self._make_estimator()

        mock_result = MagicMock()
        mock_result.pose_landmarks = [_make_full_landmark_list()]

        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(estimator._landmarker, "detect", return_value=mock_result):
            pose = estimator.estimate(frame)

        assert pose is not None
        assert isinstance(pose, Pose)

    def test_returned_pose_contains_all_17_coco_keypoints(self) -> None:
        estimator = self._make_estimator()

        mock_result = MagicMock()
        mock_result.pose_landmarks = [_make_full_landmark_list()]

        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(estimator._landmarker, "detect", return_value=mock_result):
            pose = estimator.estimate(frame)

        assert pose is not None
        assert set(pose.keypoints.keys()) == set(COCO_KEYPOINT_NAMES)

    def test_returns_none_when_all_confidences_below_threshold(self) -> None:
        estimator = self._make_estimator()

        low_conf_landmarks = [
            _make_landmark(0.5, 0.5, 0.1) for _ in range(33)
        ]
        mock_result = MagicMock()
        mock_result.pose_landmarks = [low_conf_landmarks]

        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(estimator._landmarker, "detect", return_value=mock_result):
            pose = estimator.estimate(frame)

        assert pose is None

    def test_keypoint_coordinates_are_normalized(self) -> None:
        estimator = self._make_estimator()

        landmarks = [_make_landmark(0.3, 0.7, 0.9) for _ in range(33)]
        mock_result = MagicMock()
        mock_result.pose_landmarks = [landmarks]

        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(estimator._landmarker, "detect", return_value=mock_result):
            pose = estimator.estimate(frame)

        assert pose is not None
        nose = pose.keypoints["nose"]
        assert nose.x == pytest.approx(0.3)
        assert nose.y == pytest.approx(0.7)
        assert nose.confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Tests: factory
# ---------------------------------------------------------------------------

class TestMakeEstimator:
    def test_returns_mediapipe_estimator_for_mediapipe_backend(self) -> None:
        estimator = make_estimator("mediapipe", model_path="fake.task")
        assert isinstance(estimator, MediaPipePoseEstimator)

    def test_raises_value_error_for_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            make_estimator("unknown_backend", model_path="fake.task")

    def test_coral_raises_not_implemented_without_pycoral(self) -> None:
        with pytest.raises(NotImplementedError, match="pycoral"):
            make_estimator("coral", model_path="fake.tflite")

    def test_hailo_raises_not_implemented_without_hailort(self) -> None:
        with pytest.raises(NotImplementedError, match="hailort"):
            make_estimator("hailo", model_path="fake.hef")
