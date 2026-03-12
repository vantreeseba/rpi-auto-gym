"""Tests for the exercise classifier component."""
import math
from unittest.mock import patch

import pytest

from ..angles import angle_between
from ..pose_types import Keypoint, Pose
from ..exercises.squat import SquatDetector
from ..exercises.pushup import PushupDetector
from ..exercises.jumping_jack import JumpingJackDetector
from ..classifier import ExerciseClassifier


# ---------------------------------------------------------------------------
# Pose builder helpers
# ---------------------------------------------------------------------------

def make_keypoint(x: float, y: float, confidence: float = 1.0) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=confidence)


def make_pose(keypoints: dict[str, tuple[float, float]], timestamp: float = 0.0) -> Pose:
    """Build a Pose from a dict of name -> (x, y) with full confidence."""
    return Pose(
        keypoints={name: make_keypoint(x, y) for name, (x, y) in keypoints.items()},
        timestamp=timestamp,
    )


def make_pose_low_conf(keypoints: dict[str, tuple[float, float]], timestamp: float = 0.0) -> Pose:
    """Build a Pose where all keypoints have confidence 0.1 (below threshold)."""
    return Pose(
        keypoints={name: make_keypoint(x, y, confidence=0.1) for name, (x, y) in keypoints.items()},
        timestamp=timestamp,
    )


# Helpers to build poses that produce specific joint angles
# We place joints on a unit circle to get exact angles.

def _keypoints_for_angle(angle_deg: float) -> tuple[Keypoint, Keypoint, Keypoint]:
    """
    Returns (a, b, c) such that angle_between(a, b, c) ≈ angle_deg.
    Joint b is at origin; a is to the right; c is rotated by angle_deg from a.
    """
    b = Keypoint(x=0.0, y=0.0, confidence=1.0)
    a = Keypoint(x=1.0, y=0.0, confidence=1.0)
    rad = math.radians(angle_deg)
    c = Keypoint(x=math.cos(rad), y=math.sin(rad), confidence=1.0)
    return a, b, c


# ---------------------------------------------------------------------------
# angle_between tests
# ---------------------------------------------------------------------------

class TestAngleBetween:
    def test_right_angle_is_90_degrees(self):
        a, b, c = _keypoints_for_angle(90.0)
        result = angle_between(a, b, c)
        assert abs(result - 90.0) < 0.001

    def test_straight_angle_is_180_degrees(self):
        a, b, c = _keypoints_for_angle(180.0)
        result = angle_between(a, b, c)
        assert abs(result - 180.0) < 0.001

    def test_45_degree_angle(self):
        a, b, c = _keypoints_for_angle(45.0)
        result = angle_between(a, b, c)
        assert abs(result - 45.0) < 0.001

    def test_returns_180_when_a_has_low_confidence(self):
        a, b, c = _keypoints_for_angle(90.0)
        low_conf_a = Keypoint(x=a.x, y=a.y, confidence=0.2)
        assert angle_between(low_conf_a, b, c) == 180.0

    def test_returns_180_when_b_has_low_confidence(self):
        a, b, c = _keypoints_for_angle(90.0)
        low_conf_b = Keypoint(x=b.x, y=b.y, confidence=0.1)
        assert angle_between(a, low_conf_b, c) == 180.0

    def test_returns_180_when_c_has_low_confidence(self):
        a, b, c = _keypoints_for_angle(90.0)
        low_conf_c = Keypoint(x=c.x, y=c.y, confidence=0.0)
        assert angle_between(a, b, low_conf_c) == 180.0

    def test_threshold_boundary_exactly_03_is_valid(self):
        """Confidence exactly 0.3 should not trigger missing-keypoint fallback."""
        a, b, c = _keypoints_for_angle(90.0)
        boundary_a = Keypoint(x=a.x, y=a.y, confidence=0.3)
        result = angle_between(boundary_a, b, c)
        assert abs(result - 90.0) < 0.001


# ---------------------------------------------------------------------------
# SquatDetector tests
# ---------------------------------------------------------------------------

def _joint_angle_keypoints(angle_deg: float) -> tuple[Keypoint, Keypoint, Keypoint]:
    """
    Returns (a, b, c) where b is the joint and angle_between(a, b, c) == angle_deg.
    b is at origin; a is directly above (negative y = up in image coords).
    """
    b = Keypoint(0.0, 0.0, 1.0)
    a = Keypoint(0.0, -1.0, 1.0)
    base_angle = math.atan2(-1.0, 0.0)  # direction of vector b→a
    c_angle = base_angle + math.radians(angle_deg)
    c = Keypoint(math.cos(c_angle), math.sin(c_angle), 1.0)
    return a, b, c


def _squat_pose_with_knee_angle(knee_angle_deg: float) -> Pose:
    """Builds a squat pose where both knees have the specified angle."""
    hip, knee, ankle = _joint_angle_keypoints(knee_angle_deg)

    return Pose(
        keypoints={
            "left_hip": hip,
            "left_knee": knee,
            "left_ankle": ankle,
            "right_hip": hip,
            "right_knee": knee,
            "right_ankle": ankle,
        },
        timestamp=0.0,
    )


class TestSquatDetector:
    def test_rep_counted_after_down_then_up(self):
        detector = SquatDetector()
        # Start standing, go down (angle < 120), come back up (angle > 160)
        _, rep = detector.update(_squat_pose_with_knee_angle(100.0))
        assert not rep
        phase, rep = detector.update(_squat_pose_with_knee_angle(170.0))
        assert rep is True
        assert phase == "standing"

    def test_no_rep_without_going_down_first(self):
        detector = SquatDetector()
        _, rep = detector.update(_squat_pose_with_knee_angle(170.0))
        assert not rep

    def test_phase_is_down_when_below_threshold(self):
        detector = SquatDetector()
        phase, _ = detector.update(_squat_pose_with_knee_angle(100.0))
        assert phase == "down"

    def test_reset_clears_state(self):
        detector = SquatDetector()
        detector.update(_squat_pose_with_knee_angle(100.0))
        detector.reset()
        # After reset, going up should NOT count as a rep (state is back to "standing")
        _, rep = detector.update(_squat_pose_with_knee_angle(170.0))
        assert not rep


# ---------------------------------------------------------------------------
# PushupDetector tests
# ---------------------------------------------------------------------------

def _pushup_pose_with_elbow_angle(elbow_angle_deg: float) -> Pose:
    """Builds a pushup pose where both elbows have the specified angle."""
    shoulder, elbow, wrist = _joint_angle_keypoints(elbow_angle_deg)

    return Pose(
        keypoints={
            "left_shoulder": shoulder,
            "left_elbow": elbow,
            "left_wrist": wrist,
            "right_shoulder": shoulder,
            "right_elbow": elbow,
            "right_wrist": wrist,
        },
        timestamp=0.0,
    )


class TestPushupDetector:
    def test_rep_counted_after_down_then_up(self):
        detector = PushupDetector()
        # Start in "up", go down (angle < 90), come back up (angle > 150)
        _, rep = detector.update(_pushup_pose_with_elbow_angle(80.0))
        assert not rep
        phase, rep = detector.update(_pushup_pose_with_elbow_angle(160.0))
        assert rep is True
        assert phase == "up"

    def test_no_rep_without_going_down_first(self):
        detector = PushupDetector()
        _, rep = detector.update(_pushup_pose_with_elbow_angle(160.0))
        assert not rep

    def test_phase_is_down_when_below_threshold(self):
        detector = PushupDetector()
        phase, _ = detector.update(_pushup_pose_with_elbow_angle(80.0))
        assert phase == "down"

    def test_reset_clears_state(self):
        detector = PushupDetector()
        detector.update(_pushup_pose_with_elbow_angle(80.0))
        detector.reset()
        _, rep = detector.update(_pushup_pose_with_elbow_angle(160.0))
        assert not rep


# ---------------------------------------------------------------------------
# JumpingJackDetector tests
# ---------------------------------------------------------------------------

def _jj_pose_open() -> Pose:
    """Wrists are above shoulders — 'open' position."""
    return Pose(
        keypoints={
            "left_shoulder": Keypoint(0.0, 5.0, 1.0),
            "right_shoulder": Keypoint(1.0, 5.0, 1.0),
            "left_hip": Keypoint(0.0, 8.0, 1.0),
            "right_hip": Keypoint(1.0, 8.0, 1.0),
            # Wrists above shoulders (smaller y in image coords = higher)
            "left_wrist": Keypoint(-1.0, 2.0, 1.0),
            "right_wrist": Keypoint(2.0, 2.0, 1.0),
        },
        timestamp=0.0,
    )


def _jj_pose_closed() -> Pose:
    """Wrists are below hips — 'closed' position."""
    return Pose(
        keypoints={
            "left_shoulder": Keypoint(0.0, 5.0, 1.0),
            "right_shoulder": Keypoint(1.0, 5.0, 1.0),
            "left_hip": Keypoint(0.0, 8.0, 1.0),
            "right_hip": Keypoint(1.0, 8.0, 1.0),
            # Wrists below hips (larger y = lower in image)
            "left_wrist": Keypoint(0.3, 10.0, 1.0),
            "right_wrist": Keypoint(0.7, 10.0, 1.0),
        },
        timestamp=0.0,
    )


class TestJumpingJackDetector:
    def test_rep_counted_after_open_then_closed(self):
        detector = JumpingJackDetector()
        _, rep = detector.update(_jj_pose_open())
        assert not rep
        phase, rep = detector.update(_jj_pose_closed())
        assert rep is True
        assert phase == "closed"

    def test_no_rep_without_opening_first(self):
        detector = JumpingJackDetector()
        _, rep = detector.update(_jj_pose_closed())
        assert not rep

    def test_phase_is_open_when_wrists_above_shoulders(self):
        detector = JumpingJackDetector()
        phase, _ = detector.update(_jj_pose_open())
        assert phase == "open"

    def test_reset_clears_state(self):
        detector = JumpingJackDetector()
        detector.update(_jj_pose_open())
        detector.reset()
        _, rep = detector.update(_jj_pose_closed())
        assert not rep


# ---------------------------------------------------------------------------
# ExerciseClassifier tests
# ---------------------------------------------------------------------------

class TestExerciseClassifier:
    def test_reset_zeroes_all_counts(self):
        classifier = ExerciseClassifier()
        # Perform a squat rep
        classifier.update(_squat_pose_with_knee_angle(100.0))
        classifier.update(_squat_pose_with_knee_angle(170.0))

        classifier.reset()
        result = classifier.update(_squat_pose_with_knee_angle(170.0))
        assert result.rep_count == 0
        assert result.set_count == 0

    def test_set_auto_completes_after_pause(self):
        classifier = ExerciseClassifier(set_pause_seconds=3.0)

        with patch("classifier.classifier.time.monotonic") as mock_time:
            mock_time.return_value = 0.0

            # Perform 2 squat reps
            classifier.update(_squat_pose_with_knee_angle(100.0))
            mock_time.return_value = 0.1
            classifier.update(_squat_pose_with_knee_angle(170.0))  # rep 1

            mock_time.return_value = 0.2
            classifier.update(_squat_pose_with_knee_angle(100.0))
            mock_time.return_value = 0.3
            classifier.update(_squat_pose_with_knee_angle(170.0))  # rep 2

            # Advance time past the pause threshold
            mock_time.return_value = 4.0

            # Trigger set completion via update
            result = classifier.update(_squat_pose_with_knee_angle(170.0))

        assert result.set_count == 1
        assert result.rep_count == 0

    def test_active_exercise_set_on_first_rep(self):
        classifier = ExerciseClassifier()
        classifier.update(_squat_pose_with_knee_angle(100.0))
        result = classifier.update(_squat_pose_with_knee_angle(170.0))
        assert result.exercise == "squat"
        assert result.rep_count == 1

    def test_rep_count_increments_correctly(self):
        classifier = ExerciseClassifier()

        def do_squat_rep() -> None:
            classifier.update(_squat_pose_with_knee_angle(100.0))
            classifier.update(_squat_pose_with_knee_angle(170.0))

        do_squat_rep()
        do_squat_rep()
        result = classifier.update(_squat_pose_with_knee_angle(170.0))
        assert result.rep_count == 2

    def test_initial_state_has_no_exercise(self):
        classifier = ExerciseClassifier()
        result = classifier.update(_squat_pose_with_knee_angle(170.0))
        assert result.exercise is None
        assert result.rep_count == 0
        assert result.set_count == 0
