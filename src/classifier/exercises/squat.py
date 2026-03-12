from ..angles import angle_between
from ..pose_types import Keypoint, Pose

_KNEE_DOWN_THRESHOLD = 120.0
_KNEE_UP_THRESHOLD = 160.0
_MISSING_KEYPOINT = Keypoint(x=0.0, y=0.0, confidence=0.0)


class SquatDetector:
    def __init__(self) -> None:
        self._phase = "standing"

    def update(self, pose: Pose) -> tuple[str, bool]:
        """Returns (current_phase, did_complete_rep)."""
        kp = pose.keypoints

        left_knee_angle = angle_between(
            kp.get("left_hip", _MISSING_KEYPOINT),
            kp.get("left_knee", _MISSING_KEYPOINT),
            kp.get("left_ankle", _MISSING_KEYPOINT),
        )
        right_knee_angle = angle_between(
            kp.get("right_hip", _MISSING_KEYPOINT),
            kp.get("right_knee", _MISSING_KEYPOINT),
            kp.get("right_ankle", _MISSING_KEYPOINT),
        )

        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0

        if self._phase == "standing" and avg_knee_angle < _KNEE_DOWN_THRESHOLD:
            self._phase = "down"
            return self._phase, False

        if self._phase == "down" and avg_knee_angle > _KNEE_UP_THRESHOLD:
            self._phase = "standing"
            return self._phase, True

        return self._phase, False

    def reset(self) -> None:
        self._phase = "standing"
