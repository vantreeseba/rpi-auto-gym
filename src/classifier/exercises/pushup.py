from ..angles import angle_between
from ..pose_types import Keypoint, Pose

_ELBOW_DOWN_THRESHOLD = 90.0
_ELBOW_UP_THRESHOLD = 150.0
_MISSING_KEYPOINT = Keypoint(x=0.0, y=0.0, confidence=0.0)


class PushupDetector:
    def __init__(self) -> None:
        self._phase = "up"

    def update(self, pose: Pose) -> tuple[str, bool]:
        """Returns (current_phase, did_complete_rep)."""
        kp = pose.keypoints

        left_elbow_angle = angle_between(
            kp.get("left_shoulder", _MISSING_KEYPOINT),
            kp.get("left_elbow", _MISSING_KEYPOINT),
            kp.get("left_wrist", _MISSING_KEYPOINT),
        )
        right_elbow_angle = angle_between(
            kp.get("right_shoulder", _MISSING_KEYPOINT),
            kp.get("right_elbow", _MISSING_KEYPOINT),
            kp.get("right_wrist", _MISSING_KEYPOINT),
        )

        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2.0

        if self._phase == "up" and avg_elbow_angle < _ELBOW_DOWN_THRESHOLD:
            self._phase = "down"
            return self._phase, False

        if self._phase == "down" and avg_elbow_angle > _ELBOW_UP_THRESHOLD:
            self._phase = "up"
            return self._phase, True

        return self._phase, False

    def reset(self) -> None:
        self._phase = "up"
