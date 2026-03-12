from ..pose_types import Keypoint, Pose

_MISSING_KEYPOINT = Keypoint(x=0.0, y=0.0, confidence=0.0)
_MIN_CONFIDENCE = 0.3


def _is_visible(kp: Keypoint) -> bool:
    return kp.confidence >= _MIN_CONFIDENCE


class JumpingJackDetector:
    def __init__(self) -> None:
        self._phase = "closed"

    def update(self, pose: Pose) -> tuple[str, bool]:
        """Returns (current_phase, did_complete_rep)."""
        kp = pose.keypoints

        left_wrist = kp.get("left_wrist", _MISSING_KEYPOINT)
        right_wrist = kp.get("right_wrist", _MISSING_KEYPOINT)
        left_shoulder = kp.get("left_shoulder", _MISSING_KEYPOINT)
        right_shoulder = kp.get("right_shoulder", _MISSING_KEYPOINT)
        left_hip = kp.get("left_hip", _MISSING_KEYPOINT)
        right_hip = kp.get("right_hip", _MISSING_KEYPOINT)

        all_visible = all(_is_visible(k) for k in (
            left_wrist, right_wrist, left_shoulder, right_shoulder, left_hip, right_hip
        ))

        if not all_visible:
            return self._phase, False

        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        avg_hip_y = (left_hip.y + right_hip.y) / 2.0

        # In image coordinates, smaller y = higher on screen
        wrists_above_shoulders = (
            left_wrist.y < avg_shoulder_y and right_wrist.y < avg_shoulder_y
        )
        wrists_below_hips = (
            left_wrist.y > avg_hip_y and right_wrist.y > avg_hip_y
        )

        if self._phase == "closed" and wrists_above_shoulders:
            self._phase = "open"
            return self._phase, False

        if self._phase == "open" and wrists_below_hips:
            self._phase = "closed"
            return self._phase, True

        return self._phase, False

    def reset(self) -> None:
        self._phase = "closed"
