import math
from .pose_types import Keypoint

_MISSING_CONFIDENCE_THRESHOLD = 0.3
_NEUTRAL_ANGLE = 180.0


def angle_between(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
    """
    Returns the angle at joint b, formed by points a–b–c, in degrees.
    Returns 180.0 if any keypoint has confidence < 0.3 (treat as missing).
    """
    if any(k.confidence < _MISSING_CONFIDENCE_THRESHOLD for k in (a, b, c)):
        return _NEUTRAL_ANGLE

    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    degrees = abs(math.degrees(radians))

    # Clamp to [0, 180] — angles wrap around, take the interior angle
    return min(degrees, 360.0 - degrees)
