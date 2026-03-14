from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExerciseLog:
    sets: int = 0
    total_reps: int = 0


@dataclass
class SessionState:
    active_exercise: Optional[str] = None
    current_rep_count: int = 0
    exercises: dict[str, ExerciseLog] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    last_frame: Optional[Any] = None   # np.ndarray RGB frame, or None
    last_pose: Optional[Any] = None    # pose.types.Pose, or None
