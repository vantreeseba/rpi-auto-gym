from dataclasses import dataclass, field


@dataclass
class ExerciseLog:
    sets: int = 0
    total_reps: int = 0


@dataclass
class SessionState:
    active_exercise: str | None = None
    current_rep_count: int = 0
    exercises: dict[str, ExerciseLog] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
