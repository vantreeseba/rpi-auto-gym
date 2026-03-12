from dataclasses import dataclass


@dataclass
class ClassifierResult:
    exercise: str | None   # "squat" | "pushup" | "jumping_jack" | None
    phase: str | None      # exercise-specific phase e.g. "up" | "down" | "open" | "closed"
    rep_count: int         # reps in the current set
    set_count: int         # sets completed this session
