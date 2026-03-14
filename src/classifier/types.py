from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassifierResult:
    exercise: Optional[str]   # "squat" | "pushup" | "jumping_jack" | None
    phase: Optional[str]      # exercise-specific phase e.g. "up" | "down" | "open" | "closed"
    rep_count: int         # reps in the current set
    set_count: int         # sets completed this session
