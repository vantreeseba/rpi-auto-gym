from dataclasses import dataclass


@dataclass
class ClassifierResult:
    exercise: str | None
    phase: str | None
    rep_count: int
    set_count: int
