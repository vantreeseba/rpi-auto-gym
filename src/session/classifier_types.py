from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassifierResult:
    exercise: Optional[str]
    phase: Optional[str]
    rep_count: int
    set_count: int
