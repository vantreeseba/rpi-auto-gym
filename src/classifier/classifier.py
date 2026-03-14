import time
from typing import Optional

from .types import ClassifierResult
from .pose_types import Pose
from .exercises.squat import SquatDetector
from .exercises.pushup import PushupDetector
from .exercises.jumping_jack import JumpingJackDetector


class ExerciseClassifier:
    def __init__(self, set_pause_seconds: float = 3.0) -> None:
        self._set_pause_seconds = set_pause_seconds
        self._squat = SquatDetector()
        self._pushup = PushupDetector()
        self._jumping_jack = JumpingJackDetector()

        self._active_exercise: Optional[str] = None
        self._active_phase: Optional[str] = None
        self._rep_count = 0
        self._set_count = 0
        self._last_rep_time: Optional[float] = None

    def update(self, pose: Pose) -> ClassifierResult:
        self._maybe_complete_set()

        squat_phase, squat_rep = self._squat.update(pose)
        pushup_phase, pushup_rep = self._pushup.update(pose)
        jj_phase, jj_rep = self._jumping_jack.update(pose)

        if squat_rep:
            self._record_rep("squat", squat_phase)
        elif pushup_rep:
            self._record_rep("pushup", pushup_phase)
        elif jj_rep:
            self._record_rep("jumping_jack", jj_phase)
        else:
            self._active_phase = self._current_phase(squat_phase, pushup_phase, jj_phase)

        return ClassifierResult(
            exercise=self._active_exercise,
            phase=self._active_phase,
            rep_count=self._rep_count,
            set_count=self._set_count,
        )

    def reset(self) -> None:
        self._squat.reset()
        self._pushup.reset()
        self._jumping_jack.reset()
        self._active_exercise = None
        self._active_phase = None
        self._rep_count = 0
        self._set_count = 0
        self._last_rep_time = None

    def _record_rep(self, exercise: str, phase: str) -> None:
        self._active_exercise = exercise
        self._active_phase = phase
        self._rep_count += 1
        self._last_rep_time = time.monotonic()

    def _maybe_complete_set(self) -> None:
        if self._last_rep_time is None or self._rep_count == 0:
            return
        elapsed = time.monotonic() - self._last_rep_time
        if elapsed >= self._set_pause_seconds:
            self._set_count += 1
            self._rep_count = 0
            self._last_rep_time = None

    def _current_phase(
        self, squat_phase: str, pushup_phase: str, jj_phase: str
    ) -> Optional[str]:
        if self._active_exercise == "squat":
            return squat_phase
        if self._active_exercise == "pushup":
            return pushup_phase
        if self._active_exercise == "jumping_jack":
            return jj_phase
        return None
