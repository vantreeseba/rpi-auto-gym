import threading
import time

from .classifier_types import ClassifierResult
from .types import ExerciseLog, SessionState


class Session:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._exercises: dict[str, ExerciseLog] = {}
        self._active_exercise: str | None = None
        self._current_rep_count: int = 0
        self._last_set_count: int = 0
        self._last_rep_count: int = 0

    def update(self, result: ClassifierResult) -> SessionState:
        """Thread-safe. Updates internal state from classifier output and returns current SessionState."""
        with self._lock:
            exercise = result.exercise

            if exercise is not None and result.set_count > self._last_set_count:
                log = self._exercises.setdefault(exercise, ExerciseLog())
                log.sets += 1
                log.total_reps += self._last_rep_count

            self._active_exercise = exercise
            self._current_rep_count = result.rep_count
            self._last_set_count = result.set_count
            self._last_rep_count = result.rep_count

            return self._snapshot()

    def reset(self) -> None:
        """Clears all session data and resets the start time."""
        with self._lock:
            self._exercises = {}
            self._active_exercise = None
            self._current_rep_count = 0
            self._last_set_count = 0
            self._last_rep_count = 0
            self._start_time = time.monotonic()

    def get_state(self) -> SessionState:
        """Returns a snapshot of the current SessionState."""
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> SessionState:
        """Build a SessionState from current internal state (must be called under lock)."""
        return SessionState(
            active_exercise=self._active_exercise,
            current_rep_count=self._current_rep_count,
            exercises={name: ExerciseLog(log.sets, log.total_reps) for name, log in self._exercises.items()},
            elapsed_seconds=time.monotonic() - self._start_time,
        )
