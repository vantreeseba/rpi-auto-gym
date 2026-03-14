import threading
import time
from typing import Optional
from unittest.mock import patch

import pytest

from src.session import ExerciseLog, Session, SessionState
from src.session.classifier_types import ClassifierResult


def make_result(
    exercise: Optional[str] = "squat",
    phase: Optional[str] = "down",
    rep_count: int = 0,
    set_count: int = 0,
) -> ClassifierResult:
    return ClassifierResult(exercise=exercise, phase=phase, rep_count=rep_count, set_count=set_count)


class TestSessionInitialState:
    def test_starts_with_empty_state(self) -> None:
        session = Session()
        state = session.get_state()

        assert state.active_exercise is None
        assert state.current_rep_count == 0
        assert state.exercises == {}


class TestSessionUpdate:
    def test_update_sets_active_exercise(self) -> None:
        session = Session()
        state = session.update(make_result(exercise="squat", rep_count=3))

        assert state.active_exercise == "squat"
        assert state.current_rep_count == 3

    def test_update_with_increasing_set_count_updates_exercise_sets(self) -> None:
        session = Session()
        session.update(make_result(exercise="squat", rep_count=5, set_count=0))
        state = session.update(make_result(exercise="squat", rep_count=0, set_count=1))

        assert "squat" in state.exercises
        assert state.exercises["squat"].sets == 1

    def test_update_accumulates_total_reps(self) -> None:
        session = Session()
        # Complete first set with 8 reps
        session.update(make_result(exercise="squat", rep_count=8, set_count=0))
        session.update(make_result(exercise="squat", rep_count=0, set_count=1))
        # Complete second set with 10 reps
        session.update(make_result(exercise="squat", rep_count=10, set_count=1))
        state = session.update(make_result(exercise="squat", rep_count=0, set_count=2))

        assert state.exercises["squat"].total_reps == 18
        assert state.exercises["squat"].sets == 2

    def test_update_with_none_exercise_does_not_create_log_entry(self) -> None:
        session = Session()
        session.update(make_result(exercise=None, rep_count=5, set_count=0))
        state = session.update(make_result(exercise=None, rep_count=0, set_count=1))

        assert state.exercises == {}


class TestSessionReset:
    def test_reset_clears_all_counts_and_exercises(self) -> None:
        session = Session()
        session.update(make_result(exercise="squat", rep_count=5, set_count=0))
        session.update(make_result(exercise="squat", rep_count=0, set_count=1))

        session.reset()
        state = session.get_state()

        assert state.active_exercise is None
        assert state.current_rep_count == 0
        assert state.exercises == {}

    def test_reset_reinitializes_start_time(self) -> None:
        # __init__ calls monotonic() once, reset() once, get_state()->_snapshot() once
        monotonic_values = iter([0.0, 5.0, 10.0])

        with patch("src.session.session.time.monotonic", side_effect=monotonic_values):
            session = Session()  # start_time = 0.0
            session.reset()     # start_time = 5.0
            state = session.get_state()  # now = 10.0, elapsed = 10.0 - 5.0 = 5.0

        assert state.elapsed_seconds == pytest.approx(5.0)


class TestSessionElapsedTime:
    def test_get_state_returns_positive_elapsed_seconds(self) -> None:
        session = Session()
        time.sleep(0.01)
        state = session.get_state()

        assert state.elapsed_seconds > 0

    def test_elapsed_seconds_computed_from_monotonic(self) -> None:
        monotonic_values = iter([100.0, 105.0])

        with patch("src.session.session.time.monotonic", side_effect=monotonic_values):
            session = Session()  # start_time = 100.0
            state = session.get_state()  # now = 105.0

        assert state.elapsed_seconds == pytest.approx(5.0)


class TestSessionThreadSafety:
    def test_concurrent_updates_produce_no_exceptions(self) -> None:
        session = Session()
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            for i in range(100):
                try:
                    session.update(make_result(
                        exercise="squat",
                        rep_count=i % 10,
                        set_count=i // 10,
                    ))
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_updates_final_state_is_consistent(self) -> None:
        session = Session()

        def worker() -> None:
            for i in range(100):
                session.update(make_result(
                    exercise="squat",
                    rep_count=i % 10,
                    set_count=i // 10,
                ))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        state = session.get_state()
        # State must be internally consistent — sets is non-negative
        if "squat" in state.exercises:
            assert state.exercises["squat"].sets >= 0
            assert state.exercises["squat"].total_reps >= 0
