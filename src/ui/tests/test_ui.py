import os
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.ui import WorkoutApp
from src.ui.session_types import ExerciseLog, SessionState


def _make_mock_session(state: Optional[SessionState] = None) -> MagicMock:
    """Return a mock session that returns the given state from get_state()."""
    mock = MagicMock()
    mock.get_state.return_value = state or SessionState()
    return mock


class TestWorkoutAppInstantiation:
    def test_can_instantiate_in_headless_mode(self) -> None:
        with patch.dict(os.environ, {"DISPLAY": ""}, clear=False):
            app = WorkoutApp(_make_mock_session())

        assert app is not None

    def test_run_returns_immediately_in_headless_mode(self) -> None:
        with patch.dict(os.environ, {"DISPLAY": ""}, clear=False):
            app = WorkoutApp(_make_mock_session())
            # If run() blocks, this test times out — returning means headless guard works
            app.run()


class TestUpdateDisplayExerciseName:
    def test_shows_formatted_exercise_name_when_active(self) -> None:
        with patch.dict(os.environ, {"DISPLAY": ""}, clear=False):
            app = WorkoutApp(_make_mock_session())
            state = SessionState(active_exercise="squat")
            app._update_display(state)

        assert app._exercise_label.cget("text") == "Squat"

    def test_shows_em_dash_when_no_active_exercise(self) -> None:
        with patch.dict(os.environ, {"DISPLAY": ""}, clear=False):
            app = WorkoutApp(_make_mock_session())
            state = SessionState(active_exercise=None)
            app._update_display(state)

        assert app._exercise_label.cget("text") == "—"

    def test_formats_snake_case_exercise_name(self) -> None:
        with patch.dict(os.environ, {"DISPLAY": ""}, clear=False):
            app = WorkoutApp(_make_mock_session())
            state = SessionState(active_exercise="jumping_jack")
            app._update_display(state)

        assert app._exercise_label.cget("text") == "Jumping Jack"


class TestUpdateDisplayRepCount:
    def test_shows_rep_count_as_string(self) -> None:
        with patch.dict(os.environ, {"DISPLAY": ""}, clear=False):
            app = WorkoutApp(_make_mock_session())
            state = SessionState(current_rep_count=7)
            app._update_display(state)

        assert app._rep_label.cget("text") == "7"


class TestUpdateDisplayHistory:
    def test_history_label_contains_exercise_name(self) -> None:
        with patch.dict(os.environ, {"DISPLAY": ""}, clear=False):
            app = WorkoutApp(_make_mock_session())
            state = SessionState(
                exercises={"squat": ExerciseLog(sets=2, total_reps=20)}
            )
            app._update_display(state)

        assert "squat" in app._history_label.cget("text")

    def test_history_label_contains_set_and_rep_counts(self) -> None:
        with patch.dict(os.environ, {"DISPLAY": ""}, clear=False):
            app = WorkoutApp(_make_mock_session())
            state = SessionState(
                exercises={"squat": ExerciseLog(sets=2, total_reps=20)}
            )
            app._update_display(state)

        text = app._history_label.cget("text")
        assert "2" in text
        assert "20" in text
