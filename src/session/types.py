# Re-export from the canonical definition to avoid duplication.
# Both the session layer and the UI protocol share the same SessionState type.
from src.ui.session_types import ExerciseLog, SessionState

__all__ = ["ExerciseLog", "SessionState"]
