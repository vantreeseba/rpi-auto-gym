from typing import Protocol

from .session_types import SessionState


class SessionProtocol(Protocol):
    def get_state(self) -> SessionState: ...
