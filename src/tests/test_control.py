import json
import urllib.request

import numpy as np

from src.config import EXIT_RESTART
from src.control import AppControl, start_control_server
from src.session.session import Session
from src.session.classifier_types import ClassifierResult


def _get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
        return r.status, r.read(), r.headers["Content-Type"]


def _post(port: int, path: str):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=2) as r:
        return r.status, json.loads(r.read())


def test_actions_are_queued() -> None:
    control = AppControl(Session())
    control.perform("reset_session")
    control.perform("toggle_debug")
    assert control.reset_requested.is_set()
    assert control.ui_commands.get_nowait() == "toggle_debug"
    control.perform("restart")
    assert control.exit_code == EXIT_RESTART
    assert control.stop_event.is_set()
    assert control.ui_commands.get_nowait() == "quit"


def test_http_api() -> None:
    session = Session()
    session.update(ClassifierResult(exercise="squat", phase="down", rep_count=4, set_count=0))
    control = AppControl(session)
    server = start_control_server(control, 0)
    assert server is not None
    port = server.server_address[1]
    try:
        status, body, _ = _get(port, "/state")
        state = json.loads(body)
        assert status == 200
        assert state["active_exercise"] == "squat"
        assert state["current_rep_count"] == 4

        try:
            _get(port, "/frame.jpg")
            raise AssertionError("expected 404 before any frame")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

        session.update_frame(np.zeros((48, 64, 3), dtype=np.uint8), None)
        status, body, ctype = _get(port, "/frame.jpg")
        assert status == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8"

        status, body = _post(port, "/action/reset_session")
        assert status == 200 and control.reset_requested.is_set()
    finally:
        server.shutdown()
