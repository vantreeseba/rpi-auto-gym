from pathlib import Path

from src.config import Config
from src.ui import error_screen


def test_last_error_prefers_exception_line(tmp_path: Path) -> None:
    log = tmp_path / "autogym.log"
    log.write_text(
        "2026-01-01 INFO [autogym] autogym: starting\n"
        "2026-01-01 ERROR [autogym] autogym: Startup failed\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "FileNotFoundError: models/pose_landmarker_lite.task\n"
    )
    assert error_screen.last_error(log) == "FileNotFoundError: models/pose_landmarker_lite.task"


def test_last_error_falls_back_to_error_line(tmp_path: Path) -> None:
    log = tmp_path / "autogym.log"
    log.write_text("2026-01-01 ERROR [autogym] autogym: Capture loop crashed\n")
    assert error_screen.last_error(log) == "Capture loop crashed"


def test_last_error_missing_file(tmp_path: Path) -> None:
    assert error_screen.last_error(tmp_path / "nope.log") == ""


def test_message_includes_web_address(monkeypatch) -> None:
    monkeypatch.setattr(error_screen, "last_error", lambda: "Boom")
    monkeypatch.setattr(error_screen.netinfo, "hostname", lambda: "autogym")
    monkeypatch.setattr(error_screen.netinfo, "local_ip", lambda: "192.168.1.50")
    lines = error_screen.message_lines(1, 2, Config({}))
    assert lines[0] == "Exit code 1, attempt 2"
    assert "Boom" in lines
    assert "http://autogym.local:8080" in lines
    assert "http://192.168.1.50:8080" in lines
