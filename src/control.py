"""Local control API for the running app (the zynthian "CUIA" idea).

The app listens on 127.0.0.1 only. The web interface (``src.webconf``) runs as
a separate process so it survives app restarts, and talks to the app through
this API:

    GET  /state        session state as JSON
    GET  /frame.jpg    latest camera frame with the pose skeleton drawn
    POST /action/<n>   one of ACTIONS

Actions are queued, not run on the HTTP thread: session resets happen in the
capture thread and UI changes happen on the Tk thread.
"""
import json
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from .config import EXIT_RESTART, EXIT_STOP

log = logging.getLogger(__name__)

ACTIONS = ("reset_session", "toggle_debug", "restart", "stop")


class AppControl:
    """Shared between the capture thread, the UI thread and the HTTP server."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.reset_requested = threading.Event()
        self.stop_event = threading.Event()
        self.ui_commands: "queue.Queue[str]" = queue.Queue()
        self.exit_code = EXIT_STOP
        self.started_at = time.time()
        self._fps = 0.0
        self._frames = 0
        self._fps_window_start = time.monotonic()

    # -- called by the capture loop ------------------------------------------
    def frame_done(self) -> None:
        self._frames += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._fps = self._frames / elapsed
            self._frames = 0
            self._fps_window_start = now

    # -- actions -------------------------------------------------------------
    def perform(self, action: str) -> None:
        if action not in ACTIONS:
            raise ValueError(f"unknown action '{action}'")
        log.info("Action: %s", action)
        if action == "reset_session":
            self.reset_requested.set()
        elif action == "toggle_debug":
            self.ui_commands.put("toggle_debug")
        elif action in ("restart", "stop"):
            self.request_exit(EXIT_RESTART if action == "restart" else EXIT_STOP)

    def request_exit(self, code: int) -> None:
        self.exit_code = code
        self.stop_event.set()
        self.ui_commands.put("quit")

    # -- read side -----------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        state = self.session.get_state()
        active_sets = 0
        if state.active_exercise and state.active_exercise in state.exercises:
            active_sets = state.exercises[state.active_exercise].sets
        return {
            "active_exercise": state.active_exercise,
            "current_rep_count": state.current_rep_count,
            "active_sets": active_sets,
            "elapsed_seconds": round(state.elapsed_seconds, 1),
            "exercises": {
                name: {"sets": entry.sets, "total_reps": entry.total_reps}
                for name, entry in state.exercises.items()
            },
            "pose_detected": state.last_pose is not None,
            "has_frame": state.last_frame is not None,
            "fps": round(self._fps, 1),
            "uptime_seconds": round(time.time() - self.started_at),
        }

    def frame_jpeg(self, quality: int = 70) -> Optional[bytes]:
        state = self.session.get_state()
        if state.last_frame is None:
            return None
        import cv2  # local: keeps this module importable without OpenCV

        from .ui.skeleton import draw_skeleton

        rendered = draw_skeleton(state.last_frame, state.last_pose)
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        return buf.tobytes() if ok else None


def _make_handler(control: AppControl):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            log.debug("control: " + fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: Any) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path == "/state":
                self._json(200, control.state_dict())
            elif self.path.startswith("/frame.jpg"):
                jpg = control.frame_jpeg()
                if jpg is None:
                    self._json(404, {"error": "no frame yet"})
                else:
                    self._send(200, jpg, "image/jpeg")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.startswith("/action/"):
                name = self.path[len("/action/"):]
                try:
                    control.perform(name)
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                    return
                self._json(200, {"ok": True, "action": name})
            else:
                self._json(404, {"error": "not found"})

    return Handler


def start_control_server(control: AppControl, port: int) -> Optional[ThreadingHTTPServer]:
    """Start the API on 127.0.0.1:port in a daemon thread. Returns None if the port is busy."""
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(control))
    except OSError as exc:
        log.warning("Control API disabled, can't bind 127.0.0.1:%d (%s)", port, exc)
        return None
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="control-api", daemon=True).start()
    log.info("Control API on http://127.0.0.1:%d", port)
    return server
