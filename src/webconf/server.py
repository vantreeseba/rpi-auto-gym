"""Web interface for power users (the zynthian "webconf" idea).

Runs as its own process so it stays up while the app restarts:

    python -m src.webconf            # serves on AUTOGYM_WEB_HOST:AUTOGYM_WEB_PORT

It edits the envars file and talks to the running app through the local
control API in ``src/control.py``. Standard library only.
"""
import base64
import hmac
import json
import logging
import mimetypes
import platform
import subprocess
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .. import config as conf
from .. import hardware, netinfo
from ..control import ACTIONS

log = logging.getLogger("autogym.webconf")

STATIC_DIR = Path(__file__).resolve().parent / "static"
PASSWORD_MASK = "********"


def file_values() -> Dict[str, str]:
    """Defaults plus the envars file. Ignores this process's own environment."""
    values = {s.key: s.default for s in conf.SETTINGS}
    values.update(conf.read_envars_file())
    return values


def schema() -> list:
    return [
        {"key": s.key, "section": s.section, "label": s.label, "default": s.default,
         "type": s.type, "choices": list(s.choices), "help": s.help}
        for s in conf.SETTINGS
    ]


def app_request(method: str, path: str, timeout: float = 1.5) -> Tuple[int, bytes, str]:
    """Call the app's control API. Returns (status, body, content-type); 503 if unreachable."""
    port = conf.Config(file_values()).get_int("AUTOGYM_CONTROL_PORT")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "application/json")
    except (urllib.error.URLError, OSError):
        return 503, json.dumps({"running": False, "error": "app not running"}).encode(), "application/json"


def tail_log(lines: int) -> str:
    path = conf.log_path()
    if not path.is_file():
        return f"(no log file yet at {path})"
    with path.open(errors="replace") as fh:
        return "".join(deque(fh, maxlen=lines))


def git_version() -> str:
    try:
        out = subprocess.run(["git", "-C", str(conf.repo_dir()), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=2)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def info() -> Dict[str, Any]:
    cfg = conf.Config(file_values())
    return {
        "hostname": netinfo.hostname(),
        "ip": netinfo.local_ip() or "offline",
        "web_urls": netinfo.web_urls(cfg.get_int("AUTOGYM_WEB_PORT"), cfg.get("AUTOGYM_WEB_HOST")),
        "version": git_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_path": str(conf.envars_path()),
        "log_path": str(conf.log_path()),
    }


def hardware_report() -> Dict[str, Any]:
    detected = hardware.detect()
    return {
        "detected": detected.to_dict(),
        "recommended": hardware.recommend(detected),
        "problems": hardware.problems(detected, file_values()),
        "presets": [{"id": p.id, "name": p.name, "description": p.description, "settings": p.settings}
                    for p in hardware.presets()],
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "autogym-webconf"

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("%s " + fmt, self.address_string(), *args)

    # -- helpers ---------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, data: Any) -> None:
        self._send(code, json.dumps(data).encode(), "application/json")

    def _authorized(self) -> bool:
        password = file_values().get("AUTOGYM_WEB_PASSWORD", "")
        if not password:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                _, _, given = base64.b64decode(header[6:]).decode().partition(":")
                if hmac.compare_digest(given.encode(), password.encode()):
                    return True
            except (ValueError, UnicodeDecodeError):
                pass
        self._send(401, b"Login required", "text/plain",
                   {"WWW-Authenticate": 'Basic realm="rpi-auto-gym"'})
        return False

    def _read_json(self) -> Optional[Any]:
        # Requiring a JSON content type blocks simple cross-site form posts.
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self._json(415, {"error": "expected application/json"})
            return None
        length = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return None

    def _static(self, rel: str) -> None:
        path = (STATIC_DIR / rel).resolve()
        if STATIC_DIR not in path.parents or not path.is_file():
            self._json(404, {"error": "not found"})
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    # -- routes ----------------------------------------------------------------
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        if not self._authorized():
            return
        url = urlparse(self.path)
        route = url.path
        if route in ("/", "/index.html"):
            self._static("index.html")
        elif route.startswith("/static/"):
            self._static(route[len("/static/"):])
        elif route == "/api/schema":
            self._json(200, schema())
        elif route == "/api/config":
            values = file_values()
            if values.get("AUTOGYM_WEB_PASSWORD"):
                values["AUTOGYM_WEB_PASSWORD"] = PASSWORD_MASK
            self._json(200, {"values": values, "path": str(conf.envars_path()),
                             "file_exists": conf.envars_path().is_file()})
        elif route == "/api/info":
            self._json(200, info())
        elif route == "/api/hardware":
            self._json(200, hardware_report())
        elif route == "/api/state":
            code, body, ctype = app_request("GET", "/state")
            self._send(code, body, ctype)
        elif route == "/api/frame.jpg":
            code, body, ctype = app_request("GET", "/frame.jpg")
            self._send(code, body, ctype)
        elif route == "/api/logs":
            try:
                lines = int(parse_qs(url.query).get("lines", ["300"])[0])
            except ValueError:
                lines = 300
            self._send(200, tail_log(max(10, min(lines, 5000))).encode(), "text/plain; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        route = urlparse(self.path).path
        body = self._read_json()
        if body is None:
            return
        if route == "/api/config":
            self._save_config(body)
        elif route == "/api/hardware/apply":
            self._apply_preset(body)
        elif route.startswith("/api/action/"):
            name = route[len("/api/action/"):]
            if name not in ACTIONS:
                self._json(400, {"error": f"unknown action '{name}'"})
                return
            code, resp, ctype = app_request("POST", f"/action/{name}")
            self._send(code, resp, ctype)
        else:
            self._json(404, {"error": "not found"})

    def _apply_preset(self, body: Dict[str, Any]) -> None:
        preset_id = body.get("preset", "")
        if preset_id == "recommended":
            settings = hardware.recommend(hardware.detect())
        else:
            preset = hardware.preset_by_id(preset_id)
            if preset is None:
                self._json(400, {"error": f"unknown preset '{preset_id}'"})
                return
            settings = preset.settings
        self._save_config({"values": settings, "restart": body.get("restart", False)})

    def _save_config(self, body: Dict[str, Any]) -> None:
        submitted = dict(body.get("values") or {})
        if submitted.get("AUTOGYM_WEB_PASSWORD") == PASSWORD_MASK:
            del submitted["AUTOGYM_WEB_PASSWORD"]  # unchanged
        clean, errors = conf.validate(submitted)
        if errors:
            self._json(400, {"ok": False, "errors": errors})
            return
        path = conf.write_envars_file(clean)
        log.info("Saved %d settings to %s", len(clean), path)
        result: Dict[str, Any] = {"ok": True, "path": str(path), "restarted": False}
        if body.get("restart"):
            code, _, _ = app_request("POST", "/action/restart")
            result["restarted"] = code == 200
        self._json(200, result)


def serve(host: Optional[str] = None, port: Optional[int] = None) -> None:
    cfg = conf.Config()
    conf.setup_logging(cfg.get("AUTOGYM_LOG_LEVEL"), name="webconf")
    host = host or cfg.get("AUTOGYM_WEB_HOST")
    port = port or cfg.get_int("AUTOGYM_WEB_PORT")
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    log.info("Web interface on %s (config: %s)",
             " or ".join(netinfo.web_urls(port, host)), conf.envars_path())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
