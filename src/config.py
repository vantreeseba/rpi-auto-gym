"""App configuration, stored as a shell-style envars file (zynthian-style).

Every setting is an ``AUTOGYM_*`` variable. Values are resolved in this order,
later ones winning:

1. defaults in ``SETTINGS`` below
2. the envars file (``$AUTOGYM_CONFIG_DIR/autogym_envars.sh``)
3. variables already set in the process environment
4. command-line flags (handled in ``main.py``)

The file uses ``export KEY="value"`` lines so shell scripts can ``source`` it.
The web interface edits the same file and asks the app to restart.
"""
import logging
import logging.handlers
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ENVARS_FILENAME = "autogym_envars.sh"

# Exit codes understood by scripts/autogym.sh (mirrors zynthian.sh).
EXIT_STOP = 0       # clean stop: launcher exits
EXIT_ERROR = 1      # crash: launcher waits and restarts
EXIT_RESTART = 102  # restart requested (e.g. config changed): launcher restarts at once


@dataclass(frozen=True)
class Setting:
    key: str
    section: str
    label: str
    default: str
    type: str = "str"                     # str | int | float | bool | choice | color | password
    choices: Tuple[str, ...] = ()
    help: str = ""


SETTINGS: List[Setting] = [
    # Camera
    Setting("AUTOGYM_CAMERA", "Camera", "Camera source", "webcam", "choice",
            ("webcam", "picamera", "file"), "Where frames come from. 'file' plays a video or image on loop."),
    Setting("AUTOGYM_CAMERA_PATH", "Camera", "File path", "",
            help="Video or image file, used when the camera source is 'file'."),
    Setting("AUTOGYM_CAMERA_INDEX", "Camera", "Webcam index", "0", "int",
            help="OpenCV device index for USB webcams (0 = /dev/video0)."),
    Setting("AUTOGYM_CAMERA_WIDTH", "Camera", "Capture width", "640", "int",
            help="RPi Camera Module only."),
    Setting("AUTOGYM_CAMERA_HEIGHT", "Camera", "Capture height", "480", "int",
            help="RPi Camera Module only."),
    # Pose
    Setting("AUTOGYM_POSE_BACKEND", "Pose", "Backend", "mediapipe", "choice",
            ("mediapipe", "coral", "hailo"), "Pose estimation engine."),
    Setting("AUTOGYM_POSE_MODEL_PATH", "Pose", "Model path", "models/pose_landmarker_lite.task",
            help="Model file for the chosen backend. Relative paths are from the repo root."),
    Setting("AUTOGYM_POSE_MIN_CONFIDENCE", "Pose", "Min confidence", "0.5", "float",
            help="Pose detections below this score are ignored (0.0 to 1.0)."),
    # Classifier
    Setting("AUTOGYM_SET_PAUSE_SECONDS", "Workout", "Set pause (s)", "3.0", "float",
            help="Seconds without a rep before the current set is counted as finished."),
    Setting("AUTOGYM_SQUAT_DOWN_ANGLE", "Workout", "Squat: down angle", "120", "float",
            help="Average knee angle (degrees) below which you count as squatting down."),
    Setting("AUTOGYM_SQUAT_UP_ANGLE", "Workout", "Squat: up angle", "160", "float",
            help="Knee angle above which you count as standing again (completes the rep)."),
    Setting("AUTOGYM_PUSHUP_DOWN_ANGLE", "Workout", "Push-up: down angle", "90", "float",
            help="Average elbow angle (degrees) below which you count as lowered."),
    Setting("AUTOGYM_PUSHUP_UP_ANGLE", "Workout", "Push-up: up angle", "150", "float",
            help="Elbow angle above which your arms count as straight (completes the rep)."),
    Setting("AUTOGYM_JJ_MIN_CONFIDENCE", "Workout", "Jumping jack: min confidence", "0.3", "float",
            help="Frames where a wrist, shoulder or hip is less certain than this are ignored (0.0 to 1.0)."),
    # UI
    Setting("AUTOGYM_UI_FULLSCREEN", "Display", "Fullscreen", "1", "bool",
            help="Turn off to run in a window on a desktop (dev / emulator mode)."),
    Setting("AUTOGYM_UI_WIDTH", "Display", "Window width", "1024", "int",
            help="Window size when not fullscreen."),
    Setting("AUTOGYM_UI_HEIGHT", "Display", "Window height", "600", "int",
            help="Window size when not fullscreen."),
    Setting("AUTOGYM_UI_HIDE_CURSOR", "Display", "Hide cursor", "1", "bool",
            help="Hide the mouse pointer over the workout screen."),
    Setting("AUTOGYM_UI_SHOW_DEBUG", "Display", "Open debug view at start", "0", "bool",
            help="Open the camera + skeleton window on launch (toggle with 'd')."),
    Setting("AUTOGYM_UI_FONT_FAMILY", "Display", "Font family", "Helvetica"),
    Setting("AUTOGYM_UI_FONT_SCALE", "Display", "Font scale", "1.0", "float",
            help="Multiplies every font size. Use below 1 on small screens."),
    Setting("AUTOGYM_UI_COLOR_BG", "Display", "Background colour", "#000000", "color"),
    Setting("AUTOGYM_UI_COLOR_TX", "Display", "Text colour", "#ffffff", "color"),
    Setting("AUTOGYM_UI_COLOR_HL", "Display", "Rep counter colour", "#ffffff", "color"),
    # Remote access
    Setting("AUTOGYM_CONTROL_PORT", "Remote", "App control port", "8765", "int",
            help="Local-only port the app listens on so the web interface can read state and send actions."),
    Setting("AUTOGYM_WEB_HOST", "Remote", "Web interface address", "0.0.0.0",
            help="0.0.0.0 = reachable on the LAN, 127.0.0.1 = this machine only."),
    Setting("AUTOGYM_WEB_PORT", "Remote", "Web interface port", "8080", "int"),
    Setting("AUTOGYM_WEB_PASSWORD", "Remote", "Web interface password", "", "password",
            help="If set, the web interface asks for it (any username). Leave empty for no login."),
    # System
    Setting("AUTOGYM_LOG_LEVEL", "System", "Log level", "INFO", "choice",
            ("DEBUG", "INFO", "WARNING", "ERROR")),
]

SETTINGS_BY_KEY: Dict[str, Setting] = {s.key: s for s in SETTINGS}

# (lower, higher) settings that must stay in that order.
_ORDERED_PAIRS = [
    ("AUTOGYM_SQUAT_DOWN_ANGLE", "AUTOGYM_SQUAT_UP_ANGLE"),
    ("AUTOGYM_PUSHUP_DOWN_ANGLE", "AUTOGYM_PUSHUP_UP_ANGLE"),
]

_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_TRUE = {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    env = os.environ.get("AUTOGYM_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / "rpi-auto-gym"


def envars_path() -> Path:
    return config_dir() / ENVARS_FILENAME


def log_path() -> Path:
    return config_dir() / "logs" / "autogym.log"


# ---------------------------------------------------------------------------
# Envars file read / write
# ---------------------------------------------------------------------------

def read_envars_file(path: Optional[Path] = None) -> Dict[str, str]:
    """Parse ``export KEY="value"`` lines. Unknown keys are kept; comments ignored."""
    path = path or envars_path()
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        raw = m.group(2).strip()
        try:
            parts = shlex.split(raw, comments=True)
            values[m.group(1)] = parts[0] if parts else ""
        except ValueError:
            values[m.group(1)] = raw
    return values


def write_envars_file(updates: Dict[str, str], path: Optional[Path] = None) -> Path:
    """Merge ``updates`` into the envars file, keeping other keys and the backup."""
    path = path or envars_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = read_envars_file(path)
    current.update({k: str(v) for k, v in updates.items()})

    if path.is_file():
        path.with_suffix(".sh.bak").write_text(path.read_text())

    lines = [
        "# rpi-auto-gym settings. Edited by the web interface; safe to edit by hand.",
        "# Each line is a shell export so scripts can `source` this file.",
        "",
    ]
    section = None
    for s in SETTINGS:
        if s.key not in current:
            continue
        if s.section != section:
            section = s.section
            lines.append(f"# {section}")
        lines.append(f"export {s.key}={shlex.quote(current[s.key])}")
    extra = [k for k in current if k not in SETTINGS_BY_KEY]
    if extra:
        lines.append("# Other")
        lines.extend(f"export {k}={shlex.quote(current[k])}" for k in extra)

    tmp = path.with_suffix(".sh.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)
    os.chmod(path, 0o600)  # may hold the web password
    return path


def validate(updates: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (clean_values, errors) for a dict of submitted settings."""
    clean: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    for key, value in updates.items():
        s = SETTINGS_BY_KEY.get(key)
        if s is None:
            errors[key] = "unknown setting"
            continue
        v = str(value).strip() if value is not None else ""
        try:
            if s.type == "int":
                v = str(int(v))
            elif s.type == "float":
                v = str(float(v))
            elif s.type == "bool":
                v = "1" if v.lower() in _TRUE else "0"
            elif s.type == "choice" and v not in s.choices:
                raise ValueError(f"must be one of {', '.join(s.choices)}")
            elif s.type == "color" and not re.fullmatch(r"#[0-9a-fA-F]{6}", v):
                raise ValueError("must look like #rrggbb")
        except ValueError as exc:
            msg = str(exc)
            errors[key] = msg if "must" in msg else f"expected {s.type}"
            continue
        clean[key] = v
    for low, high in _ORDERED_PAIRS:
        if low in clean and high in clean and float(clean[low]) >= float(clean[high]):
            errors[high] = f"must be greater than {SETTINGS_BY_KEY[low].label.lower()}"
            clean.pop(high)
    return clean, errors


# ---------------------------------------------------------------------------
# Resolved config
# ---------------------------------------------------------------------------

def load_raw(path: Optional[Path] = None) -> Dict[str, str]:
    """Defaults, then envars file, then process environment."""
    values = {s.key: s.default for s in SETTINGS}
    values.update(read_envars_file(path))
    for s in SETTINGS:
        if s.key in os.environ:
            values[s.key] = os.environ[s.key]
    return values


class Config:
    """Typed, read-only view over the resolved settings."""

    def __init__(self, raw: Optional[Dict[str, str]] = None) -> None:
        self.raw: Dict[str, str] = dict(raw if raw is not None else load_raw())

    def get(self, key: str) -> str:
        default = SETTINGS_BY_KEY[key].default if key in SETTINGS_BY_KEY else ""
        return self.raw.get(key, default)

    def get_int(self, key: str) -> int:
        try:
            return int(self.get(key))
        except ValueError:
            return int(SETTINGS_BY_KEY[key].default)

    def get_float(self, key: str) -> float:
        try:
            return float(self.get(key))
        except ValueError:
            return float(SETTINGS_BY_KEY[key].default)

    def get_bool(self, key: str) -> bool:
        return self.get(key).strip().lower() in _TRUE

    def override(self, **values: Optional[str]) -> "Config":
        """Return a copy with non-None values replaced (used for CLI flags)."""
        raw = dict(self.raw)
        raw.update({k: str(v) for k, v in values.items() if v is not None})
        return Config(raw)


def setup_logging(level: str, name: str = "autogym") -> None:
    """Log to stderr and to a rotating file the web interface can show."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    try:
        log_path().parent.mkdir(parents=True, exist_ok=True)
        # One file per process: two processes rotating one file would clash.
        path = log_path() if name == "autogym" else log_path().with_name(f"{name}.log")
        handlers.append(logging.handlers.RotatingFileHandler(
            path, maxBytes=1_000_000, backupCount=2))
    except OSError:
        pass
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=f"%(asctime)s %(levelname)s [{name}] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
