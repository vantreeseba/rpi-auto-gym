"""Screen rotation and power, via wlr-randr (Wayland, Pi OS default) or xrandr/xset (X11).

Failures are logged and ignored: a missing tool must never stop the app.
"""
import logging
import os
import re
import shutil
import subprocess
import time
from typing import List, Optional

log = logging.getLogger(__name__)

# Rotation is counter-clockwise, as in the Wayland protocol and xrandr's "left".
_XRANDR_ROTATE = {"0": "normal", "90": "left", "180": "inverted", "270": "right"}
_WLR_TRANSFORM = {"0": "normal", "90": "90", "180": "180", "270": "270"}


def session_type() -> Optional[str]:
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return None


def _run(cmd: List[str]) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        log.warning("Display: '%s' not installed (see scripts/setup_pi.sh)", cmd[0])
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Display: %s failed: %s", " ".join(cmd), exc)
        return None
    if out.returncode != 0:
        log.warning("Display: %s failed: %s", " ".join(cmd), out.stderr.strip())
        return None
    return out.stdout


def parse_wlr_outputs(text: str) -> List[str]:
    """Output names from `wlr-randr` (unindented lines, first word), enabled ones first."""
    outputs, enabled = [], []
    name = None
    for line in text.splitlines():
        if line and not line[0].isspace():
            name = line.split()[0]
            outputs.append(name)
        elif name and re.match(r"\s+Enabled:\s+yes", line):
            enabled.append(name)
    return enabled + [o for o in outputs if o not in enabled]


def parse_xrandr_outputs(text: str) -> List[str]:
    return re.findall(r"^(\S+) connected", text, re.MULTILINE)


def find_output(session: str, preferred: str = "") -> Optional[str]:
    if preferred:
        return preferred
    if session == "wayland":
        text = _run(["wlr-randr"])
        outputs = parse_wlr_outputs(text) if text else []
    else:
        text = _run(["xrandr", "--query"])
        outputs = parse_xrandr_outputs(text) if text else []
    return outputs[0] if outputs else None


def rotation_command(session: str, output: str, rotation: str) -> List[str]:
    if session == "wayland":
        return ["wlr-randr", "--output", output, "--transform", _WLR_TRANSFORM[rotation]]
    return ["xrandr", "--output", output, "--rotate", _XRANDR_ROTATE[rotation]]


def power_command(session: str, output: str, on: bool) -> List[str]:
    if session == "wayland":
        return ["wlr-randr", "--output", output, "--on" if on else "--off"]
    return ["xset", "dpms", "force", "on" if on else "off"]


def apply_rotation(rotation: str, output: str = "") -> bool:
    """Rotate the screen. ``rotation`` is 'system' (leave alone) or 0/90/180/270."""
    if rotation in ("", "system"):
        return False
    session = session_type()
    if session is None or rotation not in _XRANDR_ROTATE:
        return False
    target = find_output(session, output)
    if target is None:
        log.warning("Display: no connected output found to rotate")
        return False
    ok = _run(rotation_command(session, target, rotation)) is not None
    if ok:
        log.info("Display: rotated %s to %s", target, rotation)
        time.sleep(0.5)  # let the compositor settle before the window goes fullscreen
    return ok


def set_power(on: bool, output: str = "") -> bool:
    session = session_type()
    if session is None:
        return False
    target = find_output(session, output) if session == "wayland" else "-"
    if target is None:
        return False
    return _run(power_command(session, target, on)) is not None


class Blanker:
    """Decides when to blank the screen: after ``timeout`` seconds with no pose.

    ``update`` returns True to blank, False to wake, or None for no change.
    """

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.blanked = False
        self._last_seen: Optional[float] = None

    def update(self, person_in_view: bool, now: float) -> Optional[bool]:
        if self.timeout <= 0:
            return None
        if self._last_seen is None or person_in_view:
            self._last_seen = now
        if person_in_view and self.blanked:
            self.blanked = False
            return False
        if not self.blanked and now - self._last_seen >= self.timeout:
            self.blanked = True
            return True
        return None

    def wake(self, now: float) -> bool:
        """Force awake (e.g. key press). Returns True if it was blanked."""
        self._last_seen = now
        was = self.blanked
        self.blanked = False
        return was
