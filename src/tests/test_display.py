from src import display
from src.display import Blanker

WLR = """HDMI-A-1 "Some Monitor"
  Enabled: no
DSI-1 "Touch screen"
  Physical size: 155x86 mm
  Enabled: yes
  Modes:
    800x480 px, 60.000000 Hz (preferred, current)
"""


def test_parse_wlr_outputs_enabled_first() -> None:
    assert display.parse_wlr_outputs(WLR) == ["DSI-1", "HDMI-A-1"]


def test_parse_xrandr_outputs() -> None:
    text = "Screen 0: minimum 320\nHDMI-1 connected primary 1920x1080+0+0\nHDMI-2 disconnected\n"
    assert display.parse_xrandr_outputs(text) == ["HDMI-1"]


def test_commands() -> None:
    assert display.rotation_command("wayland", "DSI-1", "180") == ["wlr-randr", "--output", "DSI-1", "--transform", "180"]
    assert display.rotation_command("x11", "HDMI-1", "90") == ["xrandr", "--output", "HDMI-1", "--rotate", "left"]
    assert display.power_command("wayland", "DSI-1", False)[-1] == "--off"
    assert display.power_command("x11", "-", True) == ["xset", "dpms", "force", "on"]


def test_system_rotation_does_nothing() -> None:
    assert display.apply_rotation("system") is False


def test_blanker_off_when_timeout_zero() -> None:
    b = Blanker(0)
    assert b.update(False, 0) is None
    assert b.update(False, 1000) is None


def test_blanker_blanks_and_wakes() -> None:
    b = Blanker(10)
    assert b.update(True, 0) is None
    assert b.update(False, 5) is None
    assert b.update(False, 10) is True       # 10s since last seen
    assert b.update(False, 20) is None       # already blank
    assert b.update(True, 21) is False       # someone stepped in
    assert b.update(True, 22) is None


def test_blanker_wake_resets_timer() -> None:
    b = Blanker(10)
    b.update(False, 0)
    assert b.update(False, 10) is True
    assert b.wake(11) is True
    assert b.update(False, 15) is None
    assert b.update(False, 21) is True
