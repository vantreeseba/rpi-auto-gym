import os
import queue
import sys
import tkinter as tk
from typing import Any, Optional, Union

from ._headless import FakeLabel
from .session_protocol import SessionProtocol
from .session_types import SessionState

# Base font sizes, multiplied by AUTOGYM_UI_FONT_SCALE.
_SIZE_EXERCISE = 48
_SIZE_REPS = 120
_SIZE_SET = 36
_SIZE_SMALL = 18

# Keyboard shortcuts (handy in windowed/dev mode, like zynthian's emulator keys).
KEY_BINDINGS = {
    "d": "toggle_debug",
    "r": "reset_session",
    "F11": "toggle_fullscreen",
    "q": "stop",
    "Escape": "stop",
}


def _format_exercise_name(raw: str) -> str:
    return raw.replace("_", " ").title()


def _is_headless() -> bool:
    return sys.platform == "linux" and not os.environ.get("DISPLAY")


class WorkoutApp:
    """Workout display app. Headless-safe: skips Tk when DISPLAY is unset.

    ``cfg`` is a ``src.config.Config`` (defaults are used when omitted).
    ``control`` is a ``src.control.AppControl``; when given, the app runs the
    UI commands it queues (from the web interface) and routes key presses to it.
    """

    POLL_MS = 100

    def __init__(self, session: SessionProtocol, cfg: Optional[Any] = None,
                 control: Optional[Any] = None) -> None:
        if cfg is None:
            from ..config import Config
            cfg = Config({})
        self._session = session
        self._cfg = cfg
        self._control = control
        self._headless = _is_headless()
        self._debug_overlay: Optional[Any] = None
        self._bg = cfg.get("AUTOGYM_UI_COLOR_BG")
        self._fg = cfg.get("AUTOGYM_UI_COLOR_TX")
        self._hl = cfg.get("AUTOGYM_UI_COLOR_HL")
        self._fullscreen = cfg.get_bool("AUTOGYM_UI_FULLSCREEN")

        if self._headless:
            self._root: Optional[tk.Tk] = None
            self._elapsed_label: Union[FakeLabel, tk.Label] = FakeLabel("0:00")
            self._exercise_label: Union[FakeLabel, tk.Label] = FakeLabel("—")
            self._rep_label: Union[FakeLabel, tk.Label] = FakeLabel("0")
            self._set_label: Union[FakeLabel, tk.Label] = FakeLabel("Set 0")
            self._history_label: Union[FakeLabel, tk.Label] = FakeLabel("")
            return

        self._root = tk.Tk()
        self._root.title("Auto Gym")
        self._root.configure(bg=self._bg)
        if self._fullscreen:
            self._root.attributes("-fullscreen", True)
        else:
            w = cfg.get_int("AUTOGYM_UI_WIDTH")
            h = cfg.get_int("AUTOGYM_UI_HEIGHT")
            self._root.geometry(f"{w}x{h}")
        if cfg.get_bool("AUTOGYM_UI_HIDE_CURSOR"):
            self._root.configure(cursor="none")
        self._build_layout()
        if cfg.get_bool("AUTOGYM_UI_SHOW_DEBUG"):
            self._root.after(500, self._toggle_debug)

    @property
    def headless(self) -> bool:
        return self._headless

    def _font(self, size: int, bold: bool = False) -> tuple:
        scaled = max(6, int(size * self._cfg.get_float("AUTOGYM_UI_FONT_SCALE")))
        family = self._cfg.get("AUTOGYM_UI_FONT_FAMILY")
        return (family, scaled, "bold") if bold else (family, scaled)

    def _build_layout(self) -> None:
        """Construct all label widgets in landscape workout layout."""
        assert self._root is not None
        root = self._root
        root.columnconfigure(0, weight=1)

        bg, fg = self._bg, self._fg
        top_bar = tk.Frame(root, bg=bg)
        top_bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(8, 0))
        top_bar.columnconfigure(0, weight=1)
        self._elapsed_label = tk.Label(
            top_bar, text="0:00", font=self._font(_SIZE_SMALL), bg=bg, fg=fg, anchor="e"
        )
        self._elapsed_label.grid(row=0, column=0, sticky="e")

        self._exercise_label = tk.Label(root, text="—", font=self._font(_SIZE_EXERCISE, True), bg=bg, fg=fg)
        self._exercise_label.grid(row=1, column=0, pady=(24, 0))

        self._rep_label = tk.Label(root, text="0", font=self._font(_SIZE_REPS, True), bg=bg, fg=self._hl)
        self._rep_label.grid(row=2, column=0)

        tk.Label(root, text="REPS", font=self._font(_SIZE_SMALL), bg=bg, fg=fg).grid(row=3, column=0)

        self._set_label = tk.Label(root, text="Set 0", font=self._font(_SIZE_SET), bg=bg, fg=fg)
        self._set_label.grid(row=4, column=0, pady=(8, 0))

        self._history_label = tk.Label(
            root, text="", font=self._font(_SIZE_SMALL), bg=bg, fg=fg, justify="left"
        )
        self._history_label.grid(row=5, column=0, pady=(16, 8), padx=16, sticky="w")

        for key, action in KEY_BINDINGS.items():
            root.bind(f"<KeyPress-{key}>", lambda _e, a=action: self.handle_action(a))
    def handle_action(self, action: str) -> None:
        """Run a UI action from a key press or the control API."""
        if action == "toggle_debug":
            self._toggle_debug()
        elif action == "toggle_fullscreen":
            self._fullscreen = not self._fullscreen
            if self._root is not None:
                self._root.attributes("-fullscreen", self._fullscreen)
        elif action == "quit":
            self.destroy()
        elif self._control is not None:
            self._control.perform(action)  # reset_session / stop / restart
        elif action == "stop":
            self.destroy()

    def _toggle_debug(self, event=None) -> None:
        """Open or close the debug overlay window."""
        if self._debug_overlay is None or not self._debug_overlay.winfo_exists():
            from .debug_overlay import DebugOverlay  # lazy: avoids hard dep at import time
            assert self._root is not None
            self._debug_overlay = DebugOverlay(self._root)
        else:
            self._debug_overlay.destroy()
            self._debug_overlay = None

    def run(self) -> None:
        """Start the tkinter mainloop. In headless mode returns immediately."""
        if self._headless:
            return
        assert self._root is not None
        self._poll()
        self._root.mainloop()

    def destroy(self) -> None:
        if self._root is not None:
            self._root.destroy()
            self._root = None

    def _poll(self) -> None:
        """Read session state, update display, then reschedule."""
        assert self._root is not None
        if self._control is not None:
            while True:
                try:
                    command = self._control.ui_commands.get_nowait()
                except queue.Empty:
                    break
                self.handle_action(command)
                if command == "quit":
                    return
        state = self._session.get_state()
        self._update_display(state)
        if self._debug_overlay is not None and self._debug_overlay.winfo_exists():
            if state.last_frame is not None:
                self._debug_overlay.update_frame(state.last_frame, state.last_pose)
        self._root.after(self.POLL_MS, self._poll)

    def _update_display(self, state: SessionState) -> None:
        """Update all label widgets from a SessionState."""
        minutes, seconds = divmod(int(state.elapsed_seconds), 60)
        self._elapsed_label.config(text=f"{minutes}:{seconds:02d}")

        self._exercise_label.config(text=(
            _format_exercise_name(state.active_exercise)
            if state.active_exercise is not None
            else "—"
        ))

        self._rep_label.config(text=str(state.current_rep_count))

        active_sets = 0
        if state.active_exercise and state.active_exercise in state.exercises:
            active_sets = state.exercises[state.active_exercise].sets
        self._set_label.config(text=f"Set {active_sets}")

        self._history_label.config(text="\n".join(
            f"{name}: {log.sets} sets / {log.total_reps} reps"
            for name, log in state.exercises.items()
        ))
