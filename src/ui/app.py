import os
import sys
import tkinter as tk
from typing import Any, Optional, Union

from ._headless import FakeLabel
from .session_protocol import SessionProtocol
from .session_types import SessionState

_BG = "#000000"
_FG = "#FFFFFF"
_FONT_EXERCISE = ("Helvetica", 48, "bold")
_FONT_REPS = ("Helvetica", 120, "bold")
_FONT_SET = ("Helvetica", 36)
_FONT_SMALL = ("Helvetica", 18)


def _format_exercise_name(raw: str) -> str:
    return raw.replace("_", " ").title()


def _is_headless() -> bool:
    return sys.platform == "linux" and not os.environ.get("DISPLAY")


class WorkoutApp:
    """Workout display app. Headless-safe: skips Tk when DISPLAY is unset."""

    POLL_MS = 100

    def __init__(self, session: SessionProtocol) -> None:
        self._session = session
        self._headless = _is_headless()
        self._debug_overlay: Optional[Any] = None

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
        self._root.configure(bg=_BG)
        self._root.attributes("-fullscreen", True)
        self._build_layout()

    def _build_layout(self) -> None:
        """Construct all label widgets in landscape workout layout."""
        assert self._root is not None
        root = self._root
        root.columnconfigure(0, weight=1)

        top_bar = tk.Frame(root, bg=_BG)
        top_bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(8, 0))
        top_bar.columnconfigure(0, weight=1)
        self._elapsed_label = tk.Label(
            top_bar, text="0:00", font=_FONT_SMALL, bg=_BG, fg=_FG, anchor="e"
        )
        self._elapsed_label.grid(row=0, column=0, sticky="e")

        self._exercise_label = tk.Label(root, text="—", font=_FONT_EXERCISE, bg=_BG, fg=_FG)
        self._exercise_label.grid(row=1, column=0, pady=(24, 0))

        self._rep_label = tk.Label(root, text="0", font=_FONT_REPS, bg=_BG, fg=_FG)
        self._rep_label.grid(row=2, column=0)

        tk.Label(root, text="REPS", font=_FONT_SMALL, bg=_BG, fg=_FG).grid(row=3, column=0)

        self._set_label = tk.Label(root, text="Set 0", font=_FONT_SET, bg=_BG, fg=_FG)
        self._set_label.grid(row=4, column=0, pady=(8, 0))

        self._history_label = tk.Label(
            root, text="", font=_FONT_SMALL, bg=_BG, fg=_FG, justify="left"
        )
        self._history_label.grid(row=5, column=0, pady=(16, 8), padx=16, sticky="w")

        root.bind("<KeyPress-d>", self._toggle_debug)

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

    def _poll(self) -> None:
        """Read session state, update display, then reschedule."""
        assert self._root is not None
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
