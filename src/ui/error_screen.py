"""Full-screen error message shown by scripts/autogym.sh while it waits to restart the app.

Like zynthian's error splash: says what went wrong and where to reach the web
interface, so a wall-mounted unit with no keyboard can still be fixed.

    python -m src.ui.error_screen --code 1 --attempt 2 --seconds 10
"""
import argparse
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

from .. import netinfo
from ..config import Config, log_path

_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit)\b")


def last_error(path: Optional[Path] = None, max_lines: int = 300) -> str:
    """Best guess at the latest error from the app log: the exception line, else the last ERROR line."""
    path = path or log_path()
    if not path.is_file():
        return ""
    with path.open(errors="replace") as fh:
        lines = [line.rstrip() for line in deque(fh, maxlen=max_lines)]
    for line in reversed(lines):
        if _EXCEPTION_LINE.match(line):
            return line[:200]
        if " ERROR " in line or " CRITICAL " in line:
            return line.split(": ", 1)[-1][:200]
    return ""


def message_lines(code: int, attempt: int, cfg: Config) -> List[str]:
    urls = netinfo.web_urls(cfg.get_int("AUTOGYM_WEB_PORT"), cfg.get("AUTOGYM_WEB_HOST"))
    lines = [f"Exit code {code}, attempt {attempt}"]
    error = last_error()
    if error:
        lines.append(error)
    lines.append("")
    lines.append("Check settings and logs at:")
    lines.extend(urls)
    return lines


def show(code: int, attempt: int, seconds: float) -> None:
    cfg = Config()
    lines = message_lines(code, attempt, cfg)
    print("\n".join(["Auto Gym stopped with an error"] + lines), file=sys.stderr)

    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        time.sleep(seconds)
        return
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:  # no Tk or no display: the console message is enough
        time.sleep(seconds)
        return

    bg, fg = cfg.get("AUTOGYM_UI_COLOR_BG"), cfg.get("AUTOGYM_UI_COLOR_TX")
    family = cfg.get("AUTOGYM_UI_FONT_FAMILY")
    scale = cfg.get_float("AUTOGYM_UI_FONT_SCALE")
    root.title("Auto Gym — error")
    root.configure(bg=bg, cursor="none" if cfg.get_bool("AUTOGYM_UI_HIDE_CURSOR") else "")
    if cfg.get_bool("AUTOGYM_UI_FULLSCREEN"):
        root.attributes("-fullscreen", True)
    else:
        root.geometry(f"{cfg.get_int('AUTOGYM_UI_WIDTH')}x{cfg.get_int('AUTOGYM_UI_HEIGHT')}")

    tk.Label(root, text="Auto Gym hit a problem", bg=bg, fg="#ff5050",
             font=(family, int(40 * scale), "bold")).pack(pady=(40, 16))
    tk.Label(root, text="\n".join(lines), bg=bg, fg=fg, justify="center", wraplength=900,
             font=(family, int(22 * scale))).pack(padx=24)
    countdown = tk.Label(root, bg=bg, fg="#8a94a3", font=(family, int(18 * scale)))
    countdown.pack(pady=24)

    deadline = time.monotonic() + seconds

    def tick() -> None:
        left = deadline - time.monotonic()
        if left <= 0:
            root.destroy()
            return
        countdown.config(text=f"Restarting in {int(left) + 1}s")
        root.after(250, tick)

    root.bind("<Key>", lambda _e: root.destroy())  # any key: retry now
    tick()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=int, default=1)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    show(args.code, args.attempt, args.seconds)


if __name__ == "__main__":
    main()
