"""Headless-mode stand-ins for tkinter widgets."""
from typing import Any


class FakeLabel:
    """Minimal stand-in for tk.Label used when no display is available."""

    def __init__(self, text: str = "") -> None:
        self._text = text

    def config(self, **kwargs: Any) -> None:
        if "text" in kwargs:
            self._text = kwargs["text"]

    def cget(self, key: str) -> Any:
        if key == "text":
            return self._text
        raise KeyError(f"Unknown option: {key!r}")
