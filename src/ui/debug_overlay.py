from typing import Optional

import tkinter as tk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as exc:
    raise ImportError(
        "DebugOverlay requires opencv-python, numpy, and Pillow. "
        "Install them with: pip install opencv-python numpy Pillow"
    ) from exc


from .skeleton import draw_skeleton


class DebugOverlay(tk.Toplevel):
    """Floating window showing camera feed with pose skeleton drawn on top."""

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.title("Debug — Camera Feed")
        self.configure(bg="#000000")
        self.resizable(True, True)

        self._canvas = tk.Canvas(self, bg="#000000", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Keep a reference to prevent garbage collection
        self._photo: "Optional[ImageTk.PhotoImage]" = None

    def update_frame(self, frame: "np.ndarray", pose: "Optional[Pose]") -> None:
        """Draw frame + skeleton and update the canvas. frame is RGB uint8."""
        rendered = draw_skeleton(frame, pose)

        pil_image = Image.fromarray(rendered)
        photo = ImageTk.PhotoImage(pil_image)

        canvas_width = pil_image.width
        canvas_height = pil_image.height
        self._canvas.config(width=canvas_width, height=canvas_height)

        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        # Hold reference so GC doesn't collect the image
        self._photo = photo
