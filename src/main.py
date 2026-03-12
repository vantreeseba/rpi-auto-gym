#!/usr/bin/env python3
"""
rpi-auto-gym entry point.

Usage:
    python -m src.main [--backend mediapipe|coral|hailo] [--camera file|webcam|picamera] [--camera-path PATH] [--model-path PATH]

Defaults:
    --backend mediapipe
    --camera webcam
"""
import argparse
import threading
import time

from .camera.factory import make_camera
from .pose.factory import make_estimator
from .classifier.classifier import ExerciseClassifier
from .session.session import Session
from .ui.app import WorkoutApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rpi-auto-gym")
    parser.add_argument("--backend", choices=["mediapipe", "coral", "hailo"], default="mediapipe")
    parser.add_argument("--camera", choices=["file", "webcam", "picamera"], default="webcam")
    parser.add_argument("--camera-path", default=None, help="Path to image/video file (for --camera file)")
    parser.add_argument("--model-path", default="models/pose_landmarker_lite.task", help="Path to pose model file")
    return parser.parse_args()


def capture_loop(camera, estimator, classifier, session, stop_event: threading.Event) -> None:
    """Runs in a background thread. Reads frames, estimates pose, updates classifier and session."""
    camera.start()
    try:
        while not stop_event.is_set():
            frame = camera.read()
            pose = estimator.estimate(frame)
            if pose is not None:
                result = classifier.update(pose)
                session.update(result)
            time.sleep(0.01)  # ~100fps cap, pose estimation is the real bottleneck
    finally:
        camera.stop()


def main() -> None:
    args = parse_args()

    # Build components
    camera_kwargs = {}
    if args.camera == "file":
        if args.camera_path is None:
            raise ValueError("--camera-path is required when --camera=file")
        camera_kwargs["path"] = args.camera_path

    camera = make_camera(args.camera, **camera_kwargs)
    estimator = make_estimator(args.backend, model_path=args.model_path)
    classifier = ExerciseClassifier()
    session = Session()
    app = WorkoutApp(session)

    # Start capture loop in background thread
    stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=capture_loop,
        args=(camera, estimator, classifier, session, stop_event),
        daemon=True,
    )
    capture_thread.start()

    # Run UI on main thread (tkinter requires main thread)
    try:
        app.run()
    finally:
        stop_event.set()
        capture_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
