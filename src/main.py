#!/usr/bin/env python3
"""
rpi-auto-gym entry point.

Settings come from $AUTOGYM_CONFIG_DIR/autogym_envars.sh (see src/config.py);
command-line flags override them for one run.

Usage:
    python -m src.main [--backend mediapipe|coral|hailo] [--camera file|webcam|picamera]
                       [--camera-path PATH] [--model-path PATH] [--windowed] [--debug]

Exit codes (handled by scripts/autogym.sh): 0 stop, 102 restart, other = error.
"""
import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from .camera.factory import make_camera
from .classifier.classifier import ExerciseClassifier
from .config import EXIT_ERROR, Config, envars_path, repo_dir, setup_logging
from .control import AppControl, start_control_server
from .pose.factory import make_estimator
from .session.session import Session
from .ui.app import WorkoutApp

log = logging.getLogger("autogym")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rpi-auto-gym")
    parser.add_argument("--backend", choices=["mediapipe", "coral", "hailo"], default=None)
    parser.add_argument("--camera", choices=["file", "webcam", "picamera"], default=None)
    parser.add_argument("--camera-path", default=None, help="Path to image/video file (for --camera file)")
    parser.add_argument("--model-path", default=None, help="Path to pose model file")
    parser.add_argument("--windowed", action="store_true", help="Run in a window instead of fullscreen")
    parser.add_argument("--debug", action="store_true", help="Debug logging and open the debug view")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    return Config().override(
        AUTOGYM_POSE_BACKEND=args.backend,
        AUTOGYM_CAMERA=args.camera,
        AUTOGYM_CAMERA_PATH=args.camera_path,
        AUTOGYM_POSE_MODEL_PATH=args.model_path,
        AUTOGYM_UI_FULLSCREEN="0" if args.windowed else None,
        AUTOGYM_LOG_LEVEL="DEBUG" if args.debug else None,
        AUTOGYM_UI_SHOW_DEBUG="1" if args.debug else None,
    )


def _resolve(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists():
        p = repo_dir() / p
    return str(p)


def make_components(cfg: Config):
    source = cfg.get("AUTOGYM_CAMERA")
    camera_kwargs = {}
    if source == "file":
        if not cfg.get("AUTOGYM_CAMERA_PATH"):
            raise ValueError("A camera file path is required when the camera source is 'file'")
        camera_kwargs["path"] = _resolve(cfg.get("AUTOGYM_CAMERA_PATH"))
    elif source == "webcam":
        camera_kwargs["index"] = cfg.get_int("AUTOGYM_CAMERA_INDEX")
    elif source == "picamera":
        camera_kwargs["width"] = cfg.get_int("AUTOGYM_CAMERA_WIDTH")
        camera_kwargs["height"] = cfg.get_int("AUTOGYM_CAMERA_HEIGHT")

    camera = make_camera(source, **camera_kwargs)
    estimator = make_estimator(
        cfg.get("AUTOGYM_POSE_BACKEND"),
        model_path=_resolve(cfg.get("AUTOGYM_POSE_MODEL_PATH")),
        min_confidence=cfg.get_float("AUTOGYM_POSE_MIN_CONFIDENCE"),
    )
    classifier = ExerciseClassifier(set_pause_seconds=cfg.get_float("AUTOGYM_SET_PAUSE_SECONDS"))
    return camera, estimator, classifier


def capture_loop(camera, estimator, classifier, session, control: AppControl) -> None:
    """Runs in a background thread. Reads frames, estimates pose, updates classifier and session."""
    camera.start()
    try:
        while not control.stop_event.is_set():
            if control.reset_requested.is_set():
                control.reset_requested.clear()
                classifier.reset()
                session.reset()
            frame = camera.read()
            pose = estimator.estimate(frame)
            session.update_frame(frame, pose)
            if pose is not None:
                result = classifier.update(pose)
                session.update(result)
            control.frame_done()
            time.sleep(0.01)  # ~100fps cap, pose estimation is the real bottleneck
    except Exception:
        log.exception("Capture loop crashed")
        control.request_exit(EXIT_ERROR)
    finally:
        camera.stop()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    setup_logging(cfg.get("AUTOGYM_LOG_LEVEL"))
    log.info("Config file: %s", envars_path())

    try:
        camera, estimator, classifier = make_components(cfg)
    except Exception:
        log.exception("Startup failed")
        return EXIT_ERROR

    session = Session()
    control = AppControl(session)
    server = start_control_server(control, cfg.get_int("AUTOGYM_CONTROL_PORT"))
    app = WorkoutApp(session, cfg, control)

    capture_thread = threading.Thread(
        target=capture_loop,
        args=(camera, estimator, classifier, session, control),
        name="capture",
        daemon=True,
    )
    capture_thread.start()

    try:
        if app.headless:
            log.info("No display found: running headless (use the web interface to watch)")
            control.stop_event.wait()
        else:
            app.run()  # tkinter requires the main thread
    except KeyboardInterrupt:
        pass
    finally:
        control.stop_event.set()
        capture_thread.join(timeout=2.0)
        if server is not None:
            server.shutdown()

    log.info("Exiting with code %d", control.exit_code)
    return control.exit_code


if __name__ == "__main__":
    sys.exit(main())
