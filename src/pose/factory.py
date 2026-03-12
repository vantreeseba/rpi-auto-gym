from .base import PoseEstimator


def make_estimator(backend: str, model_path: str, **kwargs: object) -> PoseEstimator:
    """
    backend: 'mediapipe' | 'coral' | 'hailo'
    Raises ValueError for unknown backends.
    """
    if backend == "mediapipe":
        from .mediapipe_estimator import MediaPipePoseEstimator
        return MediaPipePoseEstimator(model_path, **kwargs)  # type: ignore[arg-type]

    if backend == "coral":
        from .coral_estimator import CoralPoseEstimator
        return CoralPoseEstimator(model_path, **kwargs)  # type: ignore[arg-type]

    if backend == "hailo":
        from .hailo_estimator import HailoPoseEstimator
        return HailoPoseEstimator(model_path, **kwargs)  # type: ignore[arg-type]

    raise ValueError(
        f"Unknown backend '{backend}'. Choose from: 'mediapipe', 'coral', 'hailo'."
    )
