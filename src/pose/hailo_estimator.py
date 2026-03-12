import time

import numpy as np

from .base import PoseEstimator
from .types import COCO_KEYPOINT_NAMES, Keypoint, Pose


class HailoPoseEstimator(PoseEstimator):
    def __init__(self, model_path: str, min_confidence: float = 0.5) -> None:
        try:
            from hailo_platform import (  # type: ignore[import-untyped]
                HEF,
                VDevice,
                HailoStreamInterface,
                InferVStreams,
                ConfigureParams,
                InputVStreamParams,
                OutputVStreamParams,
                FormatType,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "Install pycoral/hailort to use this backend"
            ) from exc

        self._model_path = model_path
        self._min_confidence = min_confidence
        self._hailo_platform = __import__("hailo_platform")

    def estimate(self, frame: np.ndarray) -> Pose | None:
        try:
            import hailo_platform  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "Install pycoral/hailort to use this backend"
            ) from exc

        frame_height, frame_width = frame.shape[:2]

        hef = hailo_platform.HEF(self._model_path)
        with hailo_platform.VDevice() as target:
            configure_params = hailo_platform.ConfigureParams.create_from_hef(
                hef, interface=hailo_platform.HailoStreamInterface.PCIe
            )
            network_groups = target.configure(hef, configure_params)
            network_group = network_groups[0]

            input_params = hailo_platform.InputVStreamParams.make(network_group)
            output_params = hailo_platform.OutputVStreamParams.make(network_group)

            with hailo_platform.InferVStreams(
                network_group, input_params, output_params
            ) as infer_pipeline:
                input_data = {
                    hailo_platform.InputVStream.get_name(): np.expand_dims(frame, 0)
                }
                raw_output = infer_pipeline.infer(input_data)

        detections = next(iter(raw_output.values()))
        return self._best_detection_as_pose(
            detections, frame_height, frame_width
        )

    def _best_detection_as_pose(
        self,
        detections: np.ndarray,
        frame_height: int,
        frame_width: int,
    ) -> Pose | None:
        if detections is None or len(detections) == 0:
            return None

        # YOLO pose: each detection has box confidence + 17 keypoints × 3 values
        # Pick detection with highest box confidence
        best = max(detections, key=lambda d: float(d[4]))

        box_confidence = float(best[4])
        if box_confidence < self._min_confidence:
            return None

        keypoints = self._build_keypoints(best[5:], frame_height, frame_width)
        return Pose(keypoints=keypoints, timestamp=time.monotonic())

    def _build_keypoints(
        self,
        raw_keypoints: np.ndarray,
        frame_height: int,
        frame_width: int,
    ) -> dict[str, Keypoint]:
        # YOLO keypoints: [x_px, y_px, conf] × 17 — normalize by frame dims
        return {
            name: Keypoint(
                x=float(raw_keypoints[idx * 3]) / frame_width,
                y=float(raw_keypoints[idx * 3 + 1]) / frame_height,
                confidence=float(raw_keypoints[idx * 3 + 2]),
            )
            for idx, name in enumerate(COCO_KEYPOINT_NAMES)
        }
