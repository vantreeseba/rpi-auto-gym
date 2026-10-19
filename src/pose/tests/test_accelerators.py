"""Coral and Hailo backends, and Pi camera colour order, with fake hardware."""
import sys
import types

import numpy as np
import pytest

from src.pose.coral_estimator import CoralPoseEstimator
from src.pose.hailo_estimator import HailoPoseEstimator, decode_best_pose
from src.pose.preprocess import letterbox
from src.pose.types import COCO_KEYPOINT_NAMES


# --- letterbox --------------------------------------------------------------

def test_letterbox_pads_and_maps_back() -> None:
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    image, box = letterbox(frame, 320, 320)
    assert image.shape == (320, 320, 3)
    assert box.scale == pytest.approx(0.5)
    assert (box.pad_x, box.pad_y) == (0, 40)
    assert image[0, 0, 0] == 0 and image[160, 160, 0] == 255  # black bars, picture in the middle
    assert box.to_frame(160, 160) == pytest.approx((0.5, 0.5))
    assert box.to_frame(0, 40) == pytest.approx((0.0, 0.0))
    assert box.to_frame(320, 280) == pytest.approx((1.0, 1.0))


# --- Coral --------------------------------------------------------------------

class FakeInterpreter:
    def __init__(self, output: np.ndarray, in_dtype=np.uint8, quant=(0.0, 0)) -> None:
        self.output = output
        self.in_dtype = in_dtype
        self.quant = quant
        self.fed = None

    def allocate_tensors(self) -> None:
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": np.array([1, 192, 192, 3]), "dtype": self.in_dtype}]

    def get_output_details(self):
        return [{"index": 1, "quantization": self.quant}]

    def set_tensor(self, index, value) -> None:
        self.fed = value

    def invoke(self) -> None:
        pass

    def get_tensor(self, index):
        return self.output


def _movenet_output(y: float, x: float, score: float) -> np.ndarray:
    out = np.zeros((1, 1, 17, 3), dtype=np.float32)
    out[0, 0, :] = (y, x, score)
    return out


def test_coral_maps_keypoints_back_through_letterbox() -> None:
    interp = FakeInterpreter(_movenet_output(0.5, 0.5, 0.9))
    est = CoralPoseEstimator("m.tflite", min_confidence=0.3, interpreter=interp)
    pose = est.estimate(np.zeros((480, 640, 3), dtype=np.uint8))
    assert interp.fed.shape == (1, 192, 192, 3) and interp.fed.dtype == np.uint8
    nose = pose.keypoints["nose"]
    assert (nose.x, nose.y, nose.confidence) == pytest.approx((0.5, 0.5, 0.9))
    # top of the picture area (below the letterbox bar) maps to y=0
    interp.output = _movenet_output(24 / 192, 0.0, 0.9)
    assert est.estimate(np.zeros((480, 640, 3), dtype=np.uint8)).keypoints["nose"].y == pytest.approx(0.0)


def test_coral_dequantizes_and_applies_threshold() -> None:
    raw = np.zeros((1, 1, 17, 3), dtype=np.uint8)
    raw[..., 2] = 10  # 10 * 0.01 = 0.1
    est = CoralPoseEstimator("m.tflite", min_confidence=0.3,
                             interpreter=FakeInterpreter(raw, quant=(0.01, 0)))
    assert est.estimate(np.zeros((192, 192, 3), dtype=np.uint8)) is None


# --- Hailo --------------------------------------------------------------------

def _yolo_outputs(input_size=64, person=None):
    """Raw YOLOv8-pose style outputs for strides 8/16/32: box(64), score(1), kpts(51) per stride.

    ``person`` = (stride, row, col, raw_kpts(17,3), score_logit)."""
    outs = {}
    for stride in (8, 16, 32):
        n = input_size // stride
        score = np.full((n, n, 1), -10.0, dtype=np.float32)
        kpts = np.zeros((n, n, 51), dtype=np.float32)
        if person and person[0] == stride:
            _, r, c, raw, logit = person
            score[r, c, 0] = logit
            kpts[r, c] = raw.reshape(-1)
        outs[f"box{stride}"] = np.zeros((n, n, 64), dtype=np.float32)
        outs[f"score{stride}"] = score
        outs[f"kpts{stride}"] = kpts
    return outs


def test_decode_picks_best_cell_and_decodes_keypoints() -> None:
    raw = np.zeros((17, 3), dtype=np.float32)
    raw[:, 0], raw[:, 1], raw[:, 2] = 0.25, -0.5, 5.0
    outs = _yolo_outputs(person=(16, 2, 1, raw, 3.0))
    score, kpts = decode_best_pose(outs.values(), 64, 64)
    assert score == pytest.approx(1 / (1 + np.exp(-3.0)))
    # x = (0.25*2 + col 1) * 16 = 24 ; y = (-0.5*2 + row 2) * 16 = 16
    assert kpts[0, 0] == pytest.approx(24.0)
    assert kpts[0, 1] == pytest.approx(16.0)
    assert kpts[0, 2] == pytest.approx(1 / (1 + np.exp(-5.0)))


def test_decode_ignores_unknown_outputs() -> None:
    assert decode_best_pose([np.zeros((8, 8, 64))], 64, 64) is None


class FakeRunner:
    input_shape = (64, 64, 3)

    def __init__(self, outputs) -> None:
        self.outputs = outputs
        self.seen = None

    def infer(self, image):
        self.seen = image
        return self.outputs


def test_hailo_estimator_returns_pose_in_frame_coords() -> None:
    raw = np.zeros((17, 3), dtype=np.float32)
    raw[:, 0], raw[:, 1], raw[:, 2] = 0.0, 0.0, 5.0  # keypoint at the cell corner
    runner = FakeRunner(_yolo_outputs(person=(8, 4, 4, raw, 4.0)))  # (32, 32) = centre
    est = HailoPoseEstimator("m.hef", min_confidence=0.5, runner=runner)
    pose = est.estimate(np.zeros((480, 640, 3), dtype=np.uint8))
    assert runner.seen.shape == (64, 64, 3)
    assert set(pose.keypoints) == set(COCO_KEYPOINT_NAMES)
    assert (pose.keypoints["nose"].x, pose.keypoints["nose"].y) == pytest.approx((0.5, 0.5))


def test_hailo_estimator_none_when_nobody() -> None:
    est = HailoPoseEstimator("m.hef", runner=FakeRunner(_yolo_outputs()))
    assert est.estimate(np.zeros((64, 64, 3), dtype=np.uint8)) is None


def test_hailo_estimator_raises_on_wrong_model() -> None:
    est = HailoPoseEstimator("m.hef", runner=FakeRunner({"out": np.zeros((10, 6))}))
    with pytest.raises(RuntimeError, match="YOLOv8-pose"):
        est.estimate(np.zeros((64, 64, 3), dtype=np.uint8))


# --- Pi camera colour order ------------------------------------------------------

def test_pi_camera_returns_rgb(monkeypatch) -> None:
    class FakePicamera2:
        def create_preview_configuration(self, main):
            return main

        def configure(self, config) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def capture_array(self):
            bgr = np.zeros((2, 2, 3), dtype=np.uint8)
            bgr[..., 0] = 255  # "RGB888" stores blue first
            return bgr

    monkeypatch.setitem(sys.modules, "picamera2", types.SimpleNamespace(Picamera2=FakePicamera2))
    from src.camera.pi_source import PiCameraSource

    cam = PiCameraSource(2, 2)
    cam.start()
    frame = cam.read()
    assert frame.flags["C_CONTIGUOUS"]
    assert tuple(frame[0, 0]) == (0, 0, 255)  # blue lands in the last (B) channel
