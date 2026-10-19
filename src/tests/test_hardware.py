from pathlib import Path

import pytest

from src import hardware
from src.hardware import HardwareInfo


@pytest.fixture
def fake_sys(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(hardware, "SYS", tmp_path)
    return tmp_path


def _webcam(root: Path, n: int, name: str, index: str, usb: bool) -> None:
    dev = root / ("devices/usb1/1-1" if usb else "devices/platform/codec") / f"v{n}"
    dev.mkdir(parents=True, exist_ok=True)
    node = root / f"sys/class/video4linux/video{n}"
    node.mkdir(parents=True)
    (node / "name").write_text(name + "\n")
    (node / "index").write_text(index + "\n")
    (node / "device").symlink_to(dev)


def test_detect_full_kit(fake_sys: Path) -> None:
    (fake_sys / "proc/device-tree").mkdir(parents=True)
    (fake_sys / "proc/device-tree/model").write_text("Raspberry Pi 5 Model B Rev 1.0\x00")
    _webcam(fake_sys, 0, "bcm2835-codec", "0", usb=False)
    _webcam(fake_sys, 2, "HD Webcam", "0", usb=True)
    _webcam(fake_sys, 3, "HD Webcam", "1", usb=True)  # metadata node
    (fake_sys / "dev").mkdir()
    (fake_sys / "dev/hailo0").touch()
    usb = fake_sys / "sys/bus/usb/devices/2-1"
    usb.mkdir(parents=True)
    (usb / "idVendor").write_text("18d1\n")
    (usb / "idProduct").write_text("9302\n")

    outputs = {
        "rpicam-hello": "Available cameras\n-----------------\n0 : imx708 [4608x2592 10-bit RGGB] (/base/...)\n",
        "hailortcli": "Board Name: Hailo-8\nDevice Architecture: HAILO8L\n",
    }
    info = hardware.detect(run=lambda cmd: outputs.get(cmd[0]))

    assert info.pi_model == "Raspberry Pi 5 Model B Rev 1.0"
    assert info.picameras == ["imx708"]
    assert info.webcams == [(2, "HD Webcam")]
    assert info.hailo == "hailo8l"
    assert info.coral == "usb"


def test_detect_nothing(fake_sys: Path) -> None:
    info = hardware.detect(run=lambda cmd: None)
    assert info == HardwareInfo()


def test_recommend_prefers_hailo_and_picamera() -> None:
    rec = hardware.recommend(HardwareInfo(picameras=["imx708"], webcams=[(0, "cam")], hailo="hailo8", coral="usb"))
    assert rec["AUTOGYM_POSE_BACKEND"] == "hailo"
    assert "hailo8/" in rec["AUTOGYM_POSE_MODEL_PATH"]
    assert rec["AUTOGYM_CAMERA"] == "picamera"


def test_recommend_cpu_webcam() -> None:
    rec = hardware.recommend(HardwareInfo(webcams=[(4, "cam")]))
    assert rec == {"AUTOGYM_POSE_BACKEND": "mediapipe", "AUTOGYM_POSE_MODEL_PATH": hardware.MEDIAPIPE_MODEL,
                   "AUTOGYM_CAMERA": "webcam", "AUTOGYM_CAMERA_INDEX": "4"}


def test_problems() -> None:
    found = hardware.problems(HardwareInfo(pi_model="Pi"), {
        "AUTOGYM_POSE_BACKEND": "hailo", "AUTOGYM_CAMERA": "picamera",
        "AUTOGYM_POSE_MODEL_PATH": "/nope/model.hef"})
    assert len(found) == 3
    assert hardware.problems(HardwareInfo(webcams=[(0, "c")]), {
        "AUTOGYM_POSE_BACKEND": "mediapipe", "AUTOGYM_CAMERA": "webcam",
        "AUTOGYM_POSE_MODEL_PATH": ""}) == []


def test_presets_only_use_known_settings() -> None:
    from src.config import validate
    for preset in hardware.presets():
        _, errors = validate(preset.settings)
        assert errors == {}, preset.id
