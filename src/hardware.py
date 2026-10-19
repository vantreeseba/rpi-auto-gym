"""Hardware detection and presets (like zynthian's kit profiles + autoconfig).

Detects the Pi model, RPi camera, USB webcams, Hailo and Coral accelerators,
and maps them to settings. Used by the web interface, by scripts/setup_pi.sh,
and at app start to warn when the configured hardware is missing.

    python -m src.hardware            # show what was found and the suggested settings
    python -m src.hardware --apply    # write the suggested settings
    python -m src.hardware --apply --if-new   # only if no settings file exists yet
"""
import argparse
import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

SYS = Path("/")  # replaced in tests

CORAL_USB_IDS = {("1a6e", "089a"), ("18d1", "9302")}  # before / after firmware load
HAILO_MODEL_DIR = "/usr/local/hailo/resources/models"
MEDIAPIPE_MODEL = "models/pose_landmarker_lite.task"
CORAL_MODEL = "models/movenet_lightning_edgetpu.tflite"


@dataclass
class HardwareInfo:
    pi_model: Optional[str] = None
    picameras: List[str] = field(default_factory=list)          # sensor names, e.g. "imx708"
    webcams: List[Tuple[int, str]] = field(default_factory=list)  # (index, name), USB only
    hailo: Optional[str] = None   # "hailo8" / "hailo8l", or "hailo" if the type is unknown
    coral: Optional[str] = None   # "usb" / "pcie"

    def to_dict(self) -> dict:
        return asdict(self)


Runner = Callable[[List[str]], Optional[str]]


def _run(cmd: List[str]) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout + out.stderr


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(errors="replace").strip("\x00\n ")
    except OSError:
        return None


def parse_camera_list(text: str) -> List[str]:
    """Sensor names from `rpicam-hello --list-cameras`, e.g. '0 : imx708 [4608x2592 ...]'."""
    return re.findall(r"^\s*\d+\s*:\s*(\w+)", text, re.MULTILINE)


def parse_hailo_arch(text: str) -> Optional[str]:
    m = re.search(r"Device Architecture:\s*(HAILO\w+)", text)
    return m.group(1).lower() if m else None


def detect(run: Runner = _run) -> HardwareInfo:
    info = HardwareInfo()
    info.pi_model = _read(SYS / "proc/device-tree/model")

    for tool in ("rpicam-hello", "libcamera-hello"):
        text = run([tool, "--list-cameras"])
        if text is not None:
            info.picameras = parse_camera_list(text)
            break

    v4l = SYS / "sys/class/video4linux"
    if v4l.is_dir():
        for node in sorted(v4l.glob("video*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
            device = (node / "device").resolve()
            index_text = _read(node / "index")
            if "/usb" not in str(device) or index_text not in (None, "0"):
                continue  # Pi-internal codec/ISP nodes, or a webcam's secondary metadata node
            info.webcams.append((int(re.sub(r"\D", "", node.name)), _read(node / "name") or node.name))

    if (SYS / "dev/hailo0").exists():
        text = run(["hailortcli", "fw-control", "identify"])
        info.hailo = (parse_hailo_arch(text) if text else None) or "hailo"

    if (SYS / "dev/apex_0").exists():
        info.coral = "pcie"
    else:
        for dev in (SYS / "sys/bus/usb/devices").glob("*"):
            ids = (_read(dev / "idVendor"), _read(dev / "idProduct"))
            if ids in CORAL_USB_IDS:
                info.coral = "usb"
                break
    return info


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    description: str
    settings: Dict[str, str]


def hailo_model(arch: str) -> str:
    """Path to the YOLOv8s pose model for this chip. Install layouts vary, so check a few."""
    folder = arch if arch in ("hailo8", "hailo8l") else "hailo8"
    candidates = [f"{HAILO_MODEL_DIR}/{folder}/yolov8s_pose.hef", f"{HAILO_MODEL_DIR}/yolov8s_pose.hef"]
    return next((c for c in candidates if Path(c).exists()), candidates[0])


def presets() -> List[Preset]:
    return [
        Preset("cpu", "CPU (MediaPipe)", "No accelerator. Works on any Pi 4/5 or desktop.",
               {"AUTOGYM_POSE_BACKEND": "mediapipe", "AUTOGYM_POSE_MODEL_PATH": MEDIAPIPE_MODEL}),
        Preset("coral", "Google Coral", "Coral USB or M.2 TPU running MoveNet Lightning.",
               {"AUTOGYM_POSE_BACKEND": "coral", "AUTOGYM_POSE_MODEL_PATH": CORAL_MODEL}),
        Preset("hailo8", "Hailo-8 (AI HAT+ 26 TOPS)", "YOLOv8s pose on the Hailo-8.",
               {"AUTOGYM_POSE_BACKEND": "hailo", "AUTOGYM_POSE_MODEL_PATH": hailo_model("hailo8")}),
        Preset("hailo8l", "Hailo-8L (AI HAT+ 13 TOPS / AI Kit)", "YOLOv8s pose on the Hailo-8L.",
               {"AUTOGYM_POSE_BACKEND": "hailo", "AUTOGYM_POSE_MODEL_PATH": hailo_model("hailo8l")}),
        Preset("picamera", "RPi Camera Module", "Use the ribbon-cable camera.",
               {"AUTOGYM_CAMERA": "picamera"}),
        Preset("webcam", "USB webcam", "Use the first USB webcam.",
               {"AUTOGYM_CAMERA": "webcam", "AUTOGYM_CAMERA_INDEX": "0"}),
    ]


def preset_by_id(preset_id: str) -> Optional[Preset]:
    return next((p for p in presets() if p.id == preset_id), None)


def recommend(info: HardwareInfo) -> Dict[str, str]:
    """Best settings for what was found: fastest accelerator, Pi camera over webcam."""
    if info.hailo:
        accel = "hailo8l" if info.hailo == "hailo8l" else "hailo8"
    elif info.coral:
        accel = "coral"
    else:
        accel = "cpu"
    settings = dict(preset_by_id(accel).settings)  # type: ignore[union-attr]
    if info.picameras:
        settings["AUTOGYM_CAMERA"] = "picamera"
    elif info.webcams:
        settings["AUTOGYM_CAMERA"] = "webcam"
        settings["AUTOGYM_CAMERA_INDEX"] = str(info.webcams[0][0])
    return settings


def problems(info: HardwareInfo, values: Dict[str, str]) -> List[str]:
    """Plain-language mismatches between the settings and the detected hardware."""
    out = []
    backend = values.get("AUTOGYM_POSE_BACKEND")
    if backend == "hailo" and not info.hailo:
        out.append("Backend is Hailo, but no Hailo device (/dev/hailo0) was found.")
    if backend == "coral" and not info.coral:
        out.append("Backend is Coral, but no Coral TPU was found.")
    camera = values.get("AUTOGYM_CAMERA")
    if camera == "picamera" and not info.picameras and info.pi_model:
        out.append("Camera is the RPi Camera Module, but none was detected.")
    if camera == "webcam" and not info.webcams:
        out.append("Camera is a USB webcam, but none was detected.")
    model = values.get("AUTOGYM_POSE_MODEL_PATH", "")
    if model and camera != "file":
        from .config import repo_dir
        path = Path(model) if Path(model).is_absolute() else repo_dir() / model
        if not path.exists():
            out.append(f"Model file not found: {model}")
    return out


def main() -> None:
    from . import config as conf

    parser = argparse.ArgumentParser(description="Detect hardware and suggest settings")
    parser.add_argument("--apply", action="store_true", help="Write the suggested settings")
    parser.add_argument("--if-new", action="store_true", help="With --apply: only if no settings file exists")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    info = detect()
    suggested = recommend(info)
    if args.json:
        print(json.dumps({"detected": info.to_dict(), "recommended": suggested}, indent=2))
    else:
        print(f"Pi model:   {info.pi_model or 'not a Pi'}")
        print(f"Pi camera:  {', '.join(info.picameras) or 'none'}")
        print(f"Webcams:    {', '.join(f'{i}: {n}' for i, n in info.webcams) or 'none'}")
        print(f"Hailo:      {info.hailo or 'none'}")
        print(f"Coral:      {info.coral or 'none'}")
        print("Suggested settings:")
        for k, v in suggested.items():
            print(f"  {k}={v}")

    if args.apply:
        if args.if_new and conf.envars_path().is_file():
            print(f"Settings file exists, leaving it alone: {conf.envars_path()}")
            return
        conf.write_envars_file(suggested)
        print(f"Wrote {conf.envars_path()}")


if __name__ == "__main__":
    main()
