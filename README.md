# RPI Auto Gym

A wall-mounted Raspberry Pi workout tracker. Point the camera at yourself, and it detects your exercises, counts reps and sets, and displays live feedback on screen — entirely local, no cloud, no account.

---

## What It Does

- Captures camera frames in real time (RPi Camera Module or USB webcam)
- Runs pose estimation to extract body keypoints each frame
- Classifies exercises and counts reps and sets using a rolling time-series of keypoints
- Displays current exercise, rep count, set count, and elapsed time on a wall-mounted screen
- Optionally overlays the pose skeleton on the live camera feed for debugging

All workout data lives in memory for the duration of the session. When the app exits, data is gone. This is intentional for v1 — no database, no files, no cloud sync.

---

## Hardware

### Required

| Component | Notes |
|-----------|-------|
| Raspberry Pi 4 or 5 | Pi 5 recommended for Hailo AI Hat+ |
| Camera | Raspberry Pi Camera Module (v2 or HQ) or USB webcam |
| Display | HDMI monitor or DSI touchscreen, landscape orientation |

### Optional Accelerators

Without an accelerator, pose estimation runs on CPU via MediaPipe. Adding hardware significantly improves inference speed and enables different model families.

| Accelerator | Pose Models | Python Packages | Notes |
|-------------|-------------|-----------------|-------|
| Google Coral TPU (USB or M.2) | PoseNet, MoveNet | `python3-pycoral` (apt), `ai-edge-litert` | `pycoral` archived July 2025; official support ends at Python 3.9. Use pyenv or Docker on Bookworm. |
| Hailo AI Hat+ | YOLO v8/v11 pose | `hailort` (wheel), `hailo_tappas_core_python_binding` (wheel) | Tested with HailoRT + PCIe driver 4.23 and TAPPAS Core 5.1.0. Install via the official RPi AI HAT+ guide (`hailo-all` apt meta-package). |
| None (CPU) | MediaPipe BlazePose | `mediapipe` | No official aarch64 wheel from v0.10.10 onward; works in practice on RPi OS Bookworm 64-bit inside a venv (Python 3.9–3.12). |

---

## Supported Exercises (v1)

- Squats
- Push-ups
- Jumping jacks

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| UI | tkinter (stdlib) |
| Camera | `picamera2` (RPi cam), `opencv-python` (webcam / frame drawing) |
| Pose — CPU | `mediapipe` (Tasks API, BlazePose) |
| Pose — Coral | `python3-pycoral` + `ai-edge-litert` |
| Pose — Hailo | `hailort` + `hailo_tappas_core_python_binding` |
| Exercise classification | Rules-based heuristics (v1); LSTM/1D-CNN upgrade path in v2 |

---

## Architecture

Five independent modules connected by simple interfaces:

```
Camera → Pose Model → Classifier → Session → UI
```

| Module | Path | Responsibility |
|--------|------|----------------|
| Camera | `src/camera/` | Frame capture abstraction (RPi cam, webcam, file) |
| Pose Model | `src/pose/` | Keypoint extraction — backend-agnostic over MediaPipe / Coral / Hailo |
| Classifier | `src/classifier/` | Rolling keypoint stream → exercise name, rep count, set count |
| Session | `src/session/` | In-memory session state aggregating classifier output |
| UI | `src/ui/` | tkinter display: workout view and debug skeleton overlay |

See [agents.md](./agents.md) for full component interfaces, MVP constraints, and parallel development assignments.

---

## Project Structure

```
src/
  camera/        # Frame capture abstraction and adapters
  pose/          # Pose estimation backends (MediaPipe, Coral, Hailo)
  classifier/    # Exercise detection and rep/set counting
  session/       # In-memory session state
  ui/            # tkinter display app
  main.py        # Entry point — wires all components together
src_test/        # Legacy browser prototype (ml5.js + MoveNet) — reference only
agents.md        # Component interfaces and developer assignments
README.md
pyproject.toml
```

---

## Installing on Raspberry Pi

### 1. System dependencies

```bash
sudo apt update && sudo apt upgrade -y

# Python, venv, tkinter, git
sudo apt install -y python3 python3-venv python3-tk git

# OpenCV system libraries (needed by opencv-python)
sudo apt install -y libopencv-dev python3-opencv

# picamera2 (only needed if using RPi Camera Module)
sudo apt install -y python3-picamera2
```

### 2. Clone the repo

```bash
git clone https://github.com/cubicecho/rpi-auto-gym.git
cd rpi-auto-gym
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

> `--system-site-packages` gives the venv access to system-installed packages like `picamera2` and `python3-opencv`, which are easier to install via apt than pip on ARM.

### 4. Install Python dependencies

```bash
# Core + MediaPipe (CPU backend, works on Bookworm 64-bit)
pip install -e ".[mediapipe]"
```

### 5. Download a pose model

```bash
mkdir -p models
# MediaPipe Lite model (~4 MB, fastest)
wget -q -O models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

### 6. (Optional) Coral TPU

```bash
# Add Google's APT repo and install pycoral
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install -y libedgetpu1-std python3-pycoral

pip install ai-edge-litert
```

> **Python version note:** `pycoral` officially supports Python ≤ 3.9. On Bookworm (Python 3.11/3.12), use `pyenv` to install Python 3.9 and create your venv with that interpreter, or use the Docker approach documented on the [bret.dk guide](https://bret.dk/installing-pycoral-for-google-coral-on-raspberry-pi-5/).

### 7. (Optional) Hailo AI Hat+

Follow the [official RPi AI HAT+ setup guide](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html) first, then:

```bash
# The hailo-all meta-package installs HailoRT, TAPPAS Core, and Python bindings
sudo apt install -y hailo-all
sudo reboot

# Verify the chip is recognised
hailortcli fw-control identify

# Download YOLO pose model (HEF format)
hailo-download-resources --group pose_estimation
```

> Tested with HailoRT + PCIe driver 4.23 and TAPPAS Core 5.1.0.

---

## Running

Make sure your venv is active (`source .venv/bin/activate`) and you're in the repo root.

### USB webcam + MediaPipe (CPU) — easiest starting point

```bash
python -m src.main
```

### RPi Camera Module + MediaPipe

```bash
python -m src.main --camera picamera
```

### Video file (no camera needed — good for testing)

```bash
python -m src.main --camera file --camera-path /path/to/workout.mp4
```

### Coral TPU

```bash
python -m src.main --backend coral \
  --model-path models/movenet_lightning_edgetpu.tflite
```

### Hailo AI Hat+

```bash
python -m src.main --backend hailo \
  --model-path /usr/local/hailo/resources/models/hailo8/yolov8s_pose.hef
```

### All options

```
usage: main.py [-h] [--backend {mediapipe,coral,hailo}]
               [--camera {file,webcam,picamera}]
               [--camera-path PATH] [--model-path PATH]

options:
  --backend     Pose estimation backend (default: mediapipe)
  --camera      Camera source (default: webcam)
  --camera-path Path to image/video file (required when --camera=file)
  --model-path  Path to pose model file (default: models/pose_landmarker_lite.task)
```

---

## Running Tests

```bash
python -m pytest src/ -v
```

Tests run fully headless (no camera or display required). 68 tests across all 5 modules.

---

## Autostart on Boot (optional)

To launch automatically when the Pi boots to desktop:

```bash
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/rpi-auto-gym.desktop << EOF
[Desktop Entry]
Type=Application
Name=RPI Auto Gym
Exec=/home/pi/rpi-auto-gym/.venv/bin/python -m src.main
WorkingDirectory=/home/pi/rpi-auto-gym
EOF
```

Adjust the paths if your username or install location differs.
