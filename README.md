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
| Language | Python 3.9+ (3.11 on Pi OS Bookworm) |
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
  webconf/       # Web interface for power users (separate process)
  config.py      # Settings: schema, envars file read/write
  hardware.py    # Detects camera / accelerators, hardware presets
  display.py     # Screen rotation and blanking
  netinfo.py     # Hostname / IP for "open http://autogym.local:8080"
  control.py     # Local control API the web interface talks to
  main.py        # Entry point — wires all components together
scripts/         # Launcher, dev mode, web interface and autostart install scripts
sys/             # systemd unit and autostart templates
src_test/        # Legacy browser prototype (ml5.js + MoveNet) — reference only
agents.md        # Component interfaces and developer assignments
README.md
pyproject.toml
```

---

## Installing on Raspberry Pi

### Quick setup (recommended)

On a fresh Raspberry Pi OS (Bookworm or later, 64-bit, desktop):

```bash
git clone https://github.com/vantreeseba/rpi-auto-gym.git
cd rpi-auto-gym
scripts/setup_pi.sh            # add --coral or --hailo if you have an accelerator
```

It installs the system packages, creates `.venv`, downloads the pose model, picks
settings for the camera and accelerator it finds, turns off desktop screen blanking,
and sets the app and web interface to start on boot. If the hostname is still the
stock `raspberrypi`, it renames the Pi to `autogym` so the web interface is at
**http://autogym.local:8080** (use `--hostname NAME` or `--keep-hostname` to choose).
Run `scripts/setup_pi.sh --help` for all options, or `--dry-run` to see what it would do.

The steps below are the manual equivalent.

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

### On your desktop (dev / emulator mode) — easiest starting point

No Pi needed. Runs the app in a window with debug logging, plus the web interface:

```bash
scripts/autogym_dev.sh                         # USB webcam
scripts/autogym_dev.sh path/to/workout.mp4     # recorded video, on loop
```

Then open http://127.0.0.1:8080. Dev mode keeps its settings in `./.dev-config/`,
separate from a real install.

| Key | Action |
|-----|--------|
| `d` | Open/close the debug view (camera + skeleton) |
| `r` | Reset the session |
| `F11` | Toggle fullscreen |
| `q` / `Esc` | Quit |

Without a display (e.g. over SSH), the app runs headless and you watch it from the web interface.

### On the Pi

```bash
scripts/autogym.sh          # the app, restarted automatically after crashes or settings changes
scripts/webconf.sh          # the web interface (separate process, so it stays up while the app restarts)
```

`python -m src.main` still works on its own; the launcher just adds the restart loop.

### Command-line flags

Flags override the settings file for one run:

```
python -m src.main [--backend {mediapipe,coral,hailo}] [--camera {file,webcam,picamera}]
                   [--camera-path PATH] [--model-path PATH] [--windowed] [--debug]
```

Examples:

```bash
python -m src.main --camera picamera
python -m src.main --camera file --camera-path /path/to/workout.mp4 --windowed
python -m src.main --backend coral --model-path models/movenet_lightning_edgetpu.tflite
python -m src.main --backend hailo --model-path /usr/local/hailo/resources/models/hailo8/yolov8s_pose.hef
```

---

## Settings and the web interface

Settings live in one shell-style file, `~/.config/rpi-auto-gym/autogym_envars.sh`
(override the folder with `AUTOGYM_CONFIG_DIR`). Each line is `export AUTOGYM_...=value`,
so it is easy to edit by hand or `source` from scripts. The full list, with defaults and
help text, is `SETTINGS` in [src/config.py](src/config.py).

Order of precedence: defaults → settings file → environment variables → command-line flags.

The web interface (default port 8080) has three pages:

- **Dashboard**: live exercise, reps, sets and FPS, the camera feed with the pose skeleton,
  and buttons to reset the session, toggle the debug view, restart or stop the app.
- **Hardware** (on the dashboard): what was detected (Pi model, camera, webcams, Hailo, Coral),
  warnings when the settings don't match it (e.g. backend is Hailo but there's no Hailo,
  or the model file is missing), and one-click presets ("CPU", "Coral", "Hailo-8", "Hailo-8L",
  "Pi camera", "USB webcam"). `python -m src.hardware` shows the same from the command line.
- **Settings**: every setting as a form. "Save & restart app" applies changes straight away.
  This includes the rep-counting thresholds (knee/elbow angles for squats and push-ups),
  screen rotation, and blanking the screen when nobody is in view.
- **Logs**: the app's log file, with follow mode.

Set `AUTOGYM_WEB_PASSWORD` to require a login, or `AUTOGYM_WEB_HOST=127.0.0.1` to keep it
off the network. Changes to the web interface's own address, port or password need it restarted
(`systemctl --user restart autogym-webconf`).

How the pieces talk:

```
browser ──HTTP──▶ webconf (src/webconf, :8080) ──HTTP, localhost only──▶ app control API (src/control.py, :8765)
                        │                                                     │
                        └──── reads/writes autogym_envars.sh ◀── read at start ┘
```

The app exits with code 102 to ask `scripts/autogym.sh` for a restart, 0 to stop.
On any other exit code the launcher shows a full-screen error with the last error from the
log and the web interface's address, then retries (every 10s, every 60s after 3 failures).
Press any key on the error screen to retry straight away.

---

## Running Tests

```bash
python -m pytest src/ -v
```

Tests run fully headless (no camera or display required).

---

## Autostart on Boot

```bash
scripts/install_autostart.sh            # install
scripts/install_autostart.sh --remove   # uninstall
```

This adds a desktop autostart entry that runs `scripts/autogym.sh` when the Pi logs in to
the desktop, and a systemd user service for the web interface (`systemctl --user status autogym-webconf`).
Run it as the desktop user, not root.
