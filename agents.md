# Agents

This document describes the five independent components of rpi-auto-gym. Each section covers the component's technical contract and its developer/agent assignment.

**MVP-first mandate:** Every component must resist scope creep. Build the simplest thing that satisfies the interface contract. Do not add features, configuration, or abstraction layers that are not explicitly required. When in doubt, leave it out.

---

## Data Flow

```
Camera → Pose Model → Classifier → Session → UI
```

Each arrow is a simple Python interface. Any component can be developed and tested in isolation by mocking the interfaces on either side of it.

---

## Parallelism

All five components can be started on the same day. None require another to be complete — only the interface contracts must be agreed upon first (they are defined below).

| Component | Blocks on | Mock strategy for isolated dev |
|-----------|-----------|-------------------------------|
| Camera | Hardware availability | `FileCameraSource` — reads frames from an image or video file |
| Pose Model | Camera | Pass `np.zeros((480, 640, 3), dtype=np.uint8)` or a recorded frame |
| Classifier | Pose Model | Replay a `.csv` of recorded keypoint sequences |
| Session | Classifier | Hardcode a `ClassifierResult` with fixed values |
| UI | Session | Construct a static `SessionState` dataclass instance |

---

## 1. Camera — `src/camera/`

### Purpose

Abstract camera input. Hide the difference between RPi Camera Module, USB webcam, and test fixtures. Emit raw frames as numpy arrays.

### Interface

```python
import numpy as np

class CameraSource:
    def start(self) -> None: ...
    def read(self) -> np.ndarray: ...   # H×W×3, RGB, uint8
    def stop(self) -> None: ...
```

### Adapters

| Class | Backend | Use case |
|-------|---------|----------|
| `FileCameraSource` | Static image or video file via `cv2.VideoCapture` | Dev and testing without hardware — implement first |
| `WebcamSource` | USB webcam via `cv2.VideoCapture(index)` | Desktop dev, USB cameras |
| `PiCameraSource` | RPi Camera Module via `picamera2` | Production on RPi |

### MVP Constraints

- **No** frame buffering beyond the single latest frame
- **No** streaming server or network output
- **No** config file parsing — pass parameters directly
- **No** multi-camera support
- Implement `FileCameraSource` first so all other components can run without hardware

### Can Be Developed In Parallel With

Everything. Other components depend only on the `CameraSource` interface, not its internals.

### Notes

- Output is always RGB `uint8` — convert internally if the backend gives BGR (OpenCV default)
- `FileCameraSource` should loop the video file when it reaches the end, to support long dev sessions

---

## 2. Pose Model — `src/pose/`

### Purpose

Accept a single frame, return normalized pose keypoints. The interface is identical regardless of which hardware backend is in use.

### Interface

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Keypoint:
    x: float           # normalized 0.0–1.0 (left→right)
    y: float           # normalized 0.0–1.0 (top→bottom)
    confidence: float  # 0.0–1.0

@dataclass
class Pose:
    keypoints: dict[str, Keypoint]  # COCO keypoint names as keys (see note)
    timestamp: float                # time.monotonic()

class PoseEstimator:
    def estimate(self, frame: np.ndarray) -> Pose | None: ...
    # Returns None if no person is detected
```

COCO keypoint names: `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`

### Backends

Select backend via CLI flag or environment variable (e.g. `--backend mediapipe|coral|hailo`).

| Class | Accelerator | Model | Package(s) |
|-------|-------------|-------|-----------|
| `MediaPipePoseEstimator` | None (CPU) | BlazePose (Lite/Full/Heavy `.task` files) | `mediapipe` |
| `CoralPoseEstimator` | Google Coral TPU | MoveNet Lightning or Thunder (`.tflite` EdgeTPU) | `python3-pycoral`, `ai-edge-litert` |
| `HailoPoseEstimator` | Hailo AI Hat+ | YOLOv8/v11 pose (`.hef`) | `hailort`, `hailo_tappas_core_python_binding` |

All backends emit the same `Pose` format. The rest of the app is backend-agnostic.

### MVP Constraints

- **No** multi-person support — return the highest-confidence single person only
- **No** model hot-swap at runtime
- **No** model download logic — models must be present on disk; document expected paths
- Implement `MediaPipePoseEstimator` first (no special hardware needed)

### Can Be Developed In Parallel With

- Camera (mock with dummy numpy frames)
- UI (UI only needs the `Pose` dataclass shape, not the estimator itself)

### Notes

- MediaPipe BlazePose outputs 33 landmarks; map down to the 17 COCO keypoints above before returning
- Hailo models output YOLO pose format; normalize coordinates to 0–1 before returning
- Coral MoveNet output tensor shape: `[1, 1, 17, 3]` — `(y, x, score)` per keypoint; swap x/y when populating `Keypoint`

---

## 3. Exercise Classifier — `src/classifier/`

### Purpose

Maintain a rolling window of `Pose` frames and detect the current exercise. Count reps and sets.

### Interface

```python
from dataclasses import dataclass, field
from collections import deque

@dataclass
class ClassifierResult:
    exercise: str | None   # "squat" | "pushup" | "jumping_jack" | None
    phase: str | None      # exercise-specific phase, e.g. "down" | "up"
    rep_count: int         # reps completed in the current set
    set_count: int         # sets completed this session

class ExerciseClassifier:
    def update(self, pose: Pose) -> ClassifierResult: ...
    def reset(self) -> None: ...
```

### Supported Exercises (v1)

`squat`, `pushup`, `jumping_jack` — implement in this order.

### Implementation Strategy

**Phase 1 — Rules-based heuristics (build this now):**

Each exercise has a simple state machine driven by joint angles and relative positions:

| Exercise | Key signals |
|----------|------------|
| Squat | Knee angle and hip height relative to standing baseline; transitions: `standing → down → standing` = 1 rep |
| Push-up | Elbow angle and nose-to-wrist vertical distance; transitions: `up → down → up` = 1 rep |
| Jumping jack | Wrist height above hip and foot spread relative to hip width; transitions: `closed → open → closed` = 1 rep |

Set detection: after N reps with a pause (no movement for ~3 seconds), increment set count and reset rep count.

**Phase 2 — ML upgrade path (do not build now):**

Replace the rules with an LSTM or 1D-CNN trained on labeled keypoint sequences. The `ExerciseClassifier` interface stays identical — it is a drop-in replacement. A data recording script (`tools/record_session.py`) should be added in Phase 2 to collect training data.

### MVP Constraints

- **Three exercises only** — no other exercises in v1
- **No** confidence scores on predictions
- **No** simultaneous multi-exercise detection
- **No** training data collection in v1
- Implement `squat` end-to-end before adding the others

### Can Be Developed In Parallel With

Camera and Pose Model. Replay a `.csv` of pre-recorded keypoint sequences to drive `update()` during development — no live camera needed.

### Notes

- Window size: ~30–60 frames (1–2 seconds at 30 fps) is a reasonable starting point
- Angles between joints: use `atan2` on keypoint coordinates — do not use raw pixel distances
- Low-confidence keypoints (< 0.3) should be treated as missing; handle gracefully

---

## 4. Session — `src/session/`

### Purpose

In-memory store for the current workout session. Aggregate `ClassifierResult` updates into per-exercise rep and set history. Bridge between the Classifier and the UI.

### Interface

```python
from dataclasses import dataclass, field
import time

@dataclass
class ExerciseLog:
    sets: int = 0
    total_reps: int = 0

@dataclass
class SessionState:
    active_exercise: str | None = None
    current_rep_count: int = 0
    exercises: dict[str, ExerciseLog] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

class Session:
    def update(self, result: ClassifierResult) -> SessionState: ...
    def reset(self) -> None: ...
    def get_state(self) -> SessionState: ...
```

### MVP Constraints

- **No** file I/O or persistence of any kind
- **No** history beyond the current session
- **No** undo or correction of counts
- All data is gone when the process exits — this is by design

### Can Be Developed In Parallel With

- UI (share the `SessionState` dataclass; UI can use a static instance for dev)
- Classifier (drive `update()` with hardcoded `ClassifierResult` values)

### Notes

- `Session` is the single shared object between Classifier and UI — keep it thread-safe if the UI runs in its own thread (use `threading.Lock` on writes)
- `elapsed_seconds` should be computed from `time.monotonic()` relative to session start, not accumulated from frame deltas

---

## 5. UI — `src/ui/`

### Purpose

tkinter display app for the wall-mounted RPi screen. Show the current exercise, rep count, set count, elapsed time, and an optional debug skeleton overlay.

### Interface

```python
import tkinter as tk

class WorkoutApp(tk.Tk):
    def __init__(self, session: Session) -> None: ...
    def run(self) -> None: ...   # blocks; starts tkinter mainloop
```

The app polls `session.get_state()` on a timer (e.g. every 100 ms via `after()`) and updates labels in place.

### Screens

**Workout view (default)**
- Current exercise name (large, centred)
- Rep counter (very large)
- Set counter
- Elapsed time

**Debug overlay (toggle)**
- Live camera feed displayed in a `tk.Canvas`
- Pose skeleton drawn over the frame using `cv2` before passing to tkinter
- Toggle with a button or keyboard shortcut

### MVP Constraints

- **No** animations or transitions
- **No** touch gesture recognition
- **No** settings screen
- **No** themes or style configuration
- Large, readable text only — this is a functional display, not a polished product
- Design for **landscape orientation**
- Implement a **headless mode** (`DISPLAY` env var unset → skip `mainloop`, render to offscreen surface) for CI/testing without a display

### Can Be Developed In Parallel With

Everything. Construct a static `SessionState` instance with fixed values to drive the UI during development.

### Notes

- Use `tk.Label` with large fonts for rep/set display — no canvas needed for the workout view
- Debug overlay requires `opencv-python` to draw keypoints/lines onto the frame before converting to a `tk.PhotoImage`
- `after(100, self._poll)` is the standard tkinter pattern for periodic updates without blocking the mainloop
- Test on the actual RPi display early — font sizes that look fine on a desktop monitor are often too small on a 7" DSI panel

---

## Cross-cutting: settings, control API, web interface

Modelled on zynthian (envars file + launcher loop + separate "webconf" web app).

| Piece | Path | Responsibility |
|-------|------|----------------|
| Settings | `src/config.py` | `SETTINGS` schema, read/write `autogym_envars.sh`, typed `Config` |
| Control API | `src/control.py` | Localhost-only HTTP API in the app: state, frame JPEG, actions |
| Web interface | `src/webconf/` | Separate process: dashboard, settings form, logs. Stdlib only |
| Launcher | `scripts/autogym.sh` | Restart loop driven by exit codes (0 stop, 102 restart); error screen between retries (`src/ui/error_screen.py`) |
| Hardware | `src/hardware.py` | Detect camera/accelerators, presets, mismatch warnings |
| Display | `src/display.py` | Rotation at start-up, blanking when nobody is in view |
| Pi setup | `scripts/setup_pi.sh` | Fresh Pi OS → running install, hostname for `<name>.local` |

Components still take plain constructor arguments; only `main.py` reads `Config`.
To expose a new tunable, add a `Setting` to `SETTINGS` and pass it through in `main.py`.
The web form picks it up automatically.
