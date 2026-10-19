#!/bin/bash
#------------------------------------------------------------------------------
# One-shot setup for a fresh Raspberry Pi OS (Bookworm or later, 64-bit, desktop).
# Modelled on zynthian's setup scripts. Safe to run again.
#
#   git clone https://github.com/vantreeseba/rpi-auto-gym.git && cd rpi-auto-gym
#   scripts/setup_pi.sh [options]
#
# Options:
#   --coral            also install Google Coral support and its MoveNet model
#   --hailo            also install Hailo AI HAT+ support (needs a reboot afterwards)
#   --hostname NAME    reachable as NAME.local (default: "autogym" if the hostname
#                      is still the stock "raspberrypi", otherwise unchanged)
#   --keep-hostname    never change the hostname
#   --no-autostart     don't start the app and web interface on boot
#   --reset-settings   overwrite existing settings with the detected-hardware defaults
#   --dry-run          print the commands instead of running them
#
# Run as your normal desktop user (it uses sudo where needed), not as root.
#------------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CORAL=0
HAILO=0
HOSTNAME_NEW=""
KEEP_HOSTNAME=0
AUTOSTART=1
RESET_SETTINGS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		--coral) CORAL=1 ;;
		--hailo) HAILO=1 ;;
		--hostname) HOSTNAME_NEW="$2"; shift ;;
		--keep-hostname) KEEP_HOSTNAME=1 ;;
		--no-autostart) AUTOSTART=0 ;;
		--reset-settings) RESET_SETTINGS=1 ;;
		--dry-run) DRY_RUN=1 ;;
		-h|--help) sed -n '2,22p' "$0"; exit 0 ;;
		*) echo "Unknown option: $1" >&2; exit 2 ;;
	esac
	shift
done

MEDIAPIPE_MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
CORAL_MODEL_URL="https://github.com/google-coral/test_data/raw/master/movenet_single_pose_lightning_ptq_edgetpu.tflite"

step() { echo; echo "==> $*"; }
run() {
	if [[ $DRY_RUN == 1 ]]; then
		echo "    \$ $*"
	else
		"$@"
	fi
}
have() { command -v "$1" >/dev/null 2>&1; }

if [[ $EUID -eq 0 ]]; then
	echo "Run this as your normal user, not root (it calls sudo itself)." >&2
	exit 1
fi
if ! grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
	echo "Note: this doesn't look like a Raspberry Pi. Continuing anyway; Pi-only steps will be skipped."
fi

#------------------------------------------------------------------------------
step "System packages"
APT_PACKAGES=(
	python3 python3-venv python3-tk python3-pil python3-pil.imagetk
	python3-opencv git wget avahi-daemon
	wlr-randr x11-xserver-utils   # screen rotation / power (Wayland, X11)
)
if apt-cache show python3-picamera2 >/dev/null 2>&1; then
	APT_PACKAGES+=(python3-picamera2 rpicam-apps)
fi
run sudo apt-get update
run sudo apt-get install -y "${APT_PACKAGES[@]}"
run sudo systemctl enable --now avahi-daemon

#------------------------------------------------------------------------------
step "Hostname (for http://<name>.local)"
current_host="$(hostname)"
if [[ $KEEP_HOSTNAME == 0 ]]; then
	if [[ -z "$HOSTNAME_NEW" && "$current_host" == "raspberrypi" ]]; then
		HOSTNAME_NEW="autogym"
	fi
	if [[ -n "$HOSTNAME_NEW" && "$HOSTNAME_NEW" != "$current_host" ]]; then
		if have raspi-config; then
			run sudo raspi-config nonint do_hostname "$HOSTNAME_NEW"
		else
			run sudo hostnamectl set-hostname "$HOSTNAME_NEW"
			run sudo sed -i "s/\b$current_host\b/$HOSTNAME_NEW/g" /etc/hosts
		fi
		echo "    Hostname set to $HOSTNAME_NEW (takes full effect after a reboot)."
		NEED_REBOOT=1
		current_host="$HOSTNAME_NEW"
	fi
fi
echo "    This Pi will be reachable as $current_host.local"

#------------------------------------------------------------------------------
step "Keep the screen on (the app blanks it itself when nobody is in view)"
if have raspi-config; then
	run sudo raspi-config nonint do_blanking 1   # 1 = disable desktop screen blanking
else
	echo "    raspi-config not found, skipping."
fi

#------------------------------------------------------------------------------
step "Python environment (.venv)"
if [[ ! -x "$REPO_DIR/.venv/bin/python" ]]; then
	# --system-site-packages: picamera2, OpenCV and Tk come from apt
	run python3 -m venv --system-site-packages "$REPO_DIR/.venv"
fi
run "$REPO_DIR/.venv/bin/pip" install --upgrade pip
run "$REPO_DIR/.venv/bin/pip" install -e "$REPO_DIR[mediapipe]"
PYTHON="$REPO_DIR/.venv/bin/python"

#------------------------------------------------------------------------------
step "Pose model"
run mkdir -p "$REPO_DIR/models"
if [[ ! -f "$REPO_DIR/models/pose_landmarker_lite.task" ]]; then
	run wget -q -O "$REPO_DIR/models/pose_landmarker_lite.task" "$MEDIAPIPE_MODEL_URL"
else
	echo "    MediaPipe model already present."
fi

#------------------------------------------------------------------------------
if [[ $CORAL == 1 ]]; then
	step "Google Coral"
	if [[ ! -f /etc/apt/sources.list.d/coral-edgetpu.list ]]; then
		run bash -c 'curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg'
		run bash -c 'echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list'
		run sudo apt-get update
	fi
	run sudo apt-get install -y libedgetpu1-std
	run "$REPO_DIR/.venv/bin/pip" install -e "$REPO_DIR[coral]"
	if [[ ! -f "$REPO_DIR/models/movenet_lightning_edgetpu.tflite" ]]; then
		run wget -q -O "$REPO_DIR/models/movenet_lightning_edgetpu.tflite" "$CORAL_MODEL_URL"
	fi
	echo "    Note: pycoral officially stops at Python 3.9; see the README if Coral fails to load."
fi

if [[ $HAILO == 1 ]]; then
	step "Hailo AI HAT+"
	run sudo apt-get install -y hailo-all
	if have hailo-download-resources; then
		run hailo-download-resources --group pose_estimation || true
	fi
	NEED_REBOOT=1
fi

#------------------------------------------------------------------------------
step "Settings from detected hardware"
if [[ $RESET_SETTINGS == 1 ]]; then
	run "$PYTHON" -m src.hardware --apply
else
	run "$PYTHON" -m src.hardware --apply --if-new
fi

#------------------------------------------------------------------------------
if [[ $AUTOSTART == 1 ]]; then
	step "Start on boot"
	run "$REPO_DIR/scripts/install_autostart.sh"
fi

#------------------------------------------------------------------------------
step "Done"
echo "    Web interface: http://$current_host.local:8080"
echo "    Settings file: $AUTOGYM_CONFIG_DIR/autogym_envars.sh"
if [[ "${NEED_REBOOT:-0}" == 1 ]]; then
	echo "    Reboot to finish:  sudo reboot"
else
	echo "    Start now:  scripts/autogym.sh   (or reboot to start automatically)"
fi
