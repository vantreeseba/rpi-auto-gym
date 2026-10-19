#!/bin/bash
#------------------------------------------------------------------------------
# Dev / emulator mode (like zynthian_emu.sh + zynthian_debug.sh)
#
# Runs the app in a window with debug logging, plus the web interface,
# on any Linux/macOS desktop. No Pi hardware needed.
#
#   scripts/autogym_dev.sh                      # USB webcam
#   scripts/autogym_dev.sh path/to/video.mp4    # play a recorded workout on loop
#
# Keys in the app window: d debug view, r reset session, F11 fullscreen, q quit.
# Uses its own config dir (./.dev-config) so it never touches the Pi's settings.
#------------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

export AUTOGYM_CONFIG_DIR="${AUTOGYM_DEV_CONFIG_DIR:-$REPO_DIR/.dev-config}"
export AUTOGYM_UI_FULLSCREEN="${AUTOGYM_UI_FULLSCREEN:-0}"
export AUTOGYM_UI_HIDE_CURSOR="${AUTOGYM_UI_HIDE_CURSOR:-0}"
export AUTOGYM_LOG_LEVEL="${AUTOGYM_LOG_LEVEL:-DEBUG}"
export AUTOGYM_WEB_HOST="${AUTOGYM_WEB_HOST:-127.0.0.1}"

args=()
if [[ -n "$1" && -f "$1" ]]; then
	args+=(--camera file --camera-path "$1")
	shift
fi

"$PYTHON" -m src.webconf &
web_pid=$!
trap 'kill "$web_pid" 2>/dev/null' EXIT

echo "Web interface: http://127.0.0.1:${AUTOGYM_WEB_PORT:-8080}"
"$(dirname "${BASH_SOURCE[0]}")/autogym.sh" "${args[@]}" "$@"
