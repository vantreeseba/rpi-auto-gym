# Shared by the scripts in this folder: finds the repo and its Python.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
	PYTHON="$REPO_DIR/.venv/bin/python"
else
	PYTHON="${PYTHON:-python3}"
fi
export AUTOGYM_CONFIG_DIR="${AUTOGYM_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/rpi-auto-gym}"
cd "$REPO_DIR"
