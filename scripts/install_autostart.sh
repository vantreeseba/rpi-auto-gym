#!/bin/bash
#------------------------------------------------------------------------------
# Start rpi-auto-gym on boot (run as the desktop user, not root).
#
#  - The app starts with the desktop session via an XDG autostart entry, using
#    scripts/autogym.sh so it restarts after crashes and settings changes.
#  - The web interface runs as a systemd *user* service, so it is up even when
#    the app is not, and keeps running across app restarts.
#
#   scripts/install_autostart.sh            # install
#   scripts/install_autostart.sh --remove   # uninstall
#------------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

AUTOSTART="${XDG_CONFIG_HOME:-$HOME/.config}/autostart/rpi-auto-gym.desktop"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/autogym-webconf.service"

render() {
	sed -e "s|#REPO_DIR#|$REPO_DIR|g" -e "s|#CONFIG_DIR#|$AUTOGYM_CONFIG_DIR|g" "$1"
}

if [[ "${1:-}" == "--remove" ]]; then
	systemctl --user disable --now autogym-webconf.service 2>/dev/null || true
	rm -f "$AUTOSTART" "$UNIT"
	systemctl --user daemon-reload
	echo "Removed autostart entry and web interface service."
	exit 0
fi

mkdir -p "$(dirname "$AUTOSTART")" "$UNIT_DIR" "$AUTOGYM_CONFIG_DIR"
render "$REPO_DIR/sys/autostart/rpi-auto-gym.desktop" > "$AUTOSTART"
render "$REPO_DIR/sys/systemd/autogym-webconf.service" > "$UNIT"

systemctl --user daemon-reload
systemctl --user enable --now autogym-webconf.service
# Let the user service start at boot without anyone logging in.
loginctl enable-linger "$USER" 2>/dev/null || echo "Note: run 'sudo loginctl enable-linger $USER' so the web interface starts at boot."

echo "Installed:"
echo "  app autostart:  $AUTOSTART"
echo "  web interface:  $UNIT  (systemctl --user status autogym-webconf)"
echo "  settings file:  $AUTOGYM_CONFIG_DIR/autogym_envars.sh"
