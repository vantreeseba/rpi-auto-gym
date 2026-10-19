#!/bin/bash
#------------------------------------------------------------------------------
# rpi-auto-gym launcher (modelled on zynthian.sh)
#
# Runs the app in a loop and acts on its exit code:
#   0    stop      -> exit the loop
#   102  restart   -> start again straight away (e.g. after a settings change)
#   *    error     -> wait, then restart; after 3 errors in a row, wait longer
#
# Extra arguments are passed to the app, e.g.  scripts/autogym.sh --windowed
#------------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

EXIT_STOP=0
EXIT_RESTART=102
error_counter=0

trap 'kill -TERM "$app_pid" 2>/dev/null; wait "$app_pid"; exit 0' INT TERM

while true; do
	"$PYTHON" -m src.main "$@" &
	app_pid=$!
	wait "$app_pid"
	status=$?
	echo "autogym: app exited with status $status"

	case $status in
		$EXIT_STOP)
			break
			;;
		$EXIT_RESTART)
			error_counter=0
			;;
		*)
			error_counter=$((error_counter + 1))
			if [[ $error_counter -gt 3 ]]; then
				echo "autogym: $error_counter errors in a row, waiting 30s (check the log or the web interface)"
				sleep 30
			else
				sleep 3
			fi
			;;
	esac
done
