#!/bin/bash
#------------------------------------------------------------------------------
# rpi-auto-gym launcher (modelled on zynthian.sh)
#
# Runs the app in a loop and acts on its exit code:
#   0    stop      -> exit the loop
#   102  restart   -> start again straight away (e.g. after a settings change)
#   *    error     -> show an error screen (with the web address), wait, restart;
#                    after 3 errors in a row, wait longer
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
			# Show what went wrong and the web interface address while waiting,
			# backing off after repeated failures.
			error_counter=$((error_counter + 1))
			wait_s=10
			[[ $error_counter -gt 3 ]] && wait_s=60
			"$PYTHON" -m src.ui.error_screen --code "$status" --attempt "$error_counter" --seconds "$wait_s" \
				|| sleep "$wait_s"
			;;
	esac
done
