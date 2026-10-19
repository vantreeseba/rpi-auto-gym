#!/bin/bash
# Start the web interface (python -m src.webconf). Arguments: [--host H] [--port P]
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
exec "$PYTHON" -m src.webconf "$@"
