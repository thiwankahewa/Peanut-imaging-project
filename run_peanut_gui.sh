#!/bin/bash
set -e
PROJECT_DIR="/home/thiwa/Peanut-imaging-project"
LOGFILE="$PROJECT_DIR/autostart.log"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
exec >"$LOGFILE" 2>&1
printf 'Starting Peanut GUI at %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
source "$PROJECT_DIR/.venv/bin/activate"
printf 'Activated virtualenv: %s\n' "$PROJECT_DIR/.venv/bin/python"
exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/main_v5.py"

