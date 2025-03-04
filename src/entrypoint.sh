#!/bin/bash
set -e

# Start the handler
echo "Starting the handler..."
exec python3.11 -u /rp_handler.py "$@" 