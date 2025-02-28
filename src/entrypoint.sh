#!/bin/bash
set -e

# Download models if they don't exist
echo "Checking for models and downloading if necessary..."
python3.11 /download_models.py

# Start the handler
echo "Starting the handler..."
exec python3.11 -u /rp_handler.py "$@" 