#!/bin/bash

# Dynamically determine the root directory of the project
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate the virtual environment
source "$BASE_DIR/stt_venv/bin/activate"

# Run the Python script with the dynamically resolved BASE_DIR
python3 "$BASE_DIR/stt_service_2.py" --base_dir "$BASE_DIR"