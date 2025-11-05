#!/usr/bin/env bash
# filepath: run_stt.sh
set -e
# Activate venv and run stt_vosk in mock mode
source stt_venv/bin/activate
python3 src/stt_vosk.py configs/configvosk.csv
