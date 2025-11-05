#!/bin/bash
# Set the PATH and other environment variables
export PATH="/usr/local/bin:$PATH"
export PYTHONPATH="/Volumes/HENDRIX_SSD/touchdesigner/speech_to_text_5_10_25/stt_venv/lib/python3.12/site-packages:$PYTHONPATH"

# Activate the virtual environment
source /Volumes/HENDRIX_SSD/touchdesigner/speech_to_text_5_10_25/stt_venv/bin/activate

# Run the STT service script and log output
python /Volumes/HENDRIX_SSD/touchdesigner/speech_to_text_5_10_25/stt_service_2.py > /Volumes/HENDRIX_SSD/touchdesigner/speech_to_text_5_10_25/service.log 2>&1