#!/usr/bin/env python3

import os
import csv
import argparse
import sounddevice as sd
import numpy as np
import socket
import json
import threading
import time
from vosk import Model, KaldiRecognizer
from yamnet_detector import YAMNetDetector

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", required=True, help="Base directory of the project")
args = parser.parse_args()

# Use the base directory passed from the service.sh script
BASE_DIR = args.base_dir

# Function to read configuration from the CSV file
def read_config_from_csv():
    config = {}
    try:
        config_file = os.path.join(BASE_DIR, "config.csv")  # Relative path to the config.csv file
        with open(config_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                key, value = row
                config[key] = int(value) if value.isdigit() else value  # Convert numeric values to integers
    except Exception as e:
        print(f"[CONFIG ERROR] Failed to read from CSV: {e}")
    return config

# Load configuration
config = read_config_from_csv()
SAMPLE_RATE = config.get("SAMPLE_RATE", 16000)
MODEL_PATH = os.path.join(BASE_DIR, config.get("MODEL_PATH", "models/vosk-model-small-en-us-0.15"))
BLOCK_SIZE = config.get("BLOCK_SIZE", 4000)
MAX_WORDS_PER_CHUNK = config.get("MAX_WORDS", 16)

# UDP Settings
UDP_IP = "127.0.0.1"
PORTS = {
    "partial": 7201,       # Partial transcription updates
    "final": 7202,         # Finalized transcription results
    "word_conf": 7203,     # Word + confidence stream
}

# Shared UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_udp(message: str, port: int) -> None:
    try:
        if message:
            sock.sendto(message.encode("utf-8"), (UDP_IP, port))
    except Exception as e:
        print(f"[UDP SEND ERROR] port={port} error={e}")

# ---------------- INITIALIZATION ----------------
print(f"Loading Vosk model from {MODEL_PATH} ...")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)
recognizer.SetWords(True)
print("✅ Vosk model loaded.")

# Initialize YAMNet detector
print("Initializing YAMNet sound detector...")
yamnet_detector = YAMNetDetector(sample_rate=SAMPLE_RATE, confidence_threshold=0.3)
yamnet_detector.start()
print("✅ YAMNet detector started.")
print("Starting real-time recognition with integrated sound detection.")
# ------------------------------------------------

# ---------------- AUDIO STREAM ----------------
def callback(indata, frames, time, status):
    if status:
        print(f"[AUDIO WARNING] {status}")
    
    # Get audio data as float32 array
    audio_float = np.asarray(indata, dtype=np.float32).flatten()
    
    # Share audio with YAMNet detector (non-blocking)
    yamnet_detector.process_audio(audio_float)
    
    # Convert audio input to int16 PCM bytes for Vosk
    audio_bytes = (audio_float * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()

    # Feed to recognizer (STT has real-time priority in main thread)
    if recognizer.AcceptWaveform(audio_bytes):
        # Finalized segment
        result = json.loads(recognizer.Result())
        text = result.get("text", "").strip()
        if text:
            send_udp(text, PORTS["final"])  # Send finalized transcription
        # Send word + confidence for each word in the finalized result
        for word in result.get("result", []):
            word_conf = {
                "word": word.get("word"),
                "confidence": word.get("conf"),
                "start": word.get("start"),
                "end": word.get("end"),
            }
            send_udp(json.dumps(word_conf), PORTS["word_conf"])  # Send word + confidence
    else:
        # Partial transcription (in progress)
        partial = json.loads(recognizer.PartialResult())
        ptext = partial.get("partial", "")
        if ptext:
            # Split the partial transcription into smaller chunks
            words = ptext.split()  # Split the transcription into individual words

            # Send each chunk of words as a separate partial update
            for i in range(0, len(words), MAX_WORDS_PER_CHUNK):
                chunk = " ".join(words[i:i + MAX_WORDS_PER_CHUNK])  # Create a chunk of MAX_WORDS_PER_CHUNK words
                send_udp(chunk, PORTS["partial"])  # Send the chunk as a partial transcription

                # Send each word in the chunk with confidence as None
                for word in chunk.split():
                    word_conf = {
                        "word": word,
                        "confidence": None,  # Confidence is not available for partial results
                    }
                    send_udp(json.dumps(word_conf), PORTS["word_conf"])  # Send word

# ------------------------------------------------

# ---------------- START STREAM ----------------
try:
    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                        dtype='float32', channels=1, callback=callback):
        print("🎤 Listening... (Press Ctrl+C to stop)")
        print(f"📡 STT output: UDP ports {PORTS['partial']}, {PORTS['final']}, {PORTS['word_conf']}")
        print(f"🔊 Sound detection output: UDP port {yamnet_detector.udp_port}")
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n⏹️  Stopping services...")
    yamnet_detector.stop()
    sock.close()
    print("✅ Services stopped gracefully")
except Exception as e:
    print(f"❌ Error: {e}")
    yamnet_detector.stop()
    sock.close()
    raise
