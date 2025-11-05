import sounddevice as sd
import numpy as np
import socket
import json
from vosk import Model, KaldiRecognizer

# ---------------- SETTINGS ----------------
UDP_IP = "127.0.0.1"
PORTS = {
    "partial": 7201,  # Partial transcription updates
    "final": 7202,    # Finalized transcription results
}

MODEL_PATH = "models/vosk-model-small-es-0.42"  # Path to downloaded model
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000  # About 0.5 sec blocks
# ------------------------------------------

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
print("✅ Model loaded. Starting real-time recognition.")
# ------------------------------------------------

# ---------------- AUDIO STREAM ----------------
def callback(indata, frames, time, status):
    if status:
        print(f"[AUDIO WARNING] {status}")
    # Convert audio input to int16 PCM bytes
    audio_bytes = (np.asarray(indata, dtype=np.float32).flatten() * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()

    # Feed to recognizer
    if recognizer.AcceptWaveform(audio_bytes):
        # Finalized segment
        result = json.loads(recognizer.Result())
        text = result.get("text", "").strip()
        if text:
            send_udp(text, PORTS["final"])
    else:
        # Partial transcription (in progress)
        partial = json.loads(recognizer.PartialResult())
        ptext = partial.get("partial", "")
        if ptext:
            send_udp(ptext, PORTS["partial"])

# ------------------------------------------------

# ---------------- START STREAM ----------------
with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                    dtype='float32', channels=1, callback=callback):
    print("🎤 Listening... (Press Ctrl+C to stop)")
    import time
    while True:
        time.sleep(0.1)
