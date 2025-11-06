#yamnet_detector.py
import requests
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import time

print("Loading YAMNet model...")
yamnet_model = hub.KerasLayer(
    "https://tfhub.dev/google/yamnet/1",
    trainable=False,
)
print("✅ YAMNet model loaded.")

CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)
try:
    response = requests.get(CLASS_MAP_URL, timeout=5)
    response.raise_for_status()
    CLASS_NAMES = [
        line.split(",")[2].strip('"')
        for line in response.text.splitlines()[1:]
    ]
except Exception as exc:
    print(f"[WARN] Falling back to generic class labels: {exc}")
    CLASS_NAMES = [f"class_{i}" for i in range(521)]

SAMPLE_RATE = 16_000
BLOCK_SIZE = 16_000
CONFIDENCE_THRESHOLD = 0.5  # tweak to taste

def detect_sounds(audio_chunk: np.ndarray):
    waveform = np.asarray(audio_chunk, dtype=np.float32)
    if waveform.size == 0:
        return None, 0.0

    waveform = waveform / 32768.0  # int16 PCM → [-1.0, 1.0]
    scores, _, _ = yamnet_model(waveform)
    scores = scores.numpy()  # shape: (frames, 521)
    if scores.ndim != 2 or scores.shape[0] == 0:
        return None, 0.0

    mean_scores = scores.mean(axis=0)
    top_index = int(np.argmax(mean_scores))
    confidence = float(mean_scores[top_index])
    detected = CLASS_NAMES[top_index] if top_index < len(CLASS_NAMES) else f"class_{top_index}"
    return detected, confidence

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO WARNING] {status}")

    detected, confidence = detect_sounds(indata.flatten())
    if detected and confidence >= CONFIDENCE_THRESHOLD:
        print(f"Detected: {detected} (confidence {confidence:.2f})")

print("🎤 Starting real-time sound detection... (Press Ctrl+C to stop)")
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    dtype="int16",
    channels=1,
    callback=audio_callback,
):
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping sound detection.")