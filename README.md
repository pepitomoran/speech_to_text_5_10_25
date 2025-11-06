# Speech-to-Text with Integrated Sound Detection

This project integrates speech-to-text (STT) and sound detection services for use with TouchDesigner. The services run concurrently in separate threads, sharing a common audio source.

## Features

- **Real-time Speech-to-Text**: Uses Vosk for fast, accurate speech recognition
- **Sound Event Detection**: YAMNet-based detection of 521 sound classes (speech, music, dog barking, etc.)
- **Threaded Architecture**: STT runs with real-time priority; sound detection in a separate thread
- **UDP Output**: Both services send results via UDP for easy integration with TouchDesigner
- **Shared Audio Source**: Single audio input feeds both services efficiently

## Architecture

```
┌─────────────────────┐
│   Audio Input       │
│  (TouchDesigner/    │
│   Microphone)       │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │   Callback   │
    │   (Main      │
    │   Thread)    │
    └──┬────────┬──┘
       │        │
       │        └─────────────────┐
       │                          │
       ▼                          ▼
┌─────────────┐          ┌────────────────┐
│ STT Service │          │ YAMNet Thread  │
│ (Vosk)      │          │ (TensorFlow)   │
│ Real-time   │          │ Background     │
│ Priority    │          │                │
└──────┬──────┘          └────────┬───────┘
       │                          │
       ▼                          ▼
    UDP 7201-7203              UDP 7204
    (Transcription)         (Sound Events)
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd speech_to_text_5_10_25
   ```

2. **Set up the virtual environment:**
   ```bash
   python3 -m venv stt_venv
   source stt_venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download Vosk model** (if not already present):
   ```bash
   mkdir -p models
   cd models
   # Download model from https://alphacephei.com/vosk/models
   # Extract to models/vosk-model-small-pt-0.3 (or update config.csv)
   ```

## Configuration

Edit `config.csv` to adjust settings:

```csv
Key,Value
SAMPLE_RATE,16000
MODEL_PATH,models/vosk-model-small-pt-0.3
BLOCK_SIZE,4000
MAX_WORDS,16
```

- **SAMPLE_RATE**: Audio sample rate (16000 Hz recommended)
- **MODEL_PATH**: Path to Vosk model directory
- **BLOCK_SIZE**: Audio buffer size (affects latency)
- **MAX_WORDS**: Maximum words per partial transcription chunk

## Running the Services

### Start Both Services

```bash
./service.sh
```

This launches both STT and YAMNet detection services in a single process.

### Expected Output

```
Loading Vosk model from models/vosk-model-small-pt-0.3 ...
✅ Vosk model loaded.
Initializing YAMNet sound detector...
[YAMNet] Loading model from TensorFlow Hub...
[YAMNet] ✅ Model loaded successfully
[YAMNet] Service started
✅ YAMNet detector started.
Starting real-time recognition with integrated sound detection.
🎤 Listening... (Press Ctrl+C to stop)
📡 STT output: UDP ports 7201, 7202, 7203
🔊 Sound detection output: UDP port 7204
```

### Stop Services

Press `Ctrl+C` to gracefully stop both services.

## UDP Output Ports

### STT Service (Vosk)

- **Port 7201**: Partial transcription (real-time, in-progress text)
- **Port 7202**: Final transcription (complete, finalized sentences)
- **Port 7203**: Word-level data with confidence scores

#### Message Format (Port 7203)
```json
{
  "word": "hello",
  "confidence": 0.95,
  "start": 1.2,
  "end": 1.5
}
```

### Sound Detection Service (YAMNet)

- **Port 7204**: Detected sound events

#### Message Format
```json
{
  "event": "Music",
  "confidence": 0.87,
  "timestamp": 1699281234.567
}
```

## TouchDesigner Integration

### Setup in TouchDesigner

1. **Create UDP In CHOPs** for each port:
   - Add a `UDP In CHOP` for port 7201 (partial transcription)
   - Add a `UDP In CHOP` for port 7202 (final transcription)
   - Add a `UDP In CHOP` for port 7203 (word confidence)
   - Add a `UDP In CHOP` for port 7204 (sound events)

2. **Configure UDP In CHOPs**:
   - Network Address: `127.0.0.1`
   - Network Port: Set to corresponding port
   - Protocol: UDP

3. **Parse JSON data** using a `DAT Execute` or `Text DAT`:
   ```python
   import json
   
   def onReceive(dat, rowIndex, cols, cells):
       msg = cells[0].val
       try:
           data = json.loads(msg)
           # Process STT data
           if 'word' in data:
               print(f"Word: {data['word']}, Confidence: {data['confidence']}")
           # Process sound event data
           elif 'event' in data:
               print(f"Sound: {data['event']}, Confidence: {data['confidence']}")
       except:
           # Handle plain text (ports 7201, 7202)
           print(f"Transcription: {msg}")
   ```

### Alternative: Using Custom Audio Input

To feed audio from TouchDesigner instead of the microphone:

1. **Export audio from TouchDesigner** via OSC or UDP
2. **Modify the audio callback** in `stt_service_2.py` to accept external audio:
   - Replace `sd.InputStream` with a UDP/OSC listener
   - Buffer incoming audio samples
   - Feed to both STT and YAMNet services

Example modification (for UDP audio input):
```python
# Replace the InputStream section with:
import socket

audio_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
audio_sock.bind(("127.0.0.1", 7200))  # Receive audio on port 7200

while True:
    data, addr = audio_sock.recvfrom(8192)
    audio_array = np.frombuffer(data, dtype=np.float32)
    
    # Share with YAMNet
    yamnet_detector.process_audio(audio_array)
    
    # Feed to STT
    audio_bytes = (audio_array * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
    recognizer.AcceptWaveform(audio_bytes)
    # ... rest of processing
```

## YAMNet Sound Classes

YAMNet can detect 521 audio event classes including:
- **Speech**: Speech, Conversation, Laughter
- **Music**: Music, Musical instrument, Singing
- **Animals**: Dog, Cat, Bird, Roar
- **Household**: Door, Knock, Alarm clock, Telephone
- **Transportation**: Car, Train, Airplane
- **Nature**: Water, Rain, Thunder, Wind

For the full class list, see: [AudioSet Ontology](https://research.google.com/audioset/ontology/index.html)

## Performance Tuning

### Adjust YAMNet Confidence Threshold

In `stt_service_2.py`, modify:
```python
yamnet_detector = YAMNetDetector(sample_rate=SAMPLE_RATE, confidence_threshold=0.3)
```

- Lower threshold (0.1-0.2): More detections, potentially more false positives
- Higher threshold (0.4-0.6): Fewer, more confident detections

### Optimize for Real-time Performance

1. **Reduce BLOCK_SIZE** in `config.csv` for lower latency (increases CPU usage)
2. **Increase BLOCK_SIZE** for better performance (increases latency)
3. **Adjust YAMNet queue size** in `yamnet_detector.py`:
   ```python
   self.audio_queue = queue.Queue(maxsize=50)  # Reduce for lower memory
   ```

## Threading Model

- **Main Thread**: Audio callback + STT processing (real-time priority)
- **YAMNet Thread**: Background sound detection (daemon thread)
- **Audio Sharing**: Non-blocking queue with drop-on-full policy

This design ensures STT performance is never impacted by YAMNet processing.

## Troubleshooting

### YAMNet model fails to load

**Error**: `Cannot load YAMNet model from TensorFlow Hub`

**Solutions**:
- Check internet connection (first run downloads model)
- Ensure TensorFlow is installed: `pip install tensorflow==2.15.0`
- Clear TensorFlow Hub cache: `rm -rf ~/tmp/tfhub_modules`

### High CPU usage

**Solutions**:
- Increase `BLOCK_SIZE` in `config.csv` (e.g., 8000)
- Raise YAMNet confidence threshold to reduce processing
- Use a smaller Vosk model

### No audio detected

**Solutions**:
- Check microphone permissions
- List available devices: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`
- Set default device in code or via system settings

### UDP messages not received in TouchDesigner

**Solutions**:
- Verify firewall settings allow UDP on ports 7201-7204
- Check TouchDesigner UDP In CHOP network settings
- Test with `nc -ul 7201` (netcat) to verify messages are being sent

## Development

### File Structure

```
speech_to_text_5_10_25/
├── config.csv              # Configuration settings
├── requirements.txt        # Python dependencies
├── service.sh              # Launch script
├── stt_service_2.py        # Main service (STT + YAMNet integration)
├── yamnet_detector.py      # YAMNet sound detection module
├── models/                 # Vosk models directory
│   └── vosk-model-small-pt-0.3/
└── td/                     # TouchDesigner project files
    └── stt.toe
```

### Adding New Features

To add new sound event handling:

1. Edit `yamnet_detector.py`
2. Modify the detection loop to filter/process specific events
3. Add custom UDP ports or message formats as needed

## License

This project uses:
- **Vosk**: Apache 2.0 License
- **YAMNet**: Apache 2.0 License
- **TensorFlow**: Apache 2.0 License

## References

- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
- [YAMNet Audio Event Detection](https://tfhub.dev/google/yamnet/1)
- [TouchDesigner](https://derivative.ca/)
- [AudioSet](https://research.google.com/audioset/)

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review TouchDesigner UDP In CHOP documentation
3. Open an issue on GitHub
