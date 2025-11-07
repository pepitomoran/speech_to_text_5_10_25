# Speech-to-Text with Integrated Sound Detection - Modular Architecture

This project implements a modular real-time audio system supporting multiple speech-to-text engines (Vosk, Whisper) and sound event detection (YAMNet). Each service runs independently in its own thread and communicates via UDP.

## Features

- **Modular Architecture**: Each detector runs as an independent service
- **Multiple STT Engines**: 
  - Vosk for fast, accurate speech recognition
  - Whisper for high-quality transcription (placeholder implementation)
- **Sound Event Detection**: YAMNet-based detection of 521 sound classes
- **Flexible Configuration**: Separate CSV config files for each service
- **Dynamic Service Management**: Start/stop services independently via config or CLI
- **UDP Communication**: All services send results via UDP for TouchDesigner integration
- **Thread-based Architecture**: Each service runs in its own thread with independent lifecycle

## Architecture

```
┌─────────────────────────────────────────────────┐
│         STT Service Orchestrator                │
│  (stt_service.py)                               │
│  • Loads configurations                         │
│  • Manages service lifecycle                    │
│  • Routes audio to active services              │
└──────────────┬──────────────────────────────────┘
               │
      ┌────────┴────────┐
      │  UDP Handler    │
      │ (udp_handler.py)│
      └────────┬────────┘
               │
    ───────────┴───────────────────
    │          │          │         │
    ▼          ▼          ▼         ▼
┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────────┐
│  Vosk   │ │ Whisper │ │ YAMNet │ │   Audio    │
│ Service │ │ Service │ │Service │ │   Input    │
│(thread) │ │(thread) │ │(thread)│ │ (callback) │
└────┬────┘ └────┬────┘ └───┬────┘ └─────┬──────┘
     │           │           │            │
     ▼           ▼           ▼            ▼
  UDP 7201-   UDP 7211-   UDP 7204    Audio Stream
   7203         7212                  (Microphone)
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
   # Extract to models/vosk-model-small-es-0.42 (or update vosk_config.csv)
   ```

## Configuration

The system uses separate CSV configuration files for each component:

### Orchestrator Configuration (`orchestrator_config.csv`)

Controls which services are enabled and global audio settings:

```csv
Key,Value
SAMPLE_RATE,16000
BLOCK_SIZE,4000
VOSK_ENABLED,true
WHISPER_ENABLED,false
YAMNET_ENABLED,true
```

- **SAMPLE_RATE**: Audio sample rate (16000 Hz recommended)
- **BLOCK_SIZE**: Audio buffer size (affects latency)
- **VOSK_ENABLED**: Enable/disable Vosk STT service
- **WHISPER_ENABLED**: Enable/disable Whisper STT service
- **YAMNET_ENABLED**: Enable/disable YAMNet sound detection

### Vosk Configuration (`vosk_config.csv`)

```csv
Key,Value
MODEL_PATH,models/vosk-model-small-es-0.42
MAX_WORDS,16
UDP_PORT_PARTIAL,7201
UDP_PORT_FINAL,7202
UDP_PORT_WORD_CONF,7203
```

- **MODEL_PATH**: Path to Vosk model directory
- **MAX_WORDS**: Maximum words per partial transcription chunk
- **UDP_PORT_PARTIAL**: Port for partial (in-progress) transcription
- **UDP_PORT_FINAL**: Port for final transcription results
- **UDP_PORT_WORD_CONF**: Port for word-level confidence data

### Whisper Configuration (`whisper_config.csv`)

```csv
Key,Value
MODEL_SIZE,base
LANGUAGE,es
UDP_PORT_PARTIAL,7211
UDP_PORT_FINAL,7212
NOISE_THRESHOLD,0.01
```

- **MODEL_SIZE**: Whisper model size (tiny, base, small, medium, large)
- **LANGUAGE**: Language code for transcription
- **UDP_PORT_PARTIAL**: Port for partial transcription
- **UDP_PORT_FINAL**: Port for final transcription
- **NOISE_THRESHOLD**: Minimum audio energy to process

**Note**: Whisper is currently a placeholder implementation. To fully enable:
1. Install: `pip install openai-whisper`
2. Uncomment implementation in `whisper_service.py`
3. Set `WHISPER_ENABLED=true` in `orchestrator_config.csv`

### YAMNet Configuration (`yamnet_config.csv`)

```csv
Key,Value
CONFIDENCE_THRESHOLD,0.3
UDP_PORT,7204
QUEUE_SIZE,100
```

- **CONFIDENCE_THRESHOLD**: Minimum confidence for detection reporting
- **UDP_PORT**: Port for sound event detection results
- **QUEUE_SIZE**: Audio processing queue size

## Running the Services

### Start the System

```bash
./service.sh
```

This launches the orchestrator which starts all enabled services.

### CLI Arguments

You can override configuration settings via command-line arguments:

```bash
# Enable/disable services dynamically
python3 stt_service.py --base_dir . --enable-whisper --disable-yamnet

# Available options:
#   --enable-vosk / --disable-vosk
#   --enable-whisper / --disable-whisper
#   --enable-yamnet / --disable-yamnet
```

### Expected Output

```
============================================================
STT Service Orchestrator - Modular Audio System
============================================================

[Orchestrator] Configuration loaded from orchestrator_config.csv
[Orchestrator] Initializing services...
[Vosk Service] Configuration loaded from vosk_config.csv
[Vosk Service] Loading model from models/vosk-model-small-es-0.42...
[Vosk Service] ✅ Model loaded successfully
[Orchestrator] ✅ Vosk service initialized
[YAMNet Service] Configuration loaded from yamnet_config.csv
[YAMNet Service] Loading model from TensorFlow Hub...
[YAMNet Service] ✅ Model loaded successfully
[Orchestrator] ✅ YAMNet service initialized

[Orchestrator] Starting audio stream...

============================================================
🎤 Listening... (Press Ctrl+C to stop)
============================================================

📡 Active services: vosk, yamnet
   • Vosk STT: ports 7201, 7202, 7203
   • YAMNet: port 7204
```

### Stop Services

Press `Ctrl+C` to gracefully stop all services.

## UDP Output Ports

### Vosk STT Service

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

### Whisper STT Service

- **Port 7211**: Partial transcription
- **Port 7212**: Final transcription

### YAMNet Sound Detection Service

- **Port 7204**: Detected sound events

#### Message Format
```json
{
  "event": "Speech",
  "class_id": 0,
  "confidence": 0.87,
  "timestamp": 1699281234.567
}
```

## Module Documentation

### Core Modules

#### `stt_service.py` - Orchestrator
Main entry point that:
- Loads all configuration files
- Initializes and manages service lifecycle
- Routes audio from microphone to active services
- Handles graceful shutdown

#### `udp_handler.py` - UDP Communication
Centralized UDP handler that:
- Manages UDP sockets for all services
- Provides unified send/receive interface
- Supports both text and JSON messages
- Handles listener threads for incoming messages

#### `vosk_service.py` - Vosk STT Service
Vosk speech recognition service:
- Runs in independent thread
- Processes audio with Vosk model
- Sends partial and final transcriptions
- Provides word-level confidence scores

#### `whisper_service.py` - Whisper STT Service
Whisper speech recognition service (placeholder):
- Template for Whisper integration
- Includes audio buffering logic
- Noise threshold filtering
- Ready for full implementation

#### `yamnet_service.py` - YAMNet Sound Detection
YAMNet sound event detection service:
- Detects 521 audio event classes
- Runs TensorFlow inference in separate thread
- Configurable confidence threshold
- Downloads AudioSet class names automatically

## Service Management

### Dynamic Service Control

Services can be controlled via:

1. **Configuration Files**: Edit `orchestrator_config.csv` and restart
2. **CLI Arguments**: Override config at startup
3. **Programmatic Control**: Call `start_service()` / `stop_service()` methods

Example programmatic usage:

```python
from stt_service import STTOrchestrator

orchestrator = STTOrchestrator("/path/to/base/dir")

# Start specific services
orchestrator.start_service("vosk")
orchestrator.start_service("yamnet")

# Stop a service
orchestrator.stop_service("yamnet")
```

## TouchDesigner Integration

### Setup in TouchDesigner

1. **Create UDP In CHOPs** for each active service port:
   - Vosk: ports 7201, 7202, 7203
   - Whisper: ports 7211, 7212
   - YAMNet: port 7204

2. **Configure UDP In CHOPs**:
   - Network Address: `127.0.0.1`
   - Network Port: Set to corresponding port
   - Protocol: UDP

3. **Parse JSON data** using a `DAT Execute`:
   ```python
   import json
   
   def onReceive(dat, rowIndex, cols, cells):
       msg = cells[0].val
       try:
           data = json.loads(msg)
           # Process data based on structure
           if 'word' in data:
               print(f"Word: {data['word']}, Confidence: {data['confidence']}")
           elif 'event' in data:
               print(f"Sound: {data['event']}, Confidence: {data['confidence']}")
       except:
           # Handle plain text (ports 7201, 7202, 7211, 7212)
           print(f"Transcription: {msg}")
   ```

## Performance Tuning

### Audio Latency

Adjust `BLOCK_SIZE` in `orchestrator_config.csv`:
- Smaller (2000-3000): Lower latency, higher CPU usage
- Larger (6000-8000): Higher latency, lower CPU usage

### YAMNet Detection Sensitivity

Adjust `CONFIDENCE_THRESHOLD` in `yamnet_config.csv`:
- Lower (0.1-0.2): More detections, potential false positives
- Higher (0.4-0.6): Fewer, more confident detections

### Queue Sizes

Adjust `QUEUE_SIZE` in service configs to balance memory usage vs. handling bursty audio.

## File Structure

```
speech_to_text_5_10_25/
├── orchestrator_config.csv    # Main orchestrator configuration
├── vosk_config.csv             # Vosk service configuration
├── whisper_config.csv          # Whisper service configuration
├── yamnet_config.csv           # YAMNet service configuration
├── config.csv                  # Legacy config (kept for reference)
├── requirements.txt            # Python dependencies
├── service.sh                  # Launch script
├── stt_service.py              # Main orchestrator
├── udp_handler.py              # Centralized UDP communication
├── vosk_service.py             # Vosk STT service module
├── whisper_service.py          # Whisper STT service module
├── yamnet_service.py           # YAMNet detection service module
├── stt_service_2.py            # Legacy monolithic service
├── yamnet_detector.py          # Legacy YAMNet module
├── models/                     # Vosk models directory
│   └── vosk-model-small-es-0.42/
└── td/                         # TouchDesigner project files
    └── stt.toe
```

## Troubleshooting

### Service fails to start

**Check configuration files**: Ensure all CSV files are properly formatted and paths are correct.

```bash
# Verify configs exist
ls -la *_config.csv
```

### Model not found

**For Vosk**: Update `MODEL_PATH` in `vosk_config.csv` to point to your model directory.

### High CPU usage

**Solutions**:
- Increase `BLOCK_SIZE` in `orchestrator_config.csv`
- Disable unused services (set to `false` in config)
- Increase YAMNet `CONFIDENCE_THRESHOLD` to reduce processing

### UDP messages not received

**Solutions**:
- Verify firewall settings allow UDP on configured ports
- Test with netcat: `nc -ul 7201`
- Check service status in orchestrator output

## Development

### Adding a New Service

1. Create service module (e.g., `new_service.py`)
2. Implement service class with `start()`, `stop()`, and `process_audio()` methods
3. Create configuration CSV file
4. Add service initialization to orchestrator
5. Update README documentation

### Testing Individual Services

```python
from udp_handler import UDPHandler
from vosk_service import VoskService

# Test Vosk service standalone
udp = UDPHandler()
vosk = VoskService("vosk_config.csv", udp, base_dir=".")
vosk.start()

# Feed test audio...
# vosk.process_audio(audio_data)

vosk.stop()
udp.close_all()
```

## License

This project uses:
- **Vosk**: Apache 2.0 License
- **Whisper**: MIT License
- **YAMNet**: Apache 2.0 License
- **TensorFlow**: Apache 2.0 License

## References

- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [YAMNet Audio Event Detection](https://tfhub.dev/google/yamnet/1)
- [TouchDesigner](https://derivative.ca/)
- [AudioSet Ontology](https://research.google.com/audioset/)

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration files for correctness
3. Verify all dependencies are installed
4. Open an issue on GitHub
