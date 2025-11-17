# Multilingual Dynamic STT System - User Guide

This guide explains how to use the new multilingual, dynamic speech-to-text system with automatic language detection and real-time model switching.

## Overview

The system now supports:
- **Multiple Vosk models** preloaded for different languages
- **Automatic language detection** using Whisper
- **Real-time model switching** based on detected language
- **Intelligent fallback** to Whisper for unsupported languages
- **Configurable thresholds** and detection intervals

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Audio Input (Microphone)                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         STT Orchestrator                        │
│  • Buffers audio (3s by default)                │
│  • Triggers language detection (every 10s)      │
│  • Routes audio to appropriate service          │
└──────────┬──────────────────────────┬───────────┘
           │                          │
           ▼                          ▼
┌──────────────────────┐   ┌─────────────────────┐
│  Whisper Service     │   │   Vosk Service      │
│  • Language detect   │   │   • Multiple models │
│  • Transcription     │   │   • Fast switching  │
│  • All languages     │   │   • Pre-loaded      │
└──────────────────────┘   └─────────────────────┘
```

## Configuration Files

### 1. vosk_models.csv

Defines available Vosk models for different languages:

```csv
Key,Value
en,models/vosk-model-small-en-us-0.15
es,models/vosk-model-small-es-0.42
fr,models/vosk-model-small-fr-0.22
de,models/vosk-model-small-de-0.15
pt,models/vosk-model-small-pt-0.3
```

**Important**: Only include models that are actually downloaded and present in your `models/` directory.

### 2. orchestrator_config.csv

Controls language detection and switching:

```csv
Key,Value
SAMPLE_RATE,16000
BLOCK_SIZE,4000
VOSK_ENABLED,true
WHISPER_ENABLED,true
YAMNET_ENABLED,true
DYNAMIC_SWITCHING_ENABLED,true
LANGUAGE_DETECT_BUFFER_DURATION,3.0
LANGUAGE_CONFIDENCE_THRESHOLD,0.5
LANGUAGE_DETECT_INTERVAL,10.0
```

**Configuration Options:**

- `DYNAMIC_SWITCHING_ENABLED`: Enable/disable automatic language switching
- `LANGUAGE_DETECT_BUFFER_DURATION`: How many seconds of audio to buffer for detection (default: 3.0)
- `LANGUAGE_CONFIDENCE_THRESHOLD`: Minimum confidence to trigger a language switch (0.0-1.0, default: 0.5)
- `LANGUAGE_DETECT_INTERVAL`: How often to perform language detection in seconds (default: 10.0)

## Setup Instructions

### Step 1: Download Vosk Models

Download models for the languages you want to support from https://alphacephei.com/vosk/models

Example:
```bash
cd models/

# English
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip

# Spanish
wget https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip vosk-model-small-es-0.42.zip

# French
wget https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip
unzip vosk-model-small-fr-0.22.zip
```

### Step 2: Configure vosk_models.csv

Edit `vosk_models.csv` to match the models you downloaded:

```csv
Key,Value
en,models/vosk-model-small-en-us-0.15
es,models/vosk-model-small-es-0.42
fr,models/vosk-model-small-fr-0.22
```

**Note**: The language code (en, es, fr, etc.) must match Whisper's language detection output.

### Step 3: Enable Services

Ensure both Vosk and Whisper are enabled in `orchestrator_config.csv`:

```csv
VOSK_ENABLED,true
WHISPER_ENABLED,true
DYNAMIC_SWITCHING_ENABLED,true
```

### Step 4: Install Whisper (if not already installed)

```bash
source stt_venv/bin/activate
pip install openai-whisper
```

### Step 5: Start the System

```bash
./service.sh
```

## How It Works

### Language Detection Flow

1. **Audio Buffering**: The orchestrator continuously buffers incoming audio
2. **Periodic Detection**: Every N seconds (configurable), the system:
   - Sends buffered audio to Whisper for language detection
   - Whisper analyzes the audio and returns a language code + confidence
3. **Decision Making**:
   - If confidence >= threshold AND language changed:
     - Check if Vosk has a model for this language
     - If yes: Switch to Vosk for that language
     - If no: Use Whisper for transcription
4. **Audio Routing**: Subsequent audio is routed to the selected service

### Example Scenario

```
[00:00] System starts → Default to Spanish Vosk model
[00:03] First detection → English detected (confidence: 0.92)
[00:03] Switch to English Vosk model ✓
[00:13] Second detection → English (no change)
[00:23] Third detection → French detected (confidence: 0.88)
[00:23] Check Vosk models → French model available
[00:23] Switch to French Vosk model ✓
[00:33] Fourth detection → Japanese detected (confidence: 0.85)
[00:33] Check Vosk models → No Japanese model
[00:33] Switch to Whisper for Japanese ✓
```

## Tuning Parameters

### Detection Frequency

Trade-off: **Responsiveness vs CPU Usage**

```csv
LANGUAGE_DETECT_INTERVAL,5.0   # More responsive, higher CPU
LANGUAGE_DETECT_INTERVAL,15.0  # Less responsive, lower CPU
```

**Recommended**: 10-15 seconds for most use cases

### Confidence Threshold

Trade-off: **Accuracy vs Switching Speed**

```csv
LANGUAGE_CONFIDENCE_THRESHOLD,0.3   # Switch quickly, may have false positives
LANGUAGE_CONFIDENCE_THRESHOLD,0.7   # More conservative, fewer switches
```

**Recommended**: 0.5-0.6 for balanced performance

### Buffer Duration

Trade-off: **Detection Accuracy vs Latency**

```csv
LANGUAGE_DETECT_BUFFER_DURATION,2.0   # Faster, less accurate
LANGUAGE_DETECT_BUFFER_DURATION,5.0   # Slower, more accurate
```

**Recommended**: 3.0 seconds for good balance

## Monitoring and Logs

### Language Detection Events

The system logs all language detection and switching events:

```
[Whisper Service] 🌍 Language detected: en (confidence: 0.92)
[Orchestrator] 🌍 Language detected: en (confidence: 0.92)
[Vosk Service] 🔄 Switched language: es → en
[Orchestrator] ✅ Switched to Vosk (en)
```

### Service Status

At startup, the system displays:
- Available Vosk language models
- Language detection status
- Current switching configuration

```
📡 Active services: vosk, whisper, yamnet
   • Vosk STT: ports 7201, 7202, 7203
     Languages: en, es, fr, de, pt
   • Whisper STT: ports 7211, 7212
     Language detection: enabled
   • YAMNet: port 7204

🔄 Dynamic language switching: ENABLED
   • Buffer duration: 3.0s
   • Confidence threshold: 0.5
   • Detection interval: 10.0s
   • Current STT service: vosk
```

## UDP Output

### Vosk (when active)
- Port 7201: Partial transcription
- Port 7202: Final transcription
- Port 7203: Word-level confidence

### Whisper (when active)
- Port 7211: Partial transcription (not used currently)
- Port 7212: Final transcription

The output ports remain the same regardless of which service is active for seamless TouchDesigner integration.

## Troubleshooting

### Problem: "No models loaded successfully"

**Solution**: 
1. Check that model directories exist: `ls -la models/`
2. Verify paths in `vosk_models.csv` match actual directories
3. Ensure model directories contain required files (am/, graph/, etc.)

### Problem: Language keeps switching back and forth

**Solution**:
- Increase `LANGUAGE_CONFIDENCE_THRESHOLD` (e.g., 0.7)
- Increase `LANGUAGE_DETECT_INTERVAL` (e.g., 15.0)
- Increase `LANGUAGE_DETECT_BUFFER_DURATION` (e.g., 4.0)

### Problem: Language detection too slow

**Solution**:
- Decrease `LANGUAGE_DETECT_INTERVAL` (e.g., 5.0)
- Use a smaller Whisper model in `whisper_config.csv` (e.g., "tiny")

### Problem: High memory usage

**Cause**: Loading many Vosk models consumes RAM

**Solutions**:
1. Only load models you actively use
2. Remove unused languages from `vosk_models.csv`
3. Use smaller Vosk models (check model size before downloading)

### Problem: Language not detected correctly

**Solutions**:
1. Ensure audio quality is good (no excessive noise)
2. Increase buffer duration for more context
3. Check that speaker is speaking clearly
4. Verify microphone is working properly

## Performance Optimization

### Memory Usage

Each Vosk model uses ~50-200 MB RAM depending on size. For example:
- Small models: ~50-100 MB each
- Medium models: ~100-200 MB each

To minimize memory:
```csv
# vosk_models.csv - Only load what you need
en,models/vosk-model-small-en-us-0.15
es,models/vosk-model-small-es-0.42
```

### CPU Usage

Language detection using Whisper is CPU-intensive. To optimize:
1. Use smaller Whisper model (tiny or base)
2. Increase detection interval
3. Disable switching if not needed:
   ```csv
   DYNAMIC_SWITCHING_ENABLED,false
   ```

### Real-time Performance

For minimal latency:
```csv
BLOCK_SIZE,2000                    # Smaller blocks
LANGUAGE_DETECT_INTERVAL,20.0      # Less frequent detection
LANGUAGE_DETECT_BUFFER_DURATION,2.0 # Shorter buffer
```

## Advanced Usage

### Disabling Dynamic Switching

To use the system with manual language selection:

1. Set in `orchestrator_config.csv`:
   ```csv
   DYNAMIC_SWITCHING_ENABLED,false
   ```

2. Default language is determined by:
   - First model in `vosk_models.csv` (if Vosk is active)
   - Whisper with configured language (if Whisper is active)

### Using Only Whisper

To use Whisper for all languages without Vosk:

```csv
VOSK_ENABLED,false
WHISPER_ENABLED,true
DYNAMIC_SWITCHING_ENABLED,false
```

Set desired language in `whisper_config.csv`:
```csv
LANGUAGE,en
```

### Using Only Vosk (Single Language)

For fast, single-language transcription:

```csv
VOSK_ENABLED,true
WHISPER_ENABLED,false
DYNAMIC_SWITCHING_ENABLED,false
```

Configure model in `vosk_config.csv`:
```csv
MODEL_PATH,models/vosk-model-small-en-us-0.15
```

## Integration with TouchDesigner

The dynamic switching is transparent to TouchDesigner. UDP ports remain consistent:

1. **Create UDP In CHOPs** for active service ports
2. **Parse messages** as usual
3. **No changes needed** - the system handles switching internally

The only visible change is language switching logged in console (optional monitoring).

## Best Practices

1. **Start Small**: Begin with 2-3 languages, expand as needed
2. **Test Thoroughly**: Verify each language model works individually first
3. **Monitor Performance**: Watch CPU/memory usage with your specific configuration
4. **Tune Gradually**: Adjust one parameter at a time to find optimal settings
5. **Use Appropriate Models**: Match model size to your hardware capabilities

## Language Support

### Vosk Supported Languages

Check https://alphacephei.com/vosk/models for available models:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Portuguese (pt)
- Italian (it)
- Russian (ru)
- Chinese (zh)
- Many more...

### Whisper Supported Languages

Whisper supports 99 languages including:
- All Vosk languages above
- Plus: Arabic, Hindi, Japanese, Korean, Vietnamese, Turkish, and many more

**Strategy**: Use Vosk for your most common languages (fast), Whisper as fallback (comprehensive).

## Example Configurations

### Configuration 1: Multilingual Office (3 languages)

```csv
# vosk_models.csv
en,models/vosk-model-small-en-us-0.15
es,models/vosk-model-small-es-0.42
fr,models/vosk-model-small-fr-0.22

# orchestrator_config.csv
DYNAMIC_SWITCHING_ENABLED,true
LANGUAGE_DETECT_INTERVAL,10.0
LANGUAGE_CONFIDENCE_THRESHOLD,0.6
LANGUAGE_DETECT_BUFFER_DURATION,3.0
```

### Configuration 2: Presentation Mode (fast, minimal switching)

```csv
# vosk_models.csv
en,models/vosk-model-small-en-us-0.15

# orchestrator_config.csv
DYNAMIC_SWITCHING_ENABLED,false
VOSK_ENABLED,true
WHISPER_ENABLED,false
```

### Configuration 3: Global Support (many languages)

```csv
# vosk_models.csv
en,models/vosk-model-small-en-us-0.15
es,models/vosk-model-small-es-0.42
fr,models/vosk-model-small-fr-0.22
de,models/vosk-model-small-de-0.15
zh,models/vosk-model-small-cn-0.22

# orchestrator_config.csv
DYNAMIC_SWITCHING_ENABLED,true
WHISPER_ENABLED,true
LANGUAGE_DETECT_INTERVAL,8.0
LANGUAGE_CONFIDENCE_THRESHOLD,0.5
```

## Summary

The multilingual dynamic STT system provides:
- ✅ Automatic language detection
- ✅ Real-time model switching
- ✅ Intelligent fallback mechanism
- ✅ Configurable behavior
- ✅ Seamless TouchDesigner integration
- ✅ Minimal code changes required

For questions or issues, refer to the main README or open a GitHub issue.
