# filepath: src/stt_vosk.py
"""
Simple STT Vosk module:
- Reads configs from configs/configvosk.csv using config_manager
- Writes plain text lines to the configured text pipe (creates if missing)
- Two modes:
    - mock=true -> send periodic test messages (good for verifying TD <-> Python text channel)
    - mock=false -> attempt to use vosk for real transcription from an audio FIFO (basic example)
"""
import os
import time
import sys
import errno
from datetime import datetime
from src.config_manager import read_simple_kv_csv

DEFAULT_CONFIG = "configs/configvosk.csv"

def ensure_fifo(path):
    if not os.path.exists(path):
        try:
            os.mkfifo(path, 0o600)
            print(f"Created FIFO at {path}")
        except FileExistsError:
            pass
    # Note: opening FIFO for writing will block until a reader opens it.
    return path

def open_text_pipe_for_write(path, block=False):
    flags = os.O_WRONLY
    if not block:
        flags |= os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
        return os.fdopen(fd, 'wb', buffering=0)
    except OSError as e:
        if e.errno == errno.ENXIO or e.errno == errno.EWOULDBLOCK:
            # No reader yet
            return None
        raise

def write_text_line(pipe_file, text):
    try:
        if isinstance(text, str):
            text = text.encode('utf-8')
        pipe_file.write(text + b'\\n')
        pipe_file.flush()
    except BrokenPipeError:
        raise

def mock_loop(text_pipe_path):
    ensure_fifo(text_pipe_path)
    print("Mock mode: waiting for TD to open the text pipe for reading...")
    pipe = None
    while pipe is None:
        pipe = open_text_pipe_for_write(text_pipe_path, block=False)
        if pipe is None:
            time.sleep(0.5)
    print("Text pipe opened for write. Sending mock messages.")
    try:
        i = 0
        while True:
            ts = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
            msg = f"MOCK {i} {ts} Hello from stt_vosk"
            write_text_line(pipe, msg)
            i += 1
            time.sleep(1.0)
    except (BrokenPipeError, OSError):
        print("Reader closed the pipe; exiting mock loop.")
    finally:
        try:
            pipe.close()
        except Exception:
            pass

def run_real_vosk(cfg):
    # Minimal placeholder: try to import vosk and run a basic loop reading raw PCM from audio FIFO
    try:
        from vosk import Model, KaldiRecognizer
        import json
        import numpy as np
    except Exception as e:
        print("Failed to import vosk or numpy. Make sure vosk and numpy are installed.")
        print(e)
        sys.exit(1)

    model_path = cfg.get('model_path')
    audio_pipe = cfg.get('audio_pipe_path')
    text_pipe = cfg.get('text_pipe_path')

    if not model_path or not os.path.isdir(model_path):
        print("Vosk model path not found:", model_path)
        sys.exit(1)

    sample_rate = int(cfg.get('sample_rate', '16000'))

    # create Vosk objects
    model = Model(model_path)
    rec = KaldiRecognizer(model, sample_rate)

    # ensure FIFOs exist
    ensure_fifo(audio_pipe)
    ensure_fifo(text_pipe)

    # Open the text pipe for writing (block until reader ready)
    print("Waiting for TD to open text pipe for reading...")
    text_file = None
    while text_file is None:
        text_file = open_text_pipe_for_write(text_pipe, block=True)
        if text_file is None:
            time.sleep(0.1)
    print("Opened text pipe.")

    # Open audio FIFO for reading (blocking)
    with open(audio_pipe, 'rb') as a:
        print("Opened audio FIFO for reading. Feeding Vosk recognizer.")
        try:
            while True:
                # read some bytes (e.g., 0.1s of 16-bit PCM mono)
                chunk_size = int(sample_rate * 2 * 0.1)  # sample_rate * bytes_per_sample * seconds
                data = a.read(chunk_size)
                if not data:
                    time.sleep(0.01)
                    continue
                if rec.AcceptWaveform(data):
                    res = rec.Result()
                    # extract text field simply
                    try:
                        j = json.loads(res)
                        text = j.get('text', '')
                    except Exception:
                        text = res
                    if text:
                        write_text_line(text_file, text)
                else:
                    # partial result
                    pr = rec.PartialResult()
                    # optional: send partial results with a prefix or to metadata channel
        except KeyboardInterrupt:
            print("Interrupted, exiting.")
        finally:
            try:
                text_file.close()
            except Exception:
                pass

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = read_simple_kv_csv(cfg_path)
    mock = cfg.get('mock', 'true').lower() in ('1', 'true', 'yes')
    text_pipe = cfg.get('text_pipe_path', '/tmp/stt_text.pipe')

    print("Loaded config:", cfg)
    print("Text pipe:", text_pipe)
    if mock:
        mock_loop(text_pipe)
    else:
        run_real_vosk(cfg)

if __name__ == '__main__':
    main()
