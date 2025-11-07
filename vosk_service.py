#!/usr/bin/env python3
"""
Vosk Service Module
Handles Vosk STT detection in its own thread.
"""

import os
import csv
import json
import threading
import queue
import numpy as np
from typing import Optional
from vosk import Model, KaldiRecognizer


class VoskService:
    """
    Vosk-based speech-to-text service.
    Processes audio in a separate thread and sends results via UDP.
    """
    
    def __init__(self, config_path: str, udp_handler, sample_rate: int = 16000, base_dir: str = "."):
        """
        Initialize the Vosk service.
        
        Args:
            config_path: Path to Vosk configuration CSV file
            udp_handler: UDPHandler instance for sending messages
            sample_rate: Audio sample rate
            base_dir: Base directory for relative paths
        """
        self.sample_rate = sample_rate
        self.udp_handler = udp_handler
        self.base_dir = base_dir
        self.running = False
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Model and recognizer (lazy loaded)
        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None
        
        # Thread
        self.thread: Optional[threading.Thread] = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from CSV file."""
        config = {}
        try:
            with open(config_path, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        key, value = row[0], row[1]
                        # Convert numeric values
                        try:
                            if '.' in value:
                                config[key] = float(value)
                            elif value.isdigit():
                                config[key] = int(value)
                            else:
                                config[key] = value
                        except ValueError:
                            config[key] = value
            print(f"[Vosk Service] Configuration loaded from {config_path}")
        except Exception as e:
            print(f"[Vosk Service] Error loading config: {e}")
        return config
    
    def _initialize_model(self):
        """Initialize the Vosk model and recognizer."""
        try:
            model_path = self.config.get("MODEL_PATH", "models/vosk-model-small-en-us-0.15")
            full_model_path = os.path.join(self.base_dir, model_path)
            
            print(f"[Vosk Service] Loading model from {full_model_path}...")
            self.model = Model(full_model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            print("[Vosk Service] ✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"[Vosk Service] ❌ Error loading model: {e}")
            return False
    
    def process_audio(self, audio_data: np.ndarray):
        """
        Add audio data to the processing queue.
        
        Args:
            audio_data: numpy array of audio samples (float32, -1.0 to 1.0)
        """
        if not self.running:
            return
        
        try:
            # Non-blocking put - drop audio if queue is full
            self.audio_queue.put_nowait(audio_data.copy())
        except queue.Full:
            pass  # Skip if queue is full (maintain real-time performance)
    
    def _recognition_loop(self):
        """Main recognition loop running in a separate thread."""
        print("[Vosk Service] Recognition thread started")
        
        # Get UDP ports from config
        port_partial = self.config.get("UDP_PORT_PARTIAL", 7201)
        port_final = self.config.get("UDP_PORT_FINAL", 7202)
        port_word_conf = self.config.get("UDP_PORT_WORD_CONF", 7203)
        max_words = self.config.get("MAX_WORDS", 16)
        
        while self.running:
            try:
                # Get audio from queue with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Ensure audio is 1D and in the correct format
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
                
                # Convert float32 to int16 PCM bytes for Vosk
                audio_bytes = (audio_data * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
                
                # Feed to recognizer
                if self.recognizer.AcceptWaveform(audio_bytes):
                    # Finalized segment
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        self.udp_handler.send_message(text, port_final)
                    
                    # Send word + confidence for each word
                    for word in result.get("result", []):
                        word_conf = {
                            "word": word.get("word"),
                            "confidence": word.get("conf"),
                            "start": word.get("start"),
                            "end": word.get("end"),
                        }
                        self.udp_handler.send_json(word_conf, port_word_conf)
                else:
                    # Partial transcription (in progress)
                    partial = json.loads(self.recognizer.PartialResult())
                    ptext = partial.get("partial", "")
                    if ptext:
                        # Split into chunks based on max words
                        words = ptext.split()
                        for i in range(0, len(words), max_words):
                            chunk = " ".join(words[i:i + max_words])
                            self.udp_handler.send_message(chunk, port_partial)
                            
                            # Send each word with confidence None
                            for word in chunk.split():
                                word_conf = {
                                    "word": word,
                                    "confidence": None,
                                }
                                self.udp_handler.send_json(word_conf, port_word_conf)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Vosk Service] Recognition error: {e}")
        
        print("[Vosk Service] Recognition thread stopped")
    
    def start(self) -> bool:
        """Start the Vosk service in a separate thread."""
        if self.running:
            print("[Vosk Service] Already running")
            return False
        
        # Initialize model
        if not self._initialize_model():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._recognition_loop, daemon=False)
        self.thread.start()
        print("[Vosk Service] Service started")
        return True
    
    def stop(self):
        """Stop the Vosk service."""
        print("[Vosk Service] Stopping...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("[Vosk Service] Warning: Thread did not stop within timeout")
        print("[Vosk Service] Service stopped")
