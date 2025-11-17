#!/usr/bin/env python3
"""
Whisper Service Module
Handles Whisper STT detection and language detection in its own thread.
"""

import csv
import threading
import queue
import numpy as np
import time
from typing import Optional, Tuple, Callable

import whisper  # <-- Uncommented

class WhisperService:
    """
    Whisper-based speech-to-text service.
    Processes audio in a separate thread and sends results via UDP.
    Provides language detection functionality for orchestrator.
    """
    
    def __init__(self, config_path: str, udp_handler, sample_rate: int = 16000):
        """
        Initialize the Whisper service.
        
        Args:
            config_path: Path to Whisper configuration CSV file
            udp_handler: UDPHandler instance for sending messages
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.udp_handler = udp_handler
        self.running = False
        self.audio_queue = queue.Queue(maxsize=50)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Whisper model (lazy loaded)
        self.model = None
        
        # Thread
        self.thread: Optional[threading.Thread] = None
        
        # Audio buffer for accumulation
        self.audio_buffer = []
        self.buffer_duration = 3.0  # Process every 3 seconds of audio
        
        # Language detection callback
        self.language_detection_callback: Optional[Callable[[str, float], None]] = None
        
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
            print(f"[Whisper Service] Configuration loaded from {config_path}")
        except Exception as e:
            print(f"[Whisper Service] Error loading config: {e}")
        return config
    
    def _initialize_model(self):
        """Initialize the Whisper model."""
        try:
            model_size = self.config.get("MODEL_SIZE", "base")
            print(f"[Whisper Service] Loading {model_size} model...")
            self.model = whisper.load_model(model_size)
            print("[Whisper Service] ✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"[Whisper Service] ❌ Error loading model: {e}")
            return False
    
    def set_language_detection_callback(self, callback: Callable[[str, float], None]):
        """
        Set callback for language detection events.
        
        Args:
            callback: Function to call with (language_code, confidence) when language is detected
        """
        self.language_detection_callback = callback
    
    def detect_language(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Detect language from audio buffer.
        
        Args:
            audio_data: numpy array of audio samples (float32, -1.0 to 1.0)
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if self.model is None:
            print("[Whisper Service] Model not loaded, cannot detect language")
            return ("en", 0.0)
        
        try:
            # Ensure audio is 1D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Whisper expects audio padded/trimmed to 30 seconds for language detection
            # We'll use what we have
            audio_data = whisper.pad_or_trim(audio_data)
            
            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_data).to(self.model.device)
            
            # Detect language
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            confidence = probs[detected_language]
            
            print(f"[Whisper Service] 🌍 Language detected: {detected_language} (confidence: {confidence:.2f})")
            return (detected_language, confidence)
            
        except Exception as e:
            print(f"[Whisper Service] Error detecting language: {e}")
            return ("en", 0.0)
    
    def process_audio(self, audio_data: np.ndarray):
        """
        Add audio data to the processing queue.
        
        Args:
            audio_data: numpy array of audio samples (float32, -1.0 to 1.0)
        """
        if not self.running:
            return
        try:
            self.audio_queue.put_nowait(audio_data.copy())
        except queue.Full:
            pass
    
    def _recognition_loop(self):
        print("[Whisper Service] Recognition thread started")
        port_partial = self.config.get("UDP_PORT_PARTIAL", 7211)
        port_final = self.config.get("UDP_PORT_FINAL", 7212)
        noise_threshold = self.config.get("NOISE_THRESHOLD", 0.01)
        language = self.config.get("LANGUAGE", None)
        if not language or language.lower() == "none":
            language = None
        samples_per_buffer = int(self.buffer_duration * self.sample_rate)
        
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
                self.audio_buffer.extend(audio_data)
                if len(self.audio_buffer) >= samples_per_buffer:
                    buffer_array = np.array(self.audio_buffer[:samples_per_buffer], dtype=np.float32)
                    self.audio_buffer = self.audio_buffer[samples_per_buffer:]
                    energy = np.abs(buffer_array).mean()
                    if energy < noise_threshold:
                        continue

                    # Whisper transcription
                    if self.model is None:
                        print("[Whisper Service] Model not loaded, cannot transcribe.")
                        continue
                    
                    # Set language to None if empty or "None" for auto-detection
                    language = self.config.get("LANGUAGE", None)
                    if not language or str(language).lower() == "none":
                        language = None
                    
                    # Transcribe with language detection if language not specified
                    result = self.model.transcribe(buffer_array, language=language, fp16=False)
                    
                    # Call language detection callback if available
                    if self.language_detection_callback and "language" in result:
                        detected_lang = result["language"]
                        # Estimate confidence from transcription
                        confidence = 0.8  # Default confidence
                        self.language_detection_callback(detected_lang, confidence)
                    
                    text = result.get("text", "")
                    if isinstance(text, str):
                        text = text.strip()
                    if text:
                        self.udp_handler.send_message(text, port_final)
                        detected_lang = result.get("language", "unknown")
                        print(f"[Whisper Service] Transcribed ({detected_lang}): {text}")
                    else:
                        print("[Whisper Service] No transcribed text received.")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Whisper Service] Recognition error: {e}")
        print("[Whisper Service] Recognition thread stopped")
    
    def start(self) -> bool:
        if self.running:
            print("[Whisper Service] Already running")
            return False
        if not self._initialize_model():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._recognition_loop, daemon=False)
        self.thread.start()
        print("[Whisper Service] Service started")
        return True
    
    def stop(self):
        print("[Whisper Service] Stopping...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("[Whisper Service] Warning: Thread did not stop within timeout")
        print("[Whisper Service] Service stopped")
