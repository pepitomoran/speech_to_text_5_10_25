#!/usr/bin/env python3
"""
Vosk Service Module
Handles Vosk STT detection in its own thread.
Supports multiple preloaded models for different languages.
"""

import os
import csv
import json
import threading
import queue
import numpy as np
from typing import Optional, Dict
from vosk import Model, KaldiRecognizer


class VoskService:
    """
    Vosk-based speech-to-text service.
    Processes audio in a separate thread and sends results via UDP.
    Supports multiple preloaded models for different languages.
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
        
        # Multiple models and recognizers for different languages
        self.models: Dict[str, Model] = {}
        self.recognizers: Dict[str, KaldiRecognizer] = {}
        self.current_language: str = "es"  # Default language
        self.language_lock = threading.Lock()  # Thread-safe language switching
        
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
    
    def _load_models_config(self, models_config_path: str) -> Dict[str, str]:
        """Load models configuration from CSV file."""
        models_config = {}
        try:
            with open(models_config_path, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        lang_code, model_path = row[0], row[1]
                        models_config[lang_code] = model_path
            print(f"[Vosk Service] Models configuration loaded: {list(models_config.keys())}")
        except Exception as e:
            print(f"[Vosk Service] Error loading models config: {e}")
        return models_config
    
    def _initialize_model(self):
        """Initialize the Vosk model and recognizer (legacy single model)."""
        try:
            model_path = self.config.get("MODEL_PATH", "models/vosk-model-small-en-us-0.15")
            full_model_path = os.path.join(self.base_dir, model_path)
            
            print(f"[Vosk Service] Loading single model from {full_model_path}...")
            model = Model(full_model_path)
            recognizer = KaldiRecognizer(model, self.sample_rate)
            recognizer.SetWords(True)
            
            # Store as default language model
            self.models[self.current_language] = model
            self.recognizers[self.current_language] = recognizer
            print(f"[Vosk Service] ✅ Model loaded for language: {self.current_language}")
            return True
        except Exception as e:
            print(f"[Vosk Service] ❌ Error loading model: {e}")
            return False
    
    def _initialize_multiple_models(self):
        """Initialize multiple Vosk models for different languages."""
        models_config_path = os.path.join(self.base_dir, "vosk_models.csv")
        
        # Check if multi-model config exists
        if not os.path.exists(models_config_path):
            print("[Vosk Service] No vosk_models.csv found, using single model mode")
            return self._initialize_model()
        
        models_config = self._load_models_config(models_config_path)
        
        if not models_config:
            print("[Vosk Service] No models configured, falling back to single model")
            return self._initialize_model()
        
        # Load all configured models
        loaded_count = 0
        for lang_code, model_path in models_config.items():
            try:
                full_model_path = os.path.join(self.base_dir, model_path)
                
                # Check if model directory exists
                if not os.path.exists(full_model_path):
                    print(f"[Vosk Service] ⚠️ Model not found for {lang_code}: {full_model_path}")
                    continue
                
                print(f"[Vosk Service] Loading model for {lang_code} from {full_model_path}...")
                model = Model(full_model_path)
                recognizer = KaldiRecognizer(model, self.sample_rate)
                recognizer.SetWords(True)
                
                self.models[lang_code] = model
                self.recognizers[lang_code] = recognizer
                loaded_count += 1
                print(f"[Vosk Service] ✅ Model loaded for {lang_code}")
                
            except Exception as e:
                print(f"[Vosk Service] ❌ Error loading model for {lang_code}: {e}")
        
        if loaded_count == 0:
            print("[Vosk Service] ❌ No models loaded successfully")
            return False
        
        # Set default language to first loaded model
        if self.current_language not in self.recognizers:
            self.current_language = list(self.recognizers.keys())[0]
        
        print(f"[Vosk Service] ✅ Loaded {loaded_count} models: {list(self.models.keys())}")
        print(f"[Vosk Service] Default language: {self.current_language}")
        return True
    
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
    
    def switch_language(self, language_code: str) -> bool:
        """
        Switch to a different language model.
        
        Args:
            language_code: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            True if switched successfully, False otherwise
        """
        with self.language_lock:
            if language_code not in self.recognizers:
                print(f"[Vosk Service] Language '{language_code}' not available")
                return False
            
            if language_code == self.current_language:
                return True  # Already using this language
            
            old_language = self.current_language
            self.current_language = language_code
            print(f"[Vosk Service] 🔄 Switched language: {old_language} → {language_code}")
            return True
    
    def get_current_language(self) -> str:
        """Get the current active language code."""
        with self.language_lock:
            return self.current_language
    
    def get_available_languages(self) -> list:
        """Get list of available language codes."""
        return list(self.models.keys())
    
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
                
                # Get current recognizer (thread-safe)
                with self.language_lock:
                    current_recognizer = self.recognizers.get(self.current_language)
                
                if not current_recognizer:
                    continue
                
                # Feed to recognizer
                if current_recognizer.AcceptWaveform(audio_bytes):
                    # Finalized segment
                    result = json.loads(current_recognizer.Result())
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
                    partial = json.loads(current_recognizer.PartialResult())
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
        
        # Initialize models (supports both single and multiple models)
        if not self._initialize_multiple_models():
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
