#!/usr/bin/env python3
"""
YAMNet Service Module
Handles YAMNet sound event detection in its own thread.
"""

import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import requests
import json
import threading
import queue
import time
from typing import Optional


class YAMNetService:
    """
    YAMNet-based sound detection service.
    Processes audio in a separate thread and sends detected events via UDP.
    """
    
    def __init__(self, config_path: str, udp_handler, sample_rate: int = 16000):
        """
        Initialize the YAMNet service.
        
        Args:
            config_path: Path to YAMNet configuration CSV file
            udp_handler: UDPHandler instance for sending messages
            sample_rate: Audio sample rate (YAMNet expects 16kHz)
        """
        self.sample_rate = sample_rate
        self.udp_handler = udp_handler
        self.running = False
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize queue with size from config
        queue_size = self.config.get("QUEUE_SIZE", 100)
        self.audio_queue = queue.Queue(maxsize=queue_size)
        
        # Model (lazy loaded)
        self.model = None
        self.class_names = []
        
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
            print(f"[YAMNet Service] Configuration loaded from {config_path}")
        except Exception as e:
            print(f"[YAMNet Service] Error loading config: {e}")
        return config
    
    def _load_class_names(self):
        """
        Load YAMNet class names from the AudioSet CSV.
        
        Downloads the official class map CSV from the TensorFlow models repository.
        Falls back to generic class names if download fails.
        """
        CLASS_MAP_URL = (
            "https://raw.githubusercontent.com/tensorflow/models/master/"
            "research/audioset/yamnet/yamnet_class_map.csv"
        )
        try:
            print("[YAMNet Service] Downloading class map CSV...")
            response = requests.get(CLASS_MAP_URL, timeout=5)
            response.raise_for_status()
            class_names = [
                line.split(",")[2].strip('"')
                for line in response.text.splitlines()[1:]
            ]
            print(f"[YAMNet Service] ✅ Loaded {len(class_names)} class names")
            return class_names
        except Exception as e:
            print(f"[YAMNet Service] ⚠️ Failed to load class map, using generic labels: {e}")
            return [f"class_{i}" for i in range(521)]
    
    def _initialize_model(self):
        """Initialize the YAMNet model."""
        try:
            print("[YAMNet Service] Loading model from TensorFlow Hub...")
            self.model = hub.KerasLayer(
                "https://tfhub.dev/google/yamnet/1",
                trainable=False,
            )
            # Load class names
            self.class_names = self._load_class_names()
            print("[YAMNet Service] ✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"[YAMNet Service] ❌ Error loading model: {e}")
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
            pass  # Skip if queue is full (real-time priority)
    
    def _detection_loop(self):
        """Main detection loop running in a separate thread."""
        print("[YAMNet Service] Detection thread started")
        
        # Get configuration
        udp_port = self.config.get("UDP_PORT", 7204)
        confidence_threshold = self.config.get("CONFIDENCE_THRESHOLD", 0.3)
        
        while self.running:
            try:
                # Get audio from queue with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Ensure audio is 1D and in the correct format
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
                
                # YAMNet expects float32 in range [-1.0, 1.0]
                waveform = np.asarray(audio_data, dtype=np.float32)
                
                if waveform.size == 0:
                    continue
                
                # Check if model is loaded
                if self.model is None:
                    print("[YAMNet Service] Model not loaded, cannot run inference.")
                    continue

                # Run inference
                scores, embeddings, spectrogram = self.model(waveform)
                scores = scores.numpy()  # shape: (frames, 521)
                
                if scores.ndim != 2 or scores.shape[0] == 0:
                    continue
                
                # Get mean scores across all frames
                mean_scores = scores.mean(axis=0)
                
                # Get top prediction
                top_index = int(np.argmax(mean_scores))
                confidence = float(mean_scores[top_index])
                
                # Send detection if above threshold
                if confidence >= confidence_threshold:
                    class_name = self.class_names[top_index] if top_index < len(self.class_names) else f"class_{top_index}"
                    
                    detection = {
                        "event": class_name,
                        "class_id": int(top_index),
                        "confidence": round(confidence, 3),
                        "timestamp": time.time()
                    }
                    
                    self.udp_handler.send_json(detection, udp_port)
                    #print(f"[YAMNet Service] Detected: {class_name} (confidence {confidence:.2f})")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[YAMNet Service] Detection error: {e}")
                time.sleep(0.1)
        
        print("[YAMNet Service] Detection thread stopped")
    
    def start(self) -> bool:
        """Start the YAMNet service in a separate thread."""
        if self.running:
            print("[YAMNet Service] Already running")
            return False
        
        # Initialize model
        if not self._initialize_model():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=False)
        self.thread.start()
        print("[YAMNet Service] Service started")
        return True
    
    def stop(self):
        """Stop the YAMNet service."""
        print("[YAMNet Service] Stopping...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("[YAMNet Service] Warning: Thread did not stop within timeout")
        print("[YAMNet Service] Service stopped")
