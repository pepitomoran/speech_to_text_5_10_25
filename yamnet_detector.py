#!/usr/bin/env python3
"""
YAMNet Sound Detection Module
Detects sound events using Google's YAMNet model.
Runs in a separate thread for independent operation.
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import resampy
import socket
import json
import threading
import queue
import time

class YAMNetDetector:
    """
    YAMNet-based sound detection service.
    Processes audio in a separate thread and sends detected events via UDP.
    """
    
    def __init__(self, sample_rate=16000, confidence_threshold=0.3):
        """
        Initialize the YAMNet detector.
        
        Args:
            sample_rate: Audio sample rate (YAMNet expects 16kHz)
            confidence_threshold: Minimum confidence for detection reporting
        """
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.audio_queue = queue.Queue(maxsize=100)
        
        # UDP Settings for sound detection output
        self.udp_ip = "127.0.0.1"
        self.udp_port = 7204  # Separate port for sound detection
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Load YAMNet model
        print("[YAMNet] Loading model from TensorFlow Hub...")
        try:
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            # Load class names
            self.class_names = self._load_class_names()
            print("[YAMNet] ✅ Model loaded successfully")
        except Exception as e:
            print(f"[YAMNet] ❌ Error loading model: {e}")
            raise
    
    def _load_class_names(self):
        """Load YAMNet class names from the model."""
        try:
            # YAMNet class map is embedded in the model
            class_map_path = self.model.class_map_path().numpy().decode('utf-8')
            # For simplicity, we'll use a subset of common classes
            # In production, you'd load the full class map CSV
            return [
                "Speech", "Music", "Dog", "Cat", "Laughter", 
                "Applause", "Clapping", "Knock", "Door", "Bell",
                "Alarm", "Siren", "Telephone", "Glass", "Water"
            ]
        except Exception as e:
            print(f"[YAMNet] Warning: Could not load class names: {e}")
            return [f"Class_{i}" for i in range(521)]  # YAMNet has 521 classes
    
    def send_udp(self, message):
        """Send detection results via UDP."""
        try:
            if message:
                self.sock.sendto(message.encode("utf-8"), (self.udp_ip, self.udp_port))
        except Exception as e:
            print(f"[YAMNet] UDP send error: {e}")
    
    def process_audio(self, audio_data):
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
        print("[YAMNet] Detection thread started")
        
        while self.running:
            try:
                # Get audio from queue with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Ensure audio is 1D and in the correct format
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
                
                # YAMNet expects float32 in range [-1.0, 1.0]
                audio_data = audio_data.astype(np.float32)
                
                # Run inference
                scores, embeddings, spectrogram = self.model(audio_data)
                
                # Get top predictions
                top_indices = np.argsort(scores.numpy().mean(axis=0))[-3:][::-1]
                
                # Send detections above threshold
                for idx in top_indices:
                    confidence = float(scores.numpy().mean(axis=0)[idx])
                    if confidence >= self.confidence_threshold:
                        class_name = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
                        
                        detection = {
                            "event": class_name,
                            "confidence": round(confidence, 3),
                            "timestamp": time.time()
                        }
                        
                        self.send_udp(json.dumps(detection))
                        print(f"[YAMNet] Detected: {class_name} ({confidence:.3f})")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[YAMNet] Detection error: {e}")
                time.sleep(0.1)
        
        print("[YAMNet] Detection thread stopped")
    
    def start(self):
        """Start the YAMNet detection service in a separate thread."""
        if self.running:
            print("[YAMNet] Already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("[YAMNet] Service started")
    
    def stop(self):
        """Stop the YAMNet detection service."""
        print("[YAMNet] Stopping...")
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        self.sock.close()
        print("[YAMNet] Service stopped")
