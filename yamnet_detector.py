#!/usr/bin/env python3
"""
YAMNet Sound Detection Module
Detects sound events using Google's YAMNet model.
Runs in a separate thread for independent operation.
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import requests
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
    
    def __init__(self, sample_rate=16000, confidence_threshold=0.3, queue_size=100):
        """
        Initialize the YAMNet detector.
        
        Args:
            sample_rate: Audio sample rate (YAMNet expects 16kHz)
            confidence_threshold: Minimum confidence for detection reporting
            queue_size: Maximum size of audio processing queue (default 100)
                       Larger values use more memory but handle bursty audio better.
                       When full, new audio is dropped to maintain real-time performance.
        """
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.audio_queue = queue.Queue(maxsize=queue_size)
        
        # UDP Settings for sound detection output
        self.udp_ip = "127.0.0.1"
        self.udp_port = 7204  # Separate port for sound detection
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Load YAMNet model
        print("[YAMNet] Loading model from TensorFlow Hub...")
        try:
            self.model = hub.KerasLayer(
                "https://tfhub.dev/google/yamnet/1",
                trainable=False,
            )
            # Load class names
            self.class_names = self._load_class_names()
            print("[YAMNet] ✅ Model loaded successfully")
        except Exception as e:
            print(f"[YAMNet] ❌ Error loading model: {e}")
            raise
    
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
            print("[YAMNet] Downloading class map CSV...")
            response = requests.get(CLASS_MAP_URL, timeout=5)
            response.raise_for_status()
            class_names = [
                line.split(",")[2].strip('"')
                for line in response.text.splitlines()[1:]
            ]
            print(f"[YAMNet] ✅ Loaded {len(class_names)} class names")
            return class_names
        except Exception as e:
            print(f"[YAMNet] ⚠️ Failed to load class map, using generic labels: {e}")
            return [f"class_{i}" for i in range(521)]
    
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
                # Audio comes as float32 from STT callback already normalized
                waveform = np.asarray(audio_data, dtype=np.float32)
                
                if waveform.size == 0:
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
                if confidence >= self.confidence_threshold:
                    class_name = self.class_names[top_index] if top_index < len(self.class_names) else f"class_{top_index}"
                    
                    detection = {
                        "event": class_name,
                        "class_id": int(top_index),
                        "confidence": round(confidence, 3),
                        "timestamp": time.time()
                    }
                    
                    self.send_udp(json.dumps(detection))
                    print(f"[YAMNet] Detected: {class_name} (confidence {confidence:.2f})")
                
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
        # Non-daemon thread for proper cleanup - will be joined in stop()
        self.thread = threading.Thread(target=self._detection_loop, daemon=False)
        self.thread.start()
        print("[YAMNet] Service started")
    
    def stop(self):
        """Stop the YAMNet detection service."""
        print("[YAMNet] Stopping...")
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("[YAMNet] Warning: Thread did not stop within timeout")
        self.sock.close()
        print("[YAMNet] Service stopped")
