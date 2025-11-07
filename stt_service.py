#!/usr/bin/env python3
"""
STT Service Orchestrator
Main orchestrator for the modular real-time audio system.
Manages Vosk, Whisper, and YAMNet services.
"""

import os
import csv
import argparse
import sounddevice as sd
import numpy as np
import time
import signal
import sys

from udp_handler import UDPHandler
from vosk_service import VoskService
from whisper_service import WhisperService
from yamnet_service import YAMNetService


class STTOrchestrator:
    """
    Main orchestrator for the modular audio system.
    Manages lifecycle of all audio processing services.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the orchestrator.
        
        Args:
            base_dir: Base directory for configuration files and models
        """
        self.base_dir = base_dir
        self.running = False
        
        # Load orchestrator configuration
        self.config = self._load_config(os.path.join(base_dir, "orchestrator_config.csv"))
        
        # Audio settings
        self.sample_rate = self.config.get("SAMPLE_RATE", 16000)
        self.block_size = self.config.get("BLOCK_SIZE", 4000)
        
        # Initialize UDP handler
        self.udp_handler = UDPHandler()
        
        # Initialize services
        self.services = {}
        self._initialize_services()
        
        # Audio stream
        self.stream = None
        
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
                        if value.isdigit():
                            config[key] = int(value)
                        elif value.lower() in ['true', 'false']:
                            config[key] = value.lower() == 'true'
                        else:
                            config[key] = value
            print(f"[Orchestrator] Configuration loaded from {config_path}")
        except Exception as e:
            print(f"[Orchestrator] Error loading config: {e}")
        return config
    
    def _initialize_services(self):
        """Initialize all configured services."""
        print("[Orchestrator] Initializing services...")
        
        # Vosk service
        if self.config.get("VOSK_ENABLED", True):
            try:
                vosk_config = os.path.join(self.base_dir, "vosk_config.csv")
                self.services["vosk"] = VoskService(
                    vosk_config,
                    self.udp_handler,
                    self.sample_rate,
                    self.base_dir
                )
                print("[Orchestrator] ✅ Vosk service initialized")
            except Exception as e:
                print(f"[Orchestrator] ⚠️ Failed to initialize Vosk: {e}")
        
        # Whisper service
        if self.config.get("WHISPER_ENABLED", False):
            try:
                whisper_config = os.path.join(self.base_dir, "whisper_config.csv")
                self.services["whisper"] = WhisperService(
                    whisper_config,
                    self.udp_handler,
                    self.sample_rate
                )
                print("[Orchestrator] ✅ Whisper service initialized")
            except Exception as e:
                print(f"[Orchestrator] ⚠️ Failed to initialize Whisper: {e}")
        
        # YAMNet service
        if self.config.get("YAMNET_ENABLED", True):
            try:
                yamnet_config = os.path.join(self.base_dir, "yamnet_config.csv")
                self.services["yamnet"] = YAMNetService(
                    yamnet_config,
                    self.udp_handler,
                    self.sample_rate
                )
                print("[Orchestrator] ✅ YAMNet service initialized")
            except Exception as e:
                print(f"[Orchestrator] ⚠️ Failed to initialize YAMNet: {e}")
    
    def start_service(self, service_name: str) -> bool:
        """
        Start a specific service.
        
        Args:
            service_name: Name of the service to start ("vosk", "whisper", "yamnet")
            
        Returns:
            True if service started successfully, False otherwise
        """
        if service_name not in self.services:
            print(f"[Orchestrator] Service '{service_name}' not found")
            return False
        
        try:
            return self.services[service_name].start()
        except Exception as e:
            print(f"[Orchestrator] Error starting {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str):
        """
        Stop a specific service.
        
        Args:
            service_name: Name of the service to stop
        """
        if service_name in self.services:
            try:
                self.services[service_name].stop()
            except Exception as e:
                print(f"[Orchestrator] Error stopping {service_name}: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Audio callback function.
        Routes audio to all active services.
        """
        if status:
            print(f"[Orchestrator] Audio warning: {status}")
        
        # Get audio data as float32 array
        audio_float = np.asarray(indata, dtype=np.float32).flatten()
        
        # Route audio to all active services
        for service_name, service in self.services.items():
            if hasattr(service, 'running') and service.running:
                service.process_audio(audio_float)
    
    def start(self):
        """Start the orchestrator and all enabled services."""
        print("\n" + "="*60)
        print("STT Service Orchestrator - Modular Audio System")
        print("="*60 + "\n")
        
        # Start all enabled services
        for service_name in self.services:
            if not self.start_service(service_name):
                print(f"[Orchestrator] Warning: {service_name} failed to start")
        
        # Start audio stream
        print("\n[Orchestrator] Starting audio stream...")
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype='float32',
                channels=1,
                callback=self._audio_callback
            )
            self.stream.start()
            self.running = True
            
            print("\n" + "="*60)
            print("🎤 Listening... (Press Ctrl+C to stop)")
            print("="*60)
            
            # Print active services
            active_services = [name for name, svc in self.services.items() 
                             if hasattr(svc, 'running') and svc.running]
            print(f"\n📡 Active services: {', '.join(active_services)}")
            
            # Print UDP port information
            if "vosk" in self.services and self.services["vosk"].running:
                vosk_config = self.services["vosk"].config
                print(f"   • Vosk STT: ports {vosk_config.get('UDP_PORT_PARTIAL')}, "
                      f"{vosk_config.get('UDP_PORT_FINAL')}, {vosk_config.get('UDP_PORT_WORD_CONF')}")
            
            if "whisper" in self.services and self.services["whisper"].running:
                whisper_config = self.services["whisper"].config
                print(f"   • Whisper STT: ports {whisper_config.get('UDP_PORT_PARTIAL')}, "
                      f"{whisper_config.get('UDP_PORT_FINAL')}")
            
            if "yamnet" in self.services and self.services["yamnet"].running:
                yamnet_config = self.services["yamnet"].config
                print(f"   • YAMNet: port {yamnet_config.get('UDP_PORT')}")
            
            print("\n")
            
            # Main loop
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n[Orchestrator] Received interrupt signal")
        except Exception as e:
            print(f"[Orchestrator] Error in audio stream: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the orchestrator and all services."""
        if not self.running:
            return
        
        print("\n[Orchestrator] Stopping all services...")
        self.running = False
        
        # Stop audio stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        # Stop all services
        for service_name in list(self.services.keys()):
            self.stop_service(service_name)
        
        # Close UDP handler
        self.udp_handler.close_all()
        
        print("[Orchestrator] ✅ All services stopped gracefully")


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Modular STT Service Orchestrator"
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Base directory of the project"
    )
    parser.add_argument(
        "--enable-vosk",
        action="store_true",
        help="Enable Vosk service (overrides config)"
    )
    parser.add_argument(
        "--disable-vosk",
        action="store_true",
        help="Disable Vosk service (overrides config)"
    )
    parser.add_argument(
        "--enable-whisper",
        action="store_true",
        help="Enable Whisper service (overrides config)"
    )
    parser.add_argument(
        "--disable-whisper",
        action="store_true",
        help="Disable Whisper service (overrides config)"
    )
    parser.add_argument(
        "--enable-yamnet",
        action="store_true",
        help="Enable YAMNet service (overrides config)"
    )
    parser.add_argument(
        "--disable-yamnet",
        action="store_true",
        help="Disable YAMNet service (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = STTOrchestrator(args.base_dir)
    
    # Apply CLI overrides
    if args.enable_vosk:
        orchestrator.config["VOSK_ENABLED"] = True
    if args.disable_vosk:
        orchestrator.config["VOSK_ENABLED"] = False
    if args.enable_whisper:
        orchestrator.config["WHISPER_ENABLED"] = True
    if args.disable_whisper:
        orchestrator.config["WHISPER_ENABLED"] = False
    if args.enable_yamnet:
        orchestrator.config["YAMNET_ENABLED"] = True
    if args.disable_yamnet:
        orchestrator.config["YAMNET_ENABLED"] = False
    
    # Reinitialize services with updated config
    orchestrator.services = {}
    orchestrator._initialize_services()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        print("\n[Orchestrator] Received signal, shutting down...")
        orchestrator.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the orchestrator
    try:
        orchestrator.start()
    except Exception as e:
        print(f"[Orchestrator] Fatal error: {e}")
        orchestrator.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
