#!/usr/bin/env python3
"""
UDP Handler Module
Centralizes all UDP send/receive logic for the modular audio system.
"""

import socket
import json
import threading
import queue
from typing import Dict, Callable, Optional


class UDPHandler:
    """
    Centralized UDP communication handler for all services.
    Manages UDP sockets and message routing.
    """
    
    def __init__(self, default_ip: str = "127.0.0.1"):
        """
        Initialize the UDP handler.
        
        Args:
            default_ip: Default IP address for UDP communication
        """
        self.default_ip = default_ip
        self.sockets: Dict[int, socket.socket] = {}
        self.listeners: Dict[int, threading.Thread] = {}
        self.running_ports: Dict[int, bool] = {}  # Per-port running flags
        self.callbacks: Dict[int, Callable] = {}
        
    def send_message(self, message: str, port: int, ip: Optional[str] = None) -> bool:
        """
        Send a UDP message to the specified port.
        
        Args:
            message: Message string to send
            port: Destination UDP port
            ip: Destination IP (uses default if None)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            if not message:
                return False
                
            target_ip = ip or self.default_ip
            
            # Create socket for this port if it doesn't exist
            if port not in self.sockets:
                self.sockets[port] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            self.sockets[port].sendto(message.encode("utf-8"), (target_ip, port))
            return True
            
        except Exception as e:
            print(f"[UDP Handler] Send error on port {port}: {e}")
            return False
    
    def send_json(self, data: dict, port: int, ip: Optional[str] = None) -> bool:
        """
        Send a JSON-encoded message via UDP.
        
        Args:
            data: Dictionary to send as JSON
            port: Destination UDP port
            ip: Destination IP (uses default if None)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            message = json.dumps(data)
            return self.send_message(message, port, ip)
        except Exception as e:
            print(f"[UDP Handler] JSON encoding error: {e}")
            return False
    
    def start_listener(self, port: int, callback: Callable[[str, tuple], None]) -> bool:
        """
        Start a UDP listener on the specified port.
        
        Args:
            port: Port to listen on
            callback: Function to call with (message, address) when data received
            
        Returns:
            True if listener started successfully, False otherwise
        """
        try:
            if port in self.listeners:
                print(f"[UDP Handler] Listener already running on port {port}")
                return False
            
            # Create and bind socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.default_ip, port))
            sock.settimeout(0.5)  # Allow periodic checking of running flag
            self.sockets[port] = sock
            self.callbacks[port] = callback
            
            # Start listener thread with per-port running flag
            self.running_ports[port] = True
            listener_thread = threading.Thread(
                target=self._listen_loop,
                args=(port,),
                daemon=False
            )
            listener_thread.start()
            self.listeners[port] = listener_thread
            
            print(f"[UDP Handler] Listener started on port {port}")
            return True
            
        except Exception as e:
            print(f"[UDP Handler] Failed to start listener on port {port}: {e}")
            return False
    
    def _listen_loop(self, port: int):
        """
        Internal loop for receiving UDP messages.
        
        Args:
            port: Port to listen on
        """
        sock = self.sockets[port]
        callback = self.callbacks[port]
        
        while self.running_ports.get(port, False):
            try:
                data, addr = sock.recvfrom(4096)
                message = data.decode("utf-8")
                callback(message, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running_ports.get(port, False):
                    print(f"[UDP Handler] Listener error on port {port}: {e}")
    
    def stop_listener(self, port: int):
        """
        Stop a UDP listener on the specified port.
        
        Args:
            port: Port to stop listening on
        """
        if port in self.listeners:
            print(f"[UDP Handler] Stopping listener on port {port}")
            self.running_ports[port] = False
            self.listeners[port].join(timeout=2.0)
            del self.listeners[port]
            if port in self.callbacks:
                del self.callbacks[port]
            if port in self.running_ports:
                del self.running_ports[port]
    
    def close_all(self):
        """Close all UDP sockets and stop all listeners."""
        # Stop all listeners
        for port in list(self.listeners.keys()):
            self.stop_listener(port)
        
        # Close all sockets
        for sock in self.sockets.values():
            try:
                sock.close()
            except:
                pass
        
        self.sockets.clear()
        print("[UDP Handler] All sockets closed")
