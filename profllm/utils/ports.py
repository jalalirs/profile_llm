# Port utilities
"""Port allocation and management utilities"""

import asyncio
import socket
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def find_free_port(start_port: int = 8000, end_port: int = 9000) -> int:
    """Find a free port in the specified range"""
    for port in range(start_port, end_port):
        if await check_port_available(port):
            logger.info(f"Found free port: {port}")
            return port
    
    raise RuntimeError(f"No free ports available in range {start_port}-{end_port}")

async def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is available"""
    try:
        # Create socket and try to bind
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1)
        
        result = sock.connect_ex((host, port))
        sock.close()
        
        # If connection failed, port is available
        return result != 0
        
    except Exception as e:
        logger.debug(f"Error checking port {port}: {str(e)}")
        return False

class PortManager:
    """Manages port allocation for multiple processes"""
    
    def __init__(self):
        self.allocated_ports = set()
    
    async def allocate_port(self, start_port: int = 8000, end_port: int = 9000) -> int:
        """Allocate a port and track it"""
        for port in range(start_port, end_port):
            if port not in self.allocated_ports and await check_port_available(port):
                self.allocated_ports.add(port)
                logger.info(f"Allocated port: {port}")
                return port
        
        raise RuntimeError(f"No free ports available in range {start_port}-{end_port}")
    
    def release_port(self, port: int):
        """Release a previously allocated port"""
        if port in self.allocated_ports:
            self.allocated_ports.remove(port)
            logger.info(f"Released port: {port}")
    
    def release_all(self):
        """Release all allocated ports"""
        logger.info(f"Releasing {len(self.allocated_ports)} ports")
        self.allocated_ports.clear()
