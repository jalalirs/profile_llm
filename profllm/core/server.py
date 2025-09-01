"""vLLM server lifecycle management"""

import asyncio
import logging
import subprocess
import time
import requests
import signal
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..config.models import ServerConfig
from ..config.validation import get_vllm_server_args
from ..utils.ports import find_free_port
from ..utils.gpu import allocate_gpu_devices

logger = logging.getLogger(__name__)

class VLLMServerManager:
    """Manages vLLM server lifecycle"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.base_url: Optional[str] = None
        
    async def start(self) -> str:
        """Start vLLM server and return base URL"""
        logger.info("Starting vLLM server...")
        
        # Allocate port if not specified
        if self.config.port is None:
            self.port = await find_free_port()
        else:
            self.port = self.config.port
            
        self.base_url = f"http://{self.config.host}:{self.port}"
        
        # Prepare server arguments
        server_args = self._build_server_args()
        
        # Set up environment variables
        env = os.environ.copy()
        
        # GPU device allocation
        if hasattr(self.config, 'gpu_devices') and self.config.gpu_devices:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.config.gpu_devices))
        elif self.config.tensor_parallel_size > 1:
            # Auto-allocate GPUs for tensor parallelism
            gpu_devices = await allocate_gpu_devices(self.config.tensor_parallel_size)
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_devices))
        
        # Profiling environment
        if hasattr(self.config, 'enable_profiling') and self.config.enable_profiling:
            profile_dir = getattr(self.config, 'profile_dir', '/tmp/profllm_traces')
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            env['VLLM_TORCH_PROFILER_DIR'] = profile_dir
        
        # Start server process
        cmd = ['python3', '-m', 'vllm.entrypoints.openai.api_server'] + server_args
        
        logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            logger.info(f"vLLM server started with PID {self.process.pid}")
            logger.info(f"Server will be available at {self.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {str(e)}")
            raise
        
        return self.base_url
    
    def _build_server_args(self) -> List[str]:
        """Build command line arguments for vLLM server"""
        args = []
        
        # Convert config to vLLM args
        vllm_args = get_vllm_server_args(self.config)
        
        for key, value in vllm_args.items():
            if isinstance(value, bool):
                if value:
                    args.append(f'--{key}')
            elif isinstance(value, list):
                if value:  # Only add if list is not empty
                    args.extend([f'--{key}'] + [str(v) for v in value])
            elif value is not None:
                args.extend([f'--{key}', str(value)])
        
        # Add server-specific args
        args.extend(['--host', self.config.host])
        args.extend(['--port', str(self.port)])
        
        if self.config.api_key:
            args.extend(['--api-key', self.config.api_key])
        
        return args
    
    async def wait_for_ready(self, timeout: int = 300, check_interval: float = 2.0):
        """Wait for server to be ready to accept requests"""
        logger.info("Waiting for vLLM server to be ready...")
        
        start_time = time.time()
        last_error = None
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"vLLM server process died. STDOUT: {stdout}, STDERR: {stderr}")
            
            try:
                # Try to connect to health endpoint
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM server is ready")
                    return
            except Exception as e:
                last_error = str(e)
            
            await asyncio.sleep(check_interval)
        
        raise TimeoutError(f"vLLM server did not become ready within {timeout} seconds. Last error: {last_error}")
    
    async def stop(self):
        """Stop the vLLM server"""
        if not self.process:
            return
        
        logger.info(f"Stopping vLLM server (PID: {self.process.pid})")
        
        try:
            # Try graceful shutdown first
            self.process.send_signal(signal.SIGTERM)
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=30)
                logger.info("vLLM server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning("Graceful shutdown timed out, force killing...")
                self.process.send_signal(signal.SIGKILL)
                self.process.wait(timeout=10)
                logger.info("vLLM server force killed")
                
        except Exception as e:
            logger.error(f"Error stopping vLLM server: {str(e)}")
        finally:
            self.process = None
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "base_url": self.base_url,
            "port": self.port,
            "pid": self.process.pid if self.process else None,
            "is_running": self.process is not None and self.process.poll() is None
        }

