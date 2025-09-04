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

from ..config.models import ServerConfig, SystemConfig
from ..config.validation import get_vllm_server_args
from ..utils.ports import find_free_port
from ..utils.gpu import allocate_gpu_devices

logger = logging.getLogger(__name__)

class VLLMServerManager:
    """Manages vLLM server lifecycle"""
    
    def __init__(self, config: ServerConfig, system_config: Optional[SystemConfig] = None):
        self.config = config
        self.system_config = system_config
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
        
        # Ensure we have a clean environment copy
        logger.info(f"Base environment has {len(env)} variables")
        
        # Set Hugging Face cache directory
        hf_cache_dir = os.path.join(os.getcwd(), "models")
        env['HF_HOME'] = hf_cache_dir
        env['TRANSFORMERS_CACHE'] = hf_cache_dir
        env['HF_DATASETS_CACHE'] = hf_cache_dir
        
        # Check for Hugging Face token for gated models
        hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        if hf_token:
            env['HUGGINGFACE_HUB_TOKEN'] = hf_token
            env['HF_TOKEN'] = hf_token
            logger.info("Hugging Face token found - authentication enabled for gated models")
        else:
            logger.warning("No Hugging Face token found - gated models may not be accessible")
        
        # GPU device allocation
        if hasattr(self.config, 'gpu_devices') and self.config.gpu_devices:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.config.gpu_devices))
        elif self.config.tensor_parallel_size > 1:
            # Auto-allocate GPUs for tensor parallelism
            gpu_devices = await allocate_gpu_devices(self.config.tensor_parallel_size)
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_devices))
        
        # Profiling environment - enable if configured or if experiment profile dir is set
        profiling_enabled = False
        nsight_enabled = False
        
        if self.system_config:
            profiling_enabled = self.system_config.enable_profiling
            nsight_enabled = self.system_config.enable_nsight
        elif (hasattr(self.config, 'enable_profiling') and self.config.enable_profiling) or \
           (hasattr(self.config, 'experiment_profile_dir') and self.config.experiment_profile_dir):
            profiling_enabled = True
            
        if profiling_enabled and not nsight_enabled:
            # Use experiment-specific profiling directory
            if hasattr(self.config, 'experiment_profile_dir') and self.config.experiment_profile_dir:
                profile_dir = Path(self.config.experiment_profile_dir)
            else:
                # Fallback to global directory
                profile_dir = Path(self.config.profile_dir or '/tmp/profllm_traces')
            
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            # Set environment variable for subprocess
            env['VLLM_TORCH_PROFILER_DIR'] = str(profile_dir)
            
            # Debug: Log the profiling directory being set
            logger.info(f"Torch profiling enabled, traces will be saved to: {profile_dir}")
            logger.info(f"vLLM will automatically generate traces when --profile flag is used in benchmark")
        
        # Build command based on profiling configuration
        if nsight_enabled:
            # Nsight Systems profiling
            nsight_output_dir = self.system_config.nsight_output_dir or '/tmp/profllm_nsight'
            nsight_output_path = Path(nsight_output_dir) / f"vllm_server_{int(time.time())}.nsys-rep"
            nsight_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build nsys command with high verbosity options
            nsys_cmd = ["nsys", "profile", "-o", str(nsight_output_path)]
            
            # Build trace options list
            trace_options = []
            if self.system_config.nsight_trace:
                trace_options.extend(self.system_config.nsight_trace.split(','))
            
            # Add additional trace options if enabled
            if self.system_config.nsight_mpi_trace and "mpi" not in trace_options:
                trace_options.append("mpi")
            
            if self.system_config.nsight_osrt_trace and "osrt" not in trace_options:
                trace_options.append("osrt")
            
            # Add trace option (only once)
            if trace_options:
                nsys_cmd.extend(["--trace", ",".join(trace_options)])
            
            if self.system_config.nsight_trace_fork:
                nsys_cmd.append("--trace-fork-before-exec=true")
            
            if self.system_config.nsight_cuda_graph_trace:
                nsys_cmd.extend(["--cuda-graph-trace", self.system_config.nsight_cuda_graph_trace])
            
            # High verbosity CUDA tracing
            if self.system_config.nsight_cuda_trace_all_apis:
                nsys_cmd.append("--cuda-trace-all-apis=true")
            
            if self.system_config.nsight_cuda_memory_usage:
                nsys_cmd.append("--cuda-memory-usage=true")
            
            if self.system_config.nsight_cuda_backtrace:
                nsys_cmd.extend(["--cudabacktrace", self.system_config.nsight_cuda_backtrace])
            
            if self.system_config.nsight_cuda_flush_interval:
                nsys_cmd.extend(["--cuda-flush-interval", str(self.system_config.nsight_cuda_flush_interval)])
            
            # CPU sampling options
            if self.system_config.nsight_sample:
                nsys_cmd.extend(["--sample", str(self.system_config.nsight_sample)])
            
            # Note: --sampling-frequency is not a valid nsys option, using --sampling-period instead
            if self.system_config.nsight_sampling_frequency:
                # Convert frequency to period (period = 1/frequency * 1000000 for microseconds)
                # nsys requires period between 125000 and 16000000
                period = int(1000000 / self.system_config.nsight_sampling_frequency)
                # Ensure period is within valid range
                period = max(125000, min(16000000, period))
                nsys_cmd.extend(["--sampling-period", str(period)])
            
            if self.system_config.nsight_cpu_core_events:
                nsys_cmd.extend(["--cpu-core-events", self.system_config.nsight_cpu_core_events])
            
            if self.system_config.nsight_event_sample:
                nsys_cmd.extend(["--event-sample", self.system_config.nsight_event_sample])
            
            # Advanced CPU tracing options
            # Note: "cpu" is not a valid trace option in nsys, so we use other methods for CPU tracing
            
            # Note: --trace-fork-before-exec is already handled above in the main trace options
            # No need to add it again here
            
            # OS runtime tracing includes syscalls, memory operations, I/O, etc.
            if self.system_config.nsight_cpu_trace_syscalls:
                # osrt is already included in the main trace options above
                pass
            
            # CPU sampling and profiling
            # Note: CPU sampling is handled by the main --sampling-period option above
            # For now, we'll use the main sampling options and not override them
            # This avoids complex list manipulation that might cause command parsing issues
            
            # CPU performance counters and profiling options
            # Note: Many of these options are not valid in nsys profile
            # We'll keep only the valid ones and use the main sampling/tracing options
            
            # Call stack and symbol resolution
            # Note: These options are not valid in nsys profile
            # --call-stack, --symbols, --source-lines, --debug-info are not supported
            # Symbol resolution and call stacks are handled automatically by nsys
            
            # Note: The following options are not valid in nsys profile:
            # --cpu-counters, --cpu-counter-events, --cpu-counter-scope
            # --kernel-trace, --user-trace, --context-switches
            # --memory-bandwidth, --cache-events, --branch-events
            # --sample-all-threads, --sample-kernel, --sample-user, --sample-idle
            # These features are achieved through the main trace and sampling options
            
            # Output options
            if self.system_config.nsight_stats:
                nsys_cmd.append("--stats=true")
            
            if self.system_config.nsight_export:
                nsys_cmd.extend(["--export", self.system_config.nsight_export])
            
            # Timing options
            if self.system_config.nsight_delay is not None:
                nsys_cmd.extend(["--delay", str(self.system_config.nsight_delay)])
            
            if self.system_config.nsight_duration is not None:
                nsys_cmd.extend(["--duration", str(self.system_config.nsight_duration)])
            
            # Add the vLLM command
            vllm_cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server"] + server_args
            cmd = nsys_cmd + vllm_cmd
            
            logger.info(f"Nsight Systems profiling enabled")
            logger.info(f"Nsight output will be saved to: {nsight_output_path}")
            logger.info(f"Nsight command: {' '.join(nsys_cmd)}")
            
            # Debug: Log the final command to see what's being generated
            logger.debug(f"Final command: {' '.join(cmd)}")
            
        elif profiling_enabled:
            # Torch profiler
            profile_dir = Path(self.config.experiment_profile_dir) if hasattr(self.config, 'experiment_profile_dir') and self.config.experiment_profile_dir else Path(self.config.profile_dir or '/tmp/profllm_traces')
            # Resolve to absolute path
            profile_dir = profile_dir.resolve()
            # Use shell=False but ensure env var is in env dict for proper inheritance
            cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server"] + server_args
            env['VLLM_TORCH_PROFILER_DIR'] = str(profile_dir)
            logger.info(f"Torch profiling enabled - using environment variable in subprocess env")
            logger.info(f"Profiling directory resolved to absolute path: {profile_dir}")
        else:
            # No profiling
            cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server"] + server_args
        
        logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Environment variables count: {len(env)}")
        
        # Debug: Log profiling setup
        if 'VLLM_TORCH_PROFILER_DIR' in env:
            logger.info(f"Profiling environment variable VLLM_TORCH_PROFILER_DIR={env['VLLM_TORCH_PROFILER_DIR']} is set")
        
        # Debug: Check if vLLM module is available
        try:
            import vllm
            logger.info(f"vLLM module found: {vllm.__file__}")
        except ImportError as e:
            logger.error(f"vLLM module not found: {e}")
            raise RuntimeError("vLLM module not available")
        
        try:
            # Use shell=False for better environment variable inheritance
            logger.info("Creating subprocess...")
            self.process = subprocess.Popen(
                cmd,
                shell=False,
                env=env,
                stdout=subprocess.PIPE,  # Capture stdout for debugging
                stderr=subprocess.PIPE,  # Capture stderr for debugging
            )
            
            logger.info(f"Subprocess created successfully with PID {self.process.pid}")
            logger.info(f"Server will be available at {self.base_url}")
            
            # Debug: Verify environment variable is set in subprocess
            if 'VLLM_TORCH_PROFILER_DIR' in env:
                logger.info(f"✓ Environment variable VLLM_TORCH_PROFILER_DIR={env['VLLM_TORCH_PROFILER_DIR']} is set for subprocess")
            else:
                logger.warning("⚠ VLLM_TORCH_PROFILER_DIR not set - profiling will not work")
            
            # Update config with actual port
            self.config.port = self.port
            
            # Verify process was created
            if self.process is None:
                raise RuntimeError("Failed to create subprocess - process is None")
            
            # Give the process more time to start up before checking
            # For distributed setups (TP > 1 or PP > 1), need more time for process synchronization
            total_processes = self.config.tensor_parallel_size * self.config.pipeline_parallel_size
            startup_delay = max(5.0, total_processes * 2.0)  # At least 5 seconds, or 2 seconds per process
            
            logger.info(f"Waiting for vLLM server process to stabilize...")
            logger.info(f"Distributed setup: TP={self.config.tensor_parallel_size}, PP={self.config.pipeline_parallel_size} ({total_processes} total processes)")
            logger.info(f"Startup delay: {startup_delay} seconds")
            await asyncio.sleep(startup_delay)
            
            # Now check if it's still running
            if self.process.poll() is not None:
                # Process died during startup - capture output for debugging
                stdout, stderr = self.process.communicate()
                error_msg = f"vLLM server process died during startup with exit code {self.process.returncode}"
                logger.error(error_msg)
                
                if stdout:
                    logger.error(f"STDOUT: {stdout.decode('utf-8', errors='ignore')}")
                if stderr:
                    logger.error(f"STDERR: {stderr.decode('utf-8', errors='ignore')}")
                
                raise RuntimeError(error_msg)
            
            logger.info("✓ vLLM server process is stable and running")
            
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {str(e)}")
            # Ensure process is None on failure
            self.process = None
            raise
        
        return self.base_url
    
    def _build_server_args(self) -> List[str]:
        """Build command line arguments for vLLM server"""
        args = []
        
        # Convert config to vLLM args
        vllm_args = get_vllm_server_args(self.config)
        
        # Debug: Log the vLLM args
        print(f"DEBUG: vLLM args from config: {vllm_args}")
        
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
        
        # Check if process was created
        if self.process is None:
            raise RuntimeError("Cannot wait for server - no process was created")
        
        # For distributed setups, use longer timeout and less frequent checks
        total_processes = self.config.tensor_parallel_size * self.config.pipeline_parallel_size
        if total_processes > 1:
            timeout = max(timeout, total_processes * 60)  # At least 1 minute per process
            check_interval = max(check_interval, 5.0)  # Check every 5 seconds for distributed
            logger.info(f"Distributed setup detected ({total_processes} processes) - using extended timeout: {timeout}s, check interval: {check_interval}s")
        
        start_time = time.time()
        last_error = None
        check_count = 0
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            if self.process.poll() is not None:
                # Process died - capture output for debugging
                stdout, stderr = self.process.communicate()
                exit_code = self.process.returncode
                error_msg = f"vLLM server process died with exit code {exit_code}"
                
                if stdout:
                    logger.error(f"STDOUT: {stdout.decode('utf-8', errors='ignore')}")
                if stderr:
                    logger.error(f"STDERR: {stderr.decode('utf-8', errors='ignore')}")
                
                raise RuntimeError(error_msg)
            
            try:
                # Try to connect to health endpoint
                response = requests.get(f"{self.base_url}/health", timeout=10)  # Longer timeout for health check
                if response.status_code == 200:
                    logger.info("vLLM server is ready")
                    return
            except Exception as e:
                last_error = str(e)
                check_count += 1
                if check_count % 10 == 0:  # Log every 10th failed attempt
                    logger.info(f"Health check attempt {check_count}: {last_error}")
            
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

