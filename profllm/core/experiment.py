"""Core experiment orchestration"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from ..config.models import ExperimentConfig
from ..utils.system import get_system_info
from .server import VLLMServerManager
from .client import BenchmarkClient
from .metrics import SystemMetrics
from ..output.exporter import ResultExporter

logger = logging.getLogger(__name__)

class ExperimentResult:
    """Container for experiment results"""
    
    def __init__(self, run_id: str, config: ExperimentConfig):
        self.run_id = run_id
        self.config = config
        self.timestamp = datetime.now(timezone.utc)
        self.duration: Optional[int] = None
        self.results: Dict[str, Any] = {}
        self.system: Dict[str, Any] = {}
        self.profiling: Dict[str, Any] = {}
        self.status = "initialized"
        self.error: Optional[str] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching the JSON schema"""
        return {
            "run_id": self.run_id,
            "suite": self.config.suite,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "config": {
                "server": self.config.server.dict(),
                "benchmark": self.config.benchmark.dict()
            },
            "results": self.results,
            "system": self.system,
            "profiling": self.profiling,
            "status": self.status,
            "error": self.error
        }
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Save results as JSON"""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        
        return json_str

class Experiment:
    """Main experiment orchestrator"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.run_id = self._generate_run_id()
        self.result = ExperimentResult(self.run_id, config)
        
        # Components
        self.server_manager: Optional[VLLMServerManager] = None
        self.benchmark_client: Optional[BenchmarkClient] = None
        self.system_metrics: Optional[SystemMetrics] = None
        
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        suite = self.config.suite.replace(" ", "_").lower()
        return f"exp_{timestamp}_{suite}_{short_uuid}"
    
    async def run(self) -> 'ExperimentResult':
        """Run the complete experiment"""
        start_time = time.time()
        
        try:
            # Validate configuration
            await self._validate_config()
            
            # Initialize components
            await self._initialize_components()
            
            # Start system monitoring if enabled
            if self.system_metrics:
                await self.system_metrics.start_monitoring()
                logger.info("System monitoring started")
            
            # Start vLLM server
            base_url = await self.server_manager.start()
            logger.info(f"vLLM server started at {base_url}")
            
            # Wait for server to be ready
            await self.server_manager.wait_for_ready()
            
            # Update benchmark client with server info
            self.benchmark_client.update_server_info(
                host=self.server_manager.config.host,
                port=self.server_manager.port
            )
            
            # Run benchmark
            logger.info("Starting benchmark...")
            benchmark_result = await self.benchmark_client.run()
            
            # Collect system metrics
            system_data = None
            if self.system_metrics:
                system_data = await self.system_metrics.stop_monitoring()
                logger.info("System monitoring stopped")
            
            # Collect profiling data
            profiling_data = await self._collect_profiling_data()
            
            # Update result with collected data
            self.result.results.update(benchmark_result)
            if system_data:
                self.result.system.update(system_data)
            if profiling_data:
                self.result.profiling.update(profiling_data)
            self.result.status = "completed"
            
            # Save results with organized structure
            await self._save_results()
            
            logger.info("Experiment completed successfully")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        
        finally:
            # Always cleanup, even on failure
            try:
                await self._cleanup()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
            
            # Calculate duration
            self.result.duration = int(time.time() - start_time)
        
        return self.result
    
    async def _validate_config(self):
        """Validate experiment configuration"""
        from ..config.validation import validate_config
        
        warnings = validate_config(self.config)
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")
    
    async def _initialize_components(self):
        """Initialize experiment components"""
        # Create experiment directory structure first
        if self.config.output_path:
            results_dir = Path(self.config.output_path).parent
            run_dir = results_dir / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up profiling directory within experiment
            if self.config.system.enable_profiling:
                profiling_dir = run_dir / "profiling"
                profiling_dir.mkdir(parents=True, exist_ok=True)
                # Pass profiling directory to server
                self.config.server.experiment_profile_dir = str(profiling_dir)
                logger.info(f"Created profiling directory: {profiling_dir}")
                logger.info(f"vLLM will save traces here when --profile flag is used in benchmark")
        
        # Server manager
        self.server_manager = VLLMServerManager(self.config.server, self.config.system)
        
        # System metrics
        if self.config.system.monitor_gpu or self.config.system.monitor_cpu:
            self.system_metrics = SystemMetrics(self.config.system)
        
        # Benchmark client (will be initialized after server starts)
        self.benchmark_client = BenchmarkClient(
            self.config.benchmark,
            server_config=self.config.server,
            system_config=self.config.system
        )
    
    async def _collect_profiling_data(self) -> Dict[str, Any]:
        """Collect profiling data"""
        profiling_data = {"enabled": True}
        
        if self.config.system.enable_profiling and self.config.output_path:
            # Look for trace files in the experiment's profiling directory
            results_dir = Path(self.config.output_path).parent
            run_dir = results_dir / self.run_id
            profiling_dir = run_dir / "profiling"
            
            if profiling_dir.exists():
                # Look for various trace file types that vLLM might generate
                trace_files = list(profiling_dir.glob("*.perfetto.gz")) + \
                             list(profiling_dir.glob("*.perfetto")) + \
                             list(profiling_dir.glob("*.gz"))
                
                if trace_files:
                    trace_file = trace_files[0]
                    profiling_data.update({
                        "trace_file_path": str(trace_file),
                        "trace_file_size_mb": trace_file.stat().st_size / (1024 * 1024),
                        "trace_file_count": len(trace_files),
                        "profiling_directory": str(profiling_dir),
                        "trace_file_types": [f.name for f in trace_files]
                    })
                    
                    logger.info(f"Found {len(trace_files)} trace files in {profiling_dir}")
                    logger.info(f"Trace file types: {[f.name for f in trace_files]}")
                else:
                    logger.warning(f"No trace files found in {profiling_dir}")
                    logger.info(f"Contents of profiling directory: {list(profiling_dir.iterdir())}")
            else:
                logger.warning(f"Profiling directory does not exist: {profiling_dir}")
        
        return profiling_data
    
    # Removed _copy_trace_files_to_run_dir method - no longer needed
    
    async def _save_results(self):
        """Save results with organized structure"""
        if not self.config.output_path:
            return
        
        # Create results directory structure
        results_dir = Path(self.config.output_path).parent
        run_dir = results_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main result JSON
        result_json_path = run_dir / "benchmark_results.json"
        self.result.to_json(str(result_json_path))
        logger.info(f"Benchmark results saved to: {result_json_path}")
        
        # Save configuration YAML
        config_yaml_path = run_dir / "config.yaml"
        self.config.to_yaml(str(config_yaml_path))
        logger.info(f"Configuration saved to: {config_yaml_path}")
        
        # Export to other formats if enabled
        if self.config.export.enable_export:
            exporter = ResultExporter(self.config)
            export_paths = exporter.export_results(
                self.result.to_dict(), 
                self.run_id, 
                run_dir
            )
            
            if export_paths:
                logger.info(f"Exported results to: {list(export_paths.keys())}")
        
        # Update main output path to point to run directory
        self.config.output_path = str(run_dir / "benchmark_results.json")

    async def _cleanup(self):
        """Cleanup experiment resources"""
        # Stop server manager (this kills the vLLM process)
        if self.server_manager:
            try:
                await self.server_manager.stop()
                logger.info("Server manager stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping server manager: {e}")
                # Force kill any remaining vLLM processes
                await self._force_kill_vllm_processes()
        
        # Stop system metrics
        if self.system_metrics:
            try:
                await self.system_metrics.stop_monitoring()
                logger.info("System metrics stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping system metrics: {e}")
    
    async def _force_kill_vllm_processes(self):
        """Force kill any remaining vLLM processes"""
        import subprocess
        import signal
        
        try:
            # Find and kill any vLLM processes
            result = subprocess.run(
                ['pkill', '-f', 'vllm.entrypoints.openai.api_server'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("Force killed remaining vLLM processes")
            else:
                logger.info("No remaining vLLM processes found")
        except Exception as e:
            logger.error(f"Error force killing vLLM processes: {e}")
