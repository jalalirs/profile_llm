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
    
    async def run(self) -> ExperimentResult:
        """Execute the complete experiment"""
        logger.info(f"Starting experiment {self.run_id}")
        start_time = time.time()
        
        try:
            # Validate configuration
            await self._validate_config()
            
            # Initialize components
            await self._initialize_components()
            
            # Start system monitoring
            if self.system_metrics:
                await self.system_metrics.start_monitoring()
            
            # Start vLLM server
            await self.server_manager.start()
            self.result.status = "server_started"
            
            # Wait for server to be ready
            await self.server_manager.wait_for_ready()
            
            # Run benchmark
            benchmark_results = await self.benchmark_client.run()
            self.result.results.update(benchmark_results)
            self.result.status = "benchmark_completed"
            
            # Collect system metrics
            if self.system_metrics:
                system_data = await self.system_metrics.stop_monitoring()
                self.result.system.update(system_data)
            
            # Collect profiling data
            if self.config.system.enable_profiling:
                profiling_data = await self._collect_profiling_data()
                self.result.profiling.update(profiling_data)
            
            self.result.status = "completed"
            logger.info(f"Experiment {self.run_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Experiment {self.run_id} failed: {str(e)}")
            self.result.status = "failed"
            self.result.error = str(e)
            raise
        
        finally:
            # Cleanup
            await self._cleanup()
            
            # Calculate duration
            self.result.duration = int(time.time() - start_time)
            
            # Save results if configured
            if self.config.output_path:
                output_path = Path(self.config.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.result.to_json(str(output_path))
        
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
        # Server manager
        self.server_manager = VLLMServerManager(self.config.server)
        
        # System metrics
        if self.config.system.monitor_gpu or self.config.system.monitor_cpu:
            self.system_metrics = SystemMetrics(self.config.system)
        
        # Benchmark client (will be initialized after server starts)
        self.benchmark_client = BenchmarkClient(
            self.config.benchmark,
            server_config=self.config.server
        )
    
    async def _collect_profiling_data(self) -> Dict[str, Any]:
        """Collect profiling data"""
        profiling_data = {"enabled": True}
        
        if self.config.system.profile_dir:
            profile_path = Path(self.config.system.profile_dir)
            trace_files = list(profile_path.glob(f"*{self.run_id}*.perfetto.gz"))
            
            if trace_files:
                trace_file = trace_files[0]
                profiling_data.update({
                    "trace_file_path": str(trace_file),
                    "trace_file_size_mb": trace_file.stat().st_size / (1024 * 1024)
                })
        
        return profiling_data
    
    async def _cleanup(self):
        """Cleanup experiment resources"""
        if self.server_manager:
            await self.server_manager.stop()
        
        if self.system_metrics:
            await self.system_metrics.stop_monitoring()
