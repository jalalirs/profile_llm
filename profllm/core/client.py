"""Benchmark client implementation"""

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional

from ..config.models import BenchmarkConfig, ServerConfig

logger = logging.getLogger(__name__)

class BenchmarkClient:
    """Client for running benchmarks against vLLM server"""
    
    def __init__(self, benchmark_config: BenchmarkConfig, server_config: ServerConfig, system_config: Optional['SystemConfig'] = None):
        self.benchmark_config = benchmark_config
        self.server_config = server_config
        self.system_config = system_config
        self.server_url: Optional[str] = None
    
    async def run(self) -> Dict[str, Any]:
        """Run benchmark and return results"""
        logger.info("Starting benchmark execution...")
        
        # Determine server URL
        self.server_url = f"http://{self.server_config.host}:{self.server_config.port}"
        
        # Prepare benchmark arguments
        benchmark_args = self._build_benchmark_args()
        
        # Create temporary file for results
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
            result_file = temp_file.name
        
        try:
            # Add result file to arguments
            benchmark_args.extend(['--save-result'])
            benchmark_args.extend(['--result-filename', result_file])
            
            # Run benchmark_serving.py
            cmd = ['python', '-m', 'vllm.entrypoints.benchmarks.benchmark_serving'] + benchmark_args
            
            logger.info(f"Running benchmark with command: {' '.join(cmd)}")
            
            # Set up environment for profiling if enabled
            env = None
            if self.benchmark_config.profile:
                import os
                env = os.environ.copy()
                if 'VLLM_TORCH_PROFILER_DIR' not in env:
                    env['VLLM_TORCH_PROFILER_DIR'] = '/tmp/profllm_traces'
            
            # Execute benchmark
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=3600  # 1 hour timeout
            )
            
            # Prepare dataset
            input_requests = self._prepare_dataset(tokenizer)
            
            # Debug: Log execution context
            import os
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Dataset size: {len(input_requests)}")
            logger.info(f"Max concurrency: {self.benchmark_config.max_concurrency}")
            logger.info(f"Request rate: {self.benchmark_config.request_rate}")
            logger.info(f"Server URL: {self.server_url}")
            
            # Run benchmark using imported function
            result = await benchmark(
                backend="vllm",
                api_url=f"{self.server_url}/v1/completions",
                base_url=self.server_url,
                model_id=self.server_config.model,
                model_name=self.server_config.model,
                tokenizer=tokenizer,
                input_requests=input_requests,
                logprobs=None,
                request_rate=self.benchmark_config.request_rate,
                burstiness=1.0,  # Default burstiness
                disable_tqdm=self.benchmark_config.disable_tqdm,
                profile=self.benchmark_config.profile and not (self.system_config and self.system_config.enable_nsight),
                selected_percentile_metrics=["ttft", "tpot", "itl"],
                selected_percentiles=[50.0, 95.0, 99.0],
                ignore_eos=self.benchmark_config.ignore_eos,
                goodput_config_dict={},
                max_concurrency=self.benchmark_config.max_concurrency,
                lora_modules=None,
                extra_body=self._get_sampling_params(),
                ramp_up_strategy=None,
                ramp_up_start_rps=None,
                ramp_up_end_rps=None,
            )
            
            # Parse results
            results = self._parse_results(result_file)
            logger.info("Benchmark completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {str(e)}")
            raise
        finally:
            # Clean up temp file
            try:
                Path(result_file).unlink()
            except:
                pass
    
    def _build_benchmark_args(self) -> list[str]:
        """Build command line arguments for benchmark_serving.py"""
        args = []
        
        # Basic arguments
        args.extend(['--backend', 'vllm'])
        args.extend(['--base-url', self.server_url])
        args.extend(['--endpoint', '/v1/completions'])
        
        # Dataset configuration
        if self.benchmark_config.dataset_name:
            args.extend(['--dataset-name', self.benchmark_config.dataset_name])
        
        if self.benchmark_config.dataset_path:
            args.extend(['--dataset-path', self.benchmark_config.dataset_path])
        
        args.extend(['--num-prompts', str(self.benchmark_config.num_prompts)])
        
        # Request pattern
        if isinstance(self.benchmark_config.request_rate, str):
            args.extend(['--request-rate', self.benchmark_config.request_rate])
        else:
            args.extend(['--request-rate', str(self.benchmark_config.request_rate)])
        
        if self.benchmark_config.max_concurrency:
            args.extend(['--max-concurrency', str(self.benchmark_config.max_concurrency)])
        
        # Generation parameters
        args.extend(['--max-tokens', str(self.benchmark_config.max_tokens)])
        args.extend(['--temperature', str(self.benchmark_config.temperature)])
        args.extend(['--top-p', str(self.benchmark_config.top_p)])
        args.extend(['--top-k', str(self.benchmark_config.top_k)])
        
        # Other options
        args.extend(['--seed', str(self.benchmark_config.seed)])
        
        if self.benchmark_config.ignore_eos:
            args.append('--ignore-eos')
        
        if self.benchmark_config.disable_tqdm:
            args.append('--disable-tqdm')
        
        if self.benchmark_config.profile:
            args.append('--profile')
        
        if self.benchmark_config.trust_remote_code:
            args.append('--trust-remote-code')
        
        # Tokenizer
        if self.benchmark_config.tokenizer:
            args.extend(['--tokenizer', self.benchmark_config.tokenizer])
        
        # Sampling parameters
        args.extend(['--best-of', str(self.benchmark_config.best_of)])
        
        if self.benchmark_config.use_beam_search:
            args.append('--use-beam-search')
        
        return args
    
    def _parse_results(self, result_file: str) -> Dict[str, Any]:
        """Parse benchmark results from JSON file"""
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics
            results = {
                "request_throughput": data.get("request_throughput", 0.0),
                "output_throughput": data.get("output_throughput", 0.0), 
                "median_ttft_ms": data.get("median_ttft_ms", 0.0),
                "p95_ttft_ms": data.get("p95_ttft_ms", 0.0),
                "median_tpot_ms": data.get("median_tpot_ms", 0.0),
                "p95_tpot_ms": data.get("p95_tpot_ms", 0.0),
                "completed_requests": data.get("completed_requests", 0),
                "failed_requests": data.get("failed_requests", 0),
            }
            
            # Add raw data for completeness
            results["raw_data"] = data
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to parse benchmark results: {str(e)}")
            return {
                "error": f"Failed to parse results: {str(e)}",
                "completed_requests": 0,
                "failed_requests": 0
            }

# Client management
