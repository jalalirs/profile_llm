"""Benchmark client implementation"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from ..config.models import BenchmarkConfig, ServerConfig

# Import benchmark functionality directly
from .benchmark_serving import benchmark
from .benchmark_dataset import ShareGPTDataset, RandomDataset
from .backend_request_func import get_tokenizer

logger = logging.getLogger(__name__)

class BenchmarkClient:
    """Client for running benchmarks against vLLM server"""
    
    def __init__(self, benchmark_config: BenchmarkConfig, server_config: ServerConfig, system_config: Optional['SystemConfig'] = None):
        self.benchmark_config = benchmark_config
        self.server_config = server_config
        self.system_config = system_config
        self.server_url: Optional[str] = None
    
    def update_server_info(self, host: str, port: int):
        """Update server connection info with actual values"""
        self.server_config.host = host
        self.server_config.port = port
        logger.info(f"Updated server info: {host}:{port}")
    
    async def run(self) -> Dict[str, Any]:
        """Run benchmark and return results"""
        logger.info("Starting benchmark execution...")
        
        # Use provided server info (no waiting needed in client mode)
        self.server_url = f"http://{self.server_config.host}:{self.server_config.port}"
        
        try:
            # Get tokenizer
            tokenizer = get_tokenizer(
                self.server_config.model,
                tokenizer_mode="auto",
                trust_remote_code=self.benchmark_config.trust_remote_code
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
                request_rate=float(self.benchmark_config.request_rate) if self.benchmark_config.request_rate != "inf" else float('inf'),
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
            
            logger.info("Benchmark completed successfully")
            return self._format_results(result)
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {str(e)}")
            raise
    
    def _prepare_dataset(self, tokenizer) -> list:
        """Prepare dataset for benchmarking"""
        if self.benchmark_config.dataset_name == "sharegpt":
            # Auto-download ShareGPT dataset if no path provided
            if not self.benchmark_config.dataset_path:
                self.benchmark_config.dataset_path = self._download_sharegpt_dataset()
            
            # Ensure consistent prompts by using a fixed seed
            # Use the configured seed, but default to 42 for consistency
            fixed_seed = getattr(self.benchmark_config, 'seed', 42) or 42
            logger.info(f"Using seed {fixed_seed} for consistent prompt generation")
            dataset = ShareGPTDataset(
                random_seed=fixed_seed,
                dataset_path=self.benchmark_config.dataset_path
            )
            requests = dataset.sample(
                tokenizer=tokenizer,
                num_requests=self.benchmark_config.num_prompts,
                output_len=getattr(self.benchmark_config, 'sharegpt_output_len', None),
                request_id_prefix="profllm-benchmark"
            )
            
            # Log prompt information
            self._log_prompt_info(requests)
            return requests
        elif self.benchmark_config.dataset_name == "random":
            # Ensure consistent random prompts by using a fixed seed
            # Use the configured seed, but default to 42 for consistency
            fixed_seed = getattr(self.benchmark_config, 'seed', 42) or 42
            logger.info(f"Using seed {fixed_seed} for consistent prompt generation")
            dataset = RandomDataset(
                dataset_path=self.benchmark_config.dataset_path,
                random_seed=fixed_seed
            )
            requests = dataset.sample(
                tokenizer=tokenizer,
                num_requests=self.benchmark_config.num_prompts,
                prefix_len=0,
                input_len=getattr(self.benchmark_config, 'random_input_len', 1024),
                output_len=getattr(self.benchmark_config, 'random_output_len', 128),
                range_ratio=0.0,
                request_id_prefix="profllm-benchmark"
            )
            
            # Log prompt information
            self._log_prompt_info(requests)
            return requests
        else:
            raise ValueError(f"Unsupported dataset: {self.benchmark_config.dataset_name}")
    
    def _download_sharegpt_dataset(self) -> str:
        """Download ShareGPT dataset and return the local path"""
        import os
        import requests
        from pathlib import Path
        
        # Create datasets directory
        datasets_dir = Path.home() / '.cache' / 'profllm' / 'datasets'
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # ShareGPT dataset file path
        dataset_file = datasets_dir / 'sharegpt.json'
        
        # Check if already downloaded
        if dataset_file.exists():
            logger.info(f"ShareGPT dataset already exists at {dataset_file}")
            return str(dataset_file)
        
        # Download ShareGPT dataset
        logger.info("Downloading ShareGPT dataset...")
        url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(dataset_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"ShareGPT dataset downloaded to {dataset_file}")
            return str(dataset_file)
            
        except Exception as e:
            logger.error(f"Failed to download ShareGPT dataset: {e}")
            # Fallback to a minimal dataset
            fallback_data = [
                {
                    "id": "sample_1",
                    "conversations": [
                        {"from": "human", "value": "Hello, how are you?"},
                        {"from": "gpt", "value": "I'm doing well, thank you for asking!"}
                    ]
                },
                {
                    "id": "sample_2", 
                    "conversations": [
                        {"from": "human", "value": "What is machine learning?"},
                        {"from": "gpt", "value": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."}
                    ]
                }
            ]
            
            import json
            with open(dataset_file, 'w') as f:
                json.dump(fallback_data, f)
            
            logger.info(f"Created fallback ShareGPT dataset at {dataset_file}")
            return str(dataset_file)
    
    def _log_prompt_info(self, requests: list):
        """Log information about the prompts being used"""
        if not requests:
            logger.warning("No requests generated from dataset")
            return
        
        # Calculate prompt length statistics
        prompt_lengths = [req.prompt_len for req in requests]
        output_lengths = [req.expected_output_len for req in requests]
        
        min_prompt_len = min(prompt_lengths)
        max_prompt_len = max(prompt_lengths)
        avg_prompt_len = sum(prompt_lengths) / len(prompt_lengths)
        
        min_output_len = min(output_lengths)
        max_output_len = max(output_lengths)
        avg_output_len = sum(output_lengths) / len(output_lengths)
        
        logger.info("=" * 60)
        logger.info("PROMPT INFORMATION")
        logger.info("=" * 60)
        logger.info(f"Total number of prompts: {len(requests)}")
        logger.info(f"Prompt lengths - Min: {min_prompt_len}, Max: {max_prompt_len}, Avg: {avg_prompt_len:.1f}")
        logger.info(f"Output lengths - Min: {min_output_len}, Max: {max_output_len}, Avg: {avg_output_len:.1f}")
        logger.info("=" * 60)
        
        # Log first few prompts as examples
        logger.info("SAMPLE PROMPTS:")
        for i, req in enumerate(requests[:3]):  # Show first 3 prompts
            prompt_preview = str(req.prompt)[:100] + "..." if len(str(req.prompt)) > 100 else str(req.prompt)
            logger.info(f"  Prompt {i+1} (ID: {req.request_id}):")
            logger.info(f"    Length: {req.prompt_len} tokens")
            logger.info(f"    Expected output: {req.expected_output_len} tokens")
            logger.info(f"    Preview: {prompt_preview}")
            logger.info("")
        
        if len(requests) > 3:
            logger.info(f"  ... and {len(requests) - 3} more prompts")
        
        logger.info("=" * 60)
    
    def _get_sampling_params(self) -> dict:
        """Get sampling parameters for the benchmark"""
        params = {}
        
        if hasattr(self.benchmark_config, 'temperature'):
            params['temperature'] = self.benchmark_config.temperature
        
        if hasattr(self.benchmark_config, 'top_p'):
            params['top_p'] = self.benchmark_config.top_p
        
        if hasattr(self.benchmark_config, 'top_k'):
            params['top_k'] = self.benchmark_config.top_k
        
        if hasattr(self.benchmark_config, 'max_tokens'):
            params['max_tokens'] = self.benchmark_config.max_tokens
        
        if hasattr(self.benchmark_config, 'min_p'):
            params['min_p'] = self.benchmark_config.min_p
        
        # Set default temperature if not specified
        if 'temperature' not in params:
            params['temperature'] = 0.0
        
        return params
    
    def _format_results(self, benchmark_result: dict) -> Dict[str, Any]:
        """Format benchmark results to match expected output"""
        return {
            "request_throughput": benchmark_result.get("request_throughput", 0.0),
            "output_throughput": benchmark_result.get("output_throughput", 0.0),
            "median_ttft_ms": benchmark_result.get("median_ttft_ms", 0.0),
            "p95_ttft_ms": benchmark_result.get("p95_ttft_ms", 0.0),
            "median_tpot_ms": benchmark_result.get("median_tpot_ms", 0.0),
            "p95_tpot_ms": benchmark_result.get("p95_tpot_ms", 0.0),
            "completed_requests": benchmark_result.get("completed", 0),
            "failed_requests": len(benchmark_result.get("errors", [])) if benchmark_result.get("errors") else 0,
            "duration": benchmark_result.get("duration", 0.0),
            "total_input_tokens": benchmark_result.get("total_input_tokens", 0),
            "total_output_tokens": benchmark_result.get("total_output_tokens", 0),
            "raw_data": benchmark_result
        }

# Client management
