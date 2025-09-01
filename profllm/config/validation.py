# Configuration validation
"""Configuration validation utilities"""

from typing import List, Dict, Any
from ..config.models import ExperimentConfig, ServerConfig

def validate_config(config: ExperimentConfig) -> List[str]:
    """Validate experiment configuration and return list of warnings/errors"""
    warnings = []
    
    # Check tensor parallel size vs available GPUs
    if config.server.tensor_parallel_size > 1:
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            gpu_count = nvml.nvmlDeviceGetCount()
            if config.server.tensor_parallel_size > gpu_count:
                warnings.append(f"tensor_parallel_size ({config.server.tensor_parallel_size}) > available GPUs ({gpu_count})")
        except:
            warnings.append("Could not detect GPU count for tensor_parallel_size validation")
    
    # Check memory utilization
    if config.server.gpu_memory_utilization > 0.95:
        warnings.append(f"High GPU memory utilization ({config.server.gpu_memory_utilization}) may cause OOM")
    
    # Check quantization compatibility
    if config.server.quantization and config.server.dtype not in ["auto", "half", "float16"]:
        warnings.append(f"Quantization {config.server.quantization} may not be compatible with dtype {config.server.dtype}")
    
    # Validate model and dataset compatibility
    if "vision" in config.server.model.lower() or "multimodal" in config.server.model.lower():
        if not config.benchmark.dataset_name in ["llava", "multimodal"]:
            warnings.append("Vision/multimodal model detected but dataset may not support multimodal inputs")
    
    # Check scheduling parameters
    if config.server.max_num_seqs and config.server.max_num_batched_tokens:
        if config.server.max_num_batched_tokens < config.server.max_num_seqs * 100:
            warnings.append("max_num_batched_tokens may be too low relative to max_num_seqs")
    
    return warnings

def get_vllm_server_args(config: ServerConfig) -> Dict[str, Any]:
    """Convert ServerConfig to vLLM server arguments"""
    args = {}
    
    # Get all fields from the config
    for field_name, field in config.__fields__.items():
        value = getattr(config, field_name)
        
        # Skip None values and internal fields
        if value is None or field_name in ['host', 'port', 'api_key']:
            continue
            
        # Handle aliases (e.g., tp -> tensor_parallel_size)
        arg_name = field.alias or field_name
        
        # Convert underscores to dashes for CLI compatibility
        cli_arg = arg_name.replace('_', '-')
        
        args[cli_arg] = value
    
    return args