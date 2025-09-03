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
    
    # Check prefill optimization parameters
    # if config.server.enable_chunked_prefill:
    #     if not config.server.max_num_partial_prefills or config.server.max_num_partial_prefills < 1:
    #         warnings.append("enable_chunked_prefill requires max_num_partial_prefills >= 1")
    #     if not config.server.long_prefill_token_threshold or config.server.long_prefill_token_threshold < 1:
    #         warnings.append("enable_chunked_prefill requires long_prefill_token_threshold >= 1")
    
    # Check scheduling policy compatibility
    if config.server.scheduling_policy == "priority" and not config.server.async_scheduling:
        warnings.append("Priority scheduling policy works best with async_scheduling enabled")
    
    # Check compilation configuration
    if config.server.compilation_config and config.server.compilation_config.get("use_cudagraph"):
        if config.server.compilation_config.get("cudagraph_num_of_warmups", 0) < 3:
            warnings.append("CUDA graphs may need more warmup iterations for stable performance")
    
    # Check goodput SLO definitions
    if config.benchmark.goodput:
        for slo in config.benchmark.goodput:
            if ":" not in slo:
                warnings.append(f"Invalid goodput SLO format: {slo}. Expected format: 'metric:threshold'")
    
    # Check percentile metrics
    if config.benchmark.percentile_metrics:
        valid_metrics = {"ttft", "tpot", "itl", "e2el", "throughput"}
        provided_metrics = set(config.benchmark.percentile_metrics.split(","))
        invalid_metrics = provided_metrics - valid_metrics
        if invalid_metrics:
            warnings.append(f"Unknown percentile metrics: {invalid_metrics}. Valid metrics: {valid_metrics}")
    
    return warnings

def get_vllm_server_args(config: ServerConfig) -> Dict[str, Any]:
    """Convert ServerConfig to vLLM server arguments"""
    args = {}
    
    # Debug: Log the config values
    print(f"DEBUG: ServerConfig tensor_parallel_size = {config.tensor_parallel_size}")
    print(f"DEBUG: ServerConfig type = {type(config.tensor_parallel_size)}")
    
    # Only allow arguments that are actually valid vLLM arguments
    # Based on the actual vLLM help output
    valid_vllm_args = {
        # Basic server args
        'model', 'runner', 'convert', 'task', 'tokenizer', 'tokenizer_mode', 
        'trust_remote_code', 'dtype', 'seed', 'hf_config_path', 'allowed_local_media_path',
        'revision', 'code_revision', 'rope_scaling', 'rope_theta', 'tokenizer_revision',
        'max_model_len', 'quantization', 'enforce_eager',
        'max_logprobs', 'logprobs_mode', 'disable_sliding_window', 'disable_cascade_attn',
        'skip_tokenizer_init', 'enable_prompt_embeds', 'served_model_name',
        'disable_async_output_proc', 'config_format', 'hf_token', 'hf_overrides',
        'override_neuron_config', 'override_pooler_config', 'logits_processor_pattern',
        'generation_config', 'override_generation_config', 'enable_sleep_mode',
        'model_impl', 'override_attention_dtype', 'logits_processors', 'load_format',
        'download_dir', 'model_loader_extra_config', 'ignore_patterns', 'use_tqdm_on_load',
        'pt_load_map_location', 'guided_decoding_backend', 'guided_decoding_disable_fallback',
        'guided_decoding_disable_any_whitespace', 'guided_decoding_disable_additional_properties',
        'reasoning_parser', 'distributed_executor_backend', 'pipeline_parallel_size',
        'tensor_parallel_size', 'data_parallel_size', 'data_parallel_rank',
        'data_parallel_start_rank', 'data_parallel_size_local', 'data_parallel_address',
        'data_parallel_rpc_port', 'data_parallel_backend', 'data_parallel_hybrid_lb',
        'enable_expert_parallel', 'enable_eplb', 'num_redundant_experts',
        'eplb_window_size', 'eplb_step_interval', 'eplb_log_balancedness',
        'max_parallel_loading_workers', 'ray_workers_use_nsight', 'disable_custom_all_reduce',
        'worker_cls', 'worker_extension_cls', 'enable_multimodal_encoder_data_parallel',
        'block_size', 'gpu_memory_utilization', 'swap_space', 'kv_cache_dtype',
        'num_gpu_blocks_override', 'enable_prefix_caching', 'prefix_caching_hash_algo',
        'cpu_offload_gb', 'calculate_kv_scales', 'kv_sharing_fast_prefill',
        'mamba_cache_dtype', 'mamba_ssm_cache_dtype', 'limit_mm_per_prompt',
        'media_io_kwargs', 'mm_processor_kwargs', 'mm_processor_cache_gb',
        'disable_mm_preprocessor_cache', 'interleave_mm_strings', 'skip_mm_profiling',
        'enable_lora', 'enable_lora_bias', 'max_loras', 'max_lora_rank',
        'lora_extra_vocab_size', 'lora_dtype', 'max_cpu_loras', 'fully_sharded_loras',
        'default_mm_loras', 'show_hidden_metrics_for_version', 'otlp_traces_endpoint',
        'collect_detailed_traces', 'max_num_batched_tokens', 'max_num_seqs',
        'cuda_graph_sizes', 'num_lookahead_slots', 'scheduler_delay_factor',
        #'preemption_mode', 'scheduling_policy', 'disable_chunked_mm_input',
        'scheduling_policy', 'disable_chunked_mm_input',
        'scheduler_cls', 'disable_hybrid_kv_cache_manager', 'async_scheduling',
        #'enable_chunked_prefill', 'max_num_partial_prefills', 'long_prefill_token_threshold',
        'max_seq_len_to_capture', 'speculative_config', 'kv_transfer_config', 'kv_events_config',
        'compilation_config', 'additional_config', 'disable_log_stats', 'enable_prompt_adapter',
        'enable_log_requests', 'disable_log_requests'
    }
    
    # Get all fields from the config
    # Handle both Pydantic V1 and V2 field access
    if hasattr(config, '__fields__'):
        # Pydantic V1
        fields = config.__fields__
    elif hasattr(config, 'model_fields'):
        # Pydantic V2
        fields = config.model_fields
    else:
        # Fallback: use dir() to get all attributes
        fields = {name: None for name in dir(config) if not name.startswith('_')}
    
    for field_name in fields:
        if field_name in ['host', 'port', 'api_key']:
            continue
            
        try:
            value = getattr(config, field_name)
        except AttributeError:
            continue
        
        # Skip internal fields and invalid vLLM arguments
        if field_name not in valid_vllm_args:
            continue
            
        # Skip None values and empty strings
        if value is None or (isinstance(value, str) and value == ""):
            continue
            
        # Convert underscores to dashes for CLI compatibility
        cli_arg = field_name.replace('_', '-')
        
        # Handle deprecated arguments
        if cli_arg == 'disable-log-requests':
            cli_arg = 'enable-log-requests'
            value = False  # Invert the logic
        
        # Only add simple values that vLLM can handle
        if isinstance(value, (bool, int, float, str)):
            args[cli_arg] = value
        # For lists, only add if they contain simple values
        elif isinstance(value, list):
            if all(isinstance(v, (bool, int, float, str)) for v in value):
                args[cli_arg] = value
    
    return args