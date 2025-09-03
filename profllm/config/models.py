# Configuration models
from typing import Optional, Dict, Any, List, Union, Literal
from pydantic import BaseModel, Field, validator
import json

class ServerConfig(BaseModel):
    """Configuration for vLLM server parameters - covers all 150+ vLLM engine args"""
    
    # Run identification for profiling
    run_id: Optional[str] = None
    experiment_profile_dir: Optional[str] = None
    
    # ModelConfig
    model: str = Field(default="meta-llama/Meta-Llama-3-8B", description="Model name or path")
    runner: Optional[Literal["auto", "draft", "generate", "pooling"]] = Field(default="auto")
    convert: Optional[Literal["auto", "classify", "embed", "none", "reward"]] = Field(default="auto")
    task: Optional[Literal["auto", "classify", "draft", "embed", "embedding", "generate", "reward", "score", "transcription"]] = None
    tokenizer: Optional[str] = None
    tokenizer_mode: Literal["auto", "custom", "mistral", "slow"] = "auto"
    trust_remote_code: bool = False
    dtype: Literal["auto", "bfloat16", "float", "float16", "float32", "half"] = "auto"
    seed: Optional[int] = None
    hf_config_path: Optional[str] = None
    allowed_local_media_path: str = ""
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[Dict[str, Any]] = Field(default_factory=dict)
    rope_theta: Optional[float] = None
    tokenizer_revision: Optional[str] = None
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_logprobs: int = 20
    logprobs_mode: str = "raw_logprobs"
    disable_sliding_window: bool = False
    disable_cascade_attn: bool = False
    skip_tokenizer_init: bool = False
    enable_prompt_embeds: bool = False
    served_model_name: Optional[List[str]] = None
    disable_async_output_proc: bool = False
    config_format: Literal["auto", "hf", "mistral"] = "auto"
    hf_token: Optional[Union[str, bool]] = None
    hf_overrides: Optional[Dict[str, Any]] = Field(default_factory=dict)
    override_neuron_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    override_pooler_config: Optional[Dict[str, Any]] = None
    logits_processor_pattern: Optional[str] = None
    generation_config: str = "auto"
    override_generation_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    enable_sleep_mode: bool = False
    model_impl: Literal["auto", "vllm", "transformers"] = "auto"
    override_attention_dtype: Optional[str] = None
    logits_processors: Optional[List[str]] = None
    
    # LoadConfig
    load_format: Literal["auto", "pt", "safetensors", "npcache", "dummy", "tensorizer", "runai_streamer", "bitsandbytes", "sharded_state", "gguf", "mistral"] = "auto"
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    ignore_patterns: Optional[List[str]] = None
    use_tqdm_on_load: bool = True
    pt_load_map_location: str = "cpu"
    
    # DecodingConfig
    guided_decoding_backend: Literal["auto", "guidance", "lm-format-enforcer", "outlines", "xgrammar"] = "auto"
    guided_decoding_disable_fallback: bool = False
    guided_decoding_disable_any_whitespace: bool = False
    guided_decoding_disable_additional_properties: bool = False
    reasoning_parser: str = ""
    
    # ParallelConfig
    distributed_executor_backend: Optional[Literal["external_launcher", "mp", "ray", "uni"]] = None
    pipeline_parallel_size: int = Field(default=1)
    tensor_parallel_size: int
    data_parallel_size: int = Field(default=1)
    data_parallel_rank: Optional[int] = Field(default=None)
    data_parallel_start_rank: Optional[int] = Field(default=None)
    data_parallel_size_local: Optional[int] = Field(default=None)
    data_parallel_address: Optional[str] = Field(default=None)
    data_parallel_rpc_port: Optional[int] = Field(default=None)
    data_parallel_backend: Literal["mp", "ray"] = Field(default="mp")
    data_parallel_hybrid_lb: bool = False
    enable_expert_parallel: bool = False
    enable_eplb: bool = False
    eplb_config: Optional[Dict[str, Any]] = None
    max_parallel_loading_workers: Optional[int] = None
    ray_workers_use_nsight: bool = False
    disable_custom_all_reduce: bool = False
    worker_cls: str = "auto"
    worker_extension_cls: str = ""
    enable_multimodal_encoder_data_parallel: bool = False
    
    # CacheConfig
    block_size: Optional[Literal[1, 8, 16, 32, 64, 128]] = None
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"] = "auto"
    num_gpu_blocks_override: Optional[int] = None
    enable_prefix_caching: Optional[bool] = None
    prefix_caching_hash_algo: Literal["builtin", "sha256", "sha256_cbor_64bit"] = "builtin"
    cpu_offload_gb: int = 0
    calculate_kv_scales: bool = False
    kv_sharing_fast_prefill: bool = False
    mamba_cache_dtype: Literal["auto", "float32"] = "auto"
    mamba_ssm_cache_dtype: Literal["auto", "float32"] = "auto"
    
    # MultiModalConfig
    limit_mm_per_prompt: Optional[Dict[str, int]] = Field(default_factory=dict)
    media_io_kwargs: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict)
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    mm_processor_cache_gb: int = 4
    disable_mm_preprocessor_cache: bool = False
    mm_encoder_tp_mode: Literal["data", "weights"] = "weights"
    interleave_mm_strings: bool = False
    skip_mm_profiling: bool = False
    
    # LoRAConfig
    enable_lora: Optional[bool] = None
    enable_lora_bias: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    lora_extra_vocab_size: int = 256
    lora_dtype: Literal["auto", "bfloat16", "float16"] = "auto"
    max_cpu_loras: Optional[int] = None
    fully_sharded_loras: bool = False
    default_mm_loras: Optional[Dict[str, str]] = Field(default_factory=dict)
    
    # ObservabilityConfig
    show_hidden_metrics_for_version: Optional[str] = None
    otlp_traces_endpoint: Optional[str] = None
    collect_detailed_traces: Optional[str] = None
    
    # SchedulerConfig
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
    cuda_graph_sizes: List[int] = Field(default_factory=list)
    num_lookahead_slots: int = 0
    scheduler_delay_factor: float = 0.0
    #preemption_mode: Optional[Literal["recompute", "swap"]] = None
    scheduling_policy: Literal["fcfs", "priority"] = "fcfs"
    disable_chunked_mm_input: bool = False
    scheduler_cls: str = "vllm.core.scheduler.Scheduler"
    disable_hybrid_kv_cache_manager: bool = False
    async_scheduling: bool = False
    
    # Prefill optimizations
    #enable_chunked_prefill: bool = False
    #max_num_partial_prefills: int = 4
    #long_prefill_token_threshold: int = 1024
    
    # Performance features
    max_seq_len_to_capture: Optional[int] = None
    
    # VllmConfig
    speculative_config: Optional[Dict[str, Any]] = None
    kv_transfer_config: Optional[Dict[str, Any]] = None
    kv_events_config: Optional[Dict[str, Any]] = None
    compilation_config: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "level": None,
        "debug_dump_path": "",
        "cache_dir": "",
        "backend": "",
        "custom_ops": [],
        "splitting_ops": None,
        "use_inductor": True,
        "compile_sizes": None,
        "inductor_compile_config": {"enable_auto_functionalized_v2": False},
        "inductor_passes": {},
        "cudagraph_mode": None,
        "use_cudagraph": True,
        "cudagraph_num_of_warmups": 0,
        "cudagraph_capture_sizes": None,
        "cudagraph_copy_inputs": False,
        "full_cuda_graph": False,
        "pass_config": {},
        "max_capture_size": None,
        "local_cache_dir": None
    })
    additional_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # AsyncEngineArgs
    enable_log_requests: bool = False
    disable_log_requests: bool = True
    
    # Server specific
    host: str = "127.0.0.1"
    port: Optional[int] = None  # Will be auto-assigned if None
    api_key: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True
        extra = "allow"  # Allow extra fields for forward compatibility
    
    @validator('port')
    def validate_port(cls, v):
        if v is not None and (v < 1024 or v > 65535):
            raise ValueError('Port must be between 1024 and 65535')
        return v

class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution"""
    
    # Backend and connection
    backend: str = "vllm"
    endpoint: str = "/v1/completions"
    
    # Dataset configuration
    dataset_name: str = "sharegpt"
    dataset_path: Optional[str] = None
    num_prompts: int = 1000
    sharegpt_output_len: int = 256  # Override default output length
    
    # Request patterns
    request_rate: Union[float, str] = "inf"  # requests per second or "inf" for unlimited
    burstiness: float = 1.0  # Traffic burstiness factor
    max_concurrency: Optional[int] = None
    
    # Benchmark behavior
    seed: int = 0
    save_result: bool = True
    save_detailed: bool = False  # Save detailed request-level results
    trust_remote_code: bool = False
    disable_tqdm: bool = False
    profile: bool = False
    
    # Output configuration
    result_dir: Optional[str] = None
    result_filename: Optional[str] = None
    
    # Request sampling
    tokenizer: Optional[str] = None
    best_of: int = 1
    use_beam_search: bool = False
    
    # Generation parameters
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    ignore_eos: bool = False
    
    # Metrics and analysis
    percentile_metrics: str = "ttft,tpot"  # Comma-separated metrics for percentile analysis
    metric_percentiles: str = "50,90,95,99"  # Comma-separated percentiles
    goodput: Optional[List[str]] = None  # List of SLO definitions like ["ttft:200", "tpot:50"]
    
    # Result management
    metadata: Optional[List[str]] = None  # List of metadata key-value pairs
    
    # Behavior
    request_id_prefix: Optional[str] = None  # Prefix for request IDs
    
    # Additional benchmark args
    extra_body: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('request_rate')
    def validate_request_rate(cls, v):
        if isinstance(v, str):
            if v != "inf":
                raise ValueError('request_rate as string must be "inf"')
        elif isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError('request_rate must be positive')
        return v

class SystemConfig(BaseModel):
    """Configuration for system monitoring and profiling"""
    
    # Resource monitoring
    monitor_gpu: bool = True
    monitor_cpu: bool = True
    monitor_memory: bool = True
    monitoring_interval: float = 1.0  # seconds
    
    # Profiling
    enable_profiling: bool = False
    profile_dir: Optional[str] = None
    profile_activities: List[str] = Field(default_factory=lambda: ["cpu", "cuda"])
    profile_record_shapes: bool = True
    profile_profile_memory: bool = True
    profile_with_stack: bool = True
    
    # Nsight Systems profiling
    enable_nsight: bool = False
    nsight_output_dir: Optional[str] = None
    nsight_trace_fork: bool = True
    nsight_cuda_graph_trace: str = "node"
    nsight_delay: Optional[int] = None  # seconds to wait before starting profiling
    nsight_duration: Optional[int] = None  # seconds to profile (None = manual stop)
    
    # High verbosity Nsight options
    nsight_trace: str = "cuda,cudnn,mpi,osrt,nvtx,openmp"  # Enable comprehensive tracing: CUDA, cuDNN, MPI, OS runtime, NVTX, OpenMP
    nsight_cuda_trace_all_apis: bool = True  # Trace all CUDA APIs (high overhead but comprehensive)
    nsight_cuda_memory_usage: bool = True  # Track GPU memory usage by CUDA kernels
    nsight_cuda_backtrace: str = "all"  # Collect backtraces for all CUDA APIs
    nsight_cuda_flush_interval: int = 1000  # Flush CUDA buffers every 1 second
    nsight_sample: str = "process-tree"  # CPU sampling scope
    nsight_sampling_frequency: int = 8  # 8Hz sampling frequency (converted to period, results in 125000 period - minimum valid)
    nsight_cpu_core_events: str = "2"  # Instructions Retired (default)
    nsight_event_sample: str = "system-wide"  # Enable event sampling
    nsight_stats: bool = True  # Generate summary statistics
    nsight_export: str = "text"  # Export additional text format
    
    # Additional tracing options (these are now included in nsight_trace above)
    nsight_mpi_trace: bool = True  # Enable MPI tracing for collective operations
    nsight_osrt_trace: bool = True  # Enable OS runtime tracing for system calls
    
    # Advanced CPU tracing options
    # Note: CPU function tracing is achieved through osrt (OS runtime) and sampling
    nsight_cpu_trace_children: bool = True  # Trace child processes
    nsight_cpu_trace_fork: bool = True  # Trace fork/exec calls
    nsight_cpu_trace_syscalls: bool = True  # Trace system calls (via osrt)
    nsight_cpu_trace_locks: bool = True  # Trace mutex/lock operations (via osrt)
    nsight_cpu_trace_memory: bool = True  # Trace memory allocations/deallocations (via osrt)
    nsight_cpu_trace_io: bool = True  # Trace I/O operations (via osrt)
    nsight_cpu_trace_network: bool = True  # Trace network operations (via osrt)
    nsight_cpu_trace_threads: bool = True  # Trace thread creation/destruction (via osrt)
    nsight_cpu_trace_signals: bool = True  # Trace signal handling (via osrt)
    nsight_cpu_trace_timers: bool = True  # Trace timer operations (via osrt)
    
    # CPU sampling and profiling options
    nsight_cpu_sample_rate: int = 1000  # CPU sampling rate in Hz (higher = more overhead)
    nsight_cpu_sample_scope: str = "process-tree"  # process, process-tree, system
    nsight_cpu_sample_cpu: bool = True  # Sample CPU usage
    nsight_cpu_sample_memory: bool = True  # Sample memory usage
    nsight_cpu_sample_io: bool = True  # Sample I/O operations
    nsight_cpu_sample_network: bool = True  # Sample network operations
    
    # CPU performance counters
    nsight_cpu_counters: bool = True  # Enable CPU performance counters
    nsight_cpu_counter_events: str = "cycles,instructions,cache-misses,branch-misses"  # CPU events to track
    nsight_cpu_counter_scope: str = "process-tree"  # Counter scope
    
    # Call stack and symbol resolution
    nsight_cpu_call_stacks: bool = True  # Capture call stacks
    nsight_cpu_symbols: bool = True  # Resolve function symbols
    nsight_cpu_source_lines: bool = True  # Include source line information
    nsight_cpu_debug_info: bool = True  # Include debug information
    
    # Additional CPU profiling options
    nsight_cpu_kernel_trace: bool = True  # Trace CPU kernel functions
    nsight_cpu_user_trace: bool = True  # Trace user-space functions
    nsight_cpu_context_switches: bool = True  # Track context switches
    nsight_cpu_memory_bandwidth: bool = True  # Track memory bandwidth usage
    nsight_cpu_cache_events: bool = True  # Track cache events
    nsight_cpu_branch_events: bool = True  # Track branch prediction events
    
    # Advanced sampling options
    nsight_cpu_sample_all_threads: bool = True  # Sample all threads
    nsight_cpu_sample_kernel: bool = True  # Sample kernel space
    nsight_cpu_sample_user: bool = True  # Sample user space
    nsight_cpu_sample_idle: bool = False  # Sample idle time (usually not needed)
    
    # GPU management
    gpu_devices: Optional[List[int]] = None  # If None, will auto-detect
    cuda_visible_devices: Optional[str] = None

class ExportConfig(BaseModel):
    """Configuration for exporting results to various formats"""
    
    # Enable/disable export
    enable_export: bool = True
    
    # Export formats
    export_csv: bool = True
    export_influxdb: bool = False
    
    # Export directory (relative to results directory)
    export_dir: str = "exports"
    
    # CSV export options
    csv_summary: bool = True
    csv_requests: bool = True
    csv_system: bool = True
    
    # Performance optimizations
    use_streaming_export: bool = False  # Use streaming for very large datasets
    csv_chunk_size: int = 1000  # Chunk size for streaming export
    enable_fieldname_caching: bool = True  # Cache fieldnames for repeated exports
    
    # File naming
    use_timestamp_suffix: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"

class ExperimentConfig(BaseModel):
    """Complete experiment configuration"""
    
    # Experiment metadata
    suite: str = "default"
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Core configurations
    server: ServerConfig
    benchmark: BenchmarkConfig
    system: SystemConfig
    
    # Output configuration
    output_path: Optional[str] = None
    save_traces: bool = True
    
    # Export configuration
    export: ExportConfig
    
    # Validation and safety
    dry_run: bool = False
    timeout: int = 3600  # seconds
    max_retries: int = 3
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Debug: Log the raw YAML data
        print(f"DEBUG: Raw YAML data: {data}")
        if 'server' in data:
            print(f"DEBUG: Server data from YAML: {data['server']}")
        
        # Explicitly create nested configs to avoid default value issues
        if 'server' in data:
            data['server'] = ServerConfig(**data['server'])
            print(f"DEBUG: ServerConfig created with tensor_parallel_size: {data['server'].tensor_parallel_size}")
        if 'benchmark' in data:
            data['benchmark'] = BenchmarkConfig(**data['benchmark'])
        if 'system' in data:
            data['system'] = SystemConfig(**data['system'])
        if 'export' in data:
            data['export'] = ExportConfig(**data['export'])
            
        return cls(**data)
    
    @classmethod 
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
    
    def to_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.dict(), f, indent=2)
