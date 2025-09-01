ProfLLM - Professional vLLM Benchmarking Suite
Project Overview
ProfLLM is a comprehensive benchmarking suite designed to systematically evaluate vLLM performance across different configurations, models, and hardware setups. The system enables researchers and engineers to conduct reproducible performance experiments and analyze results through a filter-based web interface.
Problem Statement
Current vLLM performance analysis lacks systematic tooling for:

Testing multiple configuration combinations efficiently
Storing and comparing results across experiments
Filtering and analyzing performance data by configuration variables
Generating reproducible benchmarking workflows

Solution Architecture
ProfLLM consists of two independent components:

ProfLLM Library: Standalone Python library for executing benchmarks
Dashboard: Web application for data ingestion, filtering, and visualization

Component 1: ProfLLM Library
Objectives

Execute single vLLM benchmark experiments with comprehensive parameter coverage
Generate standardized JSON output with all configuration and performance data
Provide resource management (GPU allocation, port management)
Integrate with existing vLLM benchmark_serving.py infrastructure
Support all 150+ vLLM engine parameters for systematic testing

Architecture
profllm/
├── profllm/                       # Core library package
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── experiment.py          # Main experiment orchestration
│   │   ├── server.py              # vLLM server lifecycle management
│   │   ├── client.py              # Benchmark client (wraps benchmark_serving.py)
│   │   └── metrics.py             # System metrics collection
│   ├── config/
│   │   ├── __init__.py
│   │   ├── models.py              # Pydantic models for configuration
│   │   └── validation.py          # Configuration validation logic
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── gpu.py                 # GPU resource management
│   │   ├── ports.py               # Port allocation management
│   │   └── system.py              # System monitoring utilities
│   └── output/
│       ├── __init__.py
│       └── json.py                # JSON output formatting and validation
├── scripts/
│   └── run.py                     # CLI entry point
├── configs/
│   └── examples/                  # Example experiment configurations
├── requirements.txt
└── setup.py
Key Features
Configuration Management

Comprehensive coverage of all vLLM engine arguments
Type-safe configuration using Pydantic models
Validation of parameter combinations
Support for hardware-specific constraints

Resource Management

Automatic GPU allocation and isolation
Dynamic port assignment for concurrent experiments
Process lifecycle management with cleanup
Resource conflict detection and resolution

Benchmark Integration

Direct integration with vLLM's benchmark_serving.py
Support for all existing datasets (ShareGPT, synthetic, HuggingFace)
Custom workload generation capabilities
Profiler integration with torch.profiler

Output Standardization

Structured JSON schema with complete experiment metadata
All vLLM server parameters captured
Comprehensive performance metrics (throughput, latency percentiles)
System resource utilization data
Error handling and status reporting

Usage Pattern
bashprofllm run --config experiment.yaml --output results.json
JSON Output Schema
json{
  "run_id": "exp_20241201_143022_abc123",
  "suite": "baseline_performance", 
  "timestamp": "2024-12-01T14:30:22Z",
  "duration": 1847,
  
  "config": {
    "server": {
      "model": "meta-llama/Meta-Llama-3-8B",
      "tensor_parallel_size": 2,
      "gpu_memory_utilization": 0.9,
      "dtype": "bfloat16",
      "quantization": null,
      "max_model_len": 4096,
      "max_num_seqs": 32,
      // ... all 150+ vLLM parameters
    },
    "benchmark": {
      "dataset_name": "sharegpt",
      "num_prompts": 1000,
      "request_rate": "inf",
      "max_concurrency": null
    }
  },
  
  "results": {
    "request_throughput": 1247.52,
    "output_throughput": 8934.21,
    "median_ttft_ms": 145.7,
    "p95_ttft_ms": 342.1,
    "median_tpot_ms": 23.4,
    "p95_tpot_ms": 67.2,
    "completed_requests": 987,
    "failed_requests": 13
  },
  
  "system": {
    "peak_gpu_memory_gb": 45.2,
    "avg_gpu_utilization": 87.3,
    "hardware_info": {...}
  },
  
  "profiling": {
    "enabled": true,
    "trace_file_path": "/traces/exp_20241201_143022.perfetto.gz",
    "trace_file_size_mb": 23.7
  },
  
  "status": "completed"
}
Component 2: Dashboard
Objectives

Accept JSON experiment results via web interface
Provide comprehensive filtering system for all configuration parameters
Enable performance analysis and comparison across experiments
Support data export for external analysis tools
Offer visualization capabilities for performance trends

Architecture
dashboard/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI application entry point
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── data.py            # Experiment data endpoints
│   │   │   ├── filters.py         # Dynamic filtering endpoints
│   │   │   ├── upload.py          # JSON upload and processing
│   │   │   └── export.py          # Data export endpoints
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── db.py              # SQLAlchemy database models
│   │   │   ├── request.py         # API request models
│   │   │   └── response.py        # API response models
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── query.py           # Database query service
│   │   │   ├── analysis.py        # Performance analysis service
│   │   │   └── ingest.py          # JSON ingestion service
│   │   └── db/
│   │       ├── __init__.py
│   │       ├── connection.py      # Database connection management
│   │       └── schema.py          # Database schema definitions
│   ├── migrations/
│   │   └── init.sql               # Database initialization
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── filters/
│   │   │   │   ├── Panel.tsx      # Main filter panel component
│   │   │   │   ├── Model.tsx      # Model-specific filters
│   │   │   │   └── Config.tsx     # Configuration filters
│   │   │   ├── charts/
│   │   │   │   ├── Performance.tsx # Performance visualization
│   │   │   │   ├── Compare.tsx    # Experiment comparison
│   │   │   │   └── Table.tsx      # Tabular data display
│   │   │   ├── upload/
│   │   │   │   └── Upload.tsx     # JSON file upload interface
│   │   │   └── layout/
│   │   │       ├── Header.tsx     # Application header
│   │   │       └── Sidebar.tsx    # Navigation sidebar
│   │   ├── services/
│   │   │   └── api.ts             # API client service
│   │   ├── types/
│   │   │   └── benchmark.ts       # TypeScript type definitions
│   │   ├── hooks/
│   │   │   └── filters.ts         # Filter management hooks
│   │   ├── App.tsx                # Main application component
│   │   └── index.tsx              # Application entry point
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
│
└── docker-compose.yml             # Multi-service deployment
Database Design
Central Table: benchmark_runs

Single denormalized table with 150+ columns
Each row represents one complete experiment
All vLLM configuration parameters as individual columns
Performance metrics as dedicated columns
JSON columns for complex nested data
Comprehensive indexing for filter performance

Key Design Principles:

Filter-first architecture: every parameter is filterable
Optimized for analytical queries rather than transactional operations
Support for complex filter combinations
Efficient CSV export of filtered datasets

API Design
Data Upload Endpoints
POST /api/upload/json               # Single JSON file upload
POST /api/upload/batch              # Multiple file upload
POST /api/upload/raw                # Direct JSON payload
GET  /api/upload/status/{job_id}    # Upload processing status
Filtering System
GET  /api/filters/options           # Available filter values for all parameters
POST /api/filters/apply             # Apply filter combination, return results
GET  /api/filters/presets           # Saved filter combinations
POST /api/filters/presets           # Save current filter state
Data Access
GET  /api/data/experiments          # List experiments with pagination
GET  /api/data/experiments/{id}     # Single experiment details
POST /api/data/compare              # Multi-experiment comparison
GET  /api/data/summary              # Performance summary statistics
Export & Analysis
POST /api/export/csv                # Export filtered data as CSV
GET  /api/export/traces/{id}        # Download trace files
POST /api/analysis/trends           # Performance trends over time
POST /api/analysis/regression       # Regression detection
Frontend Features
Filter Interface

Dynamic filter generation based on available data
Support for range filters (numeric values)
Multi-select filters (categorical values)
Text search filters
Filter combination logic (AND/OR)
Saved filter presets

Visualization Components

Performance comparison charts (throughput, latency)
Trend analysis over time
Resource utilization plots
Configuration impact analysis
Interactive data tables with sorting and pagination

Data Management

Drag-and-drop JSON upload
Batch processing with progress indication
Upload validation and error reporting
Duplicate detection and handling
Data quality indicators

Technical Requirements
Hardware Support

Multi-GPU configurations (up to 8x H100)
Tensor parallelism scaling analysis
Pipeline parallelism support
Memory optimization testing

vLLM Integration

Complete coverage of all engine arguments
Integration with existing benchmarking infrastructure
Support for all model types (dense, MoE, multimodal)
Profiler integration for detailed performance analysis

Performance Requirements

Handle thousands of experiment records efficiently
Sub-second filter application on large datasets
Concurrent experiment execution support
Real-time system monitoring during benchmarks

Data Management

Structured data export (CSV, JSON)
Long-term experiment result storage
Data versioning and migration support
Backup and recovery capabilities

Use Cases
Research Scenarios

Model Comparison: Compare performance across different model sizes and architectures
Configuration Optimization: Find optimal settings for specific hardware configurations
Scaling Analysis: Analyze tensor parallelism efficiency across GPU counts
Memory Optimization: Test different memory management strategies
Precision Analysis: Compare impact of different data types and quantization methods

Operational Scenarios

Performance Regression Detection: Track performance changes over time
Hardware Utilization Analysis: Optimize resource allocation
Configuration Validation: Verify new configurations before deployment
Capacity Planning: Determine optimal hardware configurations for workloads

Implementation Phases
Phase 1: Core Library (Weeks 1-3)

Basic experiment execution framework
vLLM server management
JSON output formatting
Essential resource management

Phase 2: Backend Infrastructure (Weeks 4-6)

Database schema implementation
JSON ingestion API
Basic filtering endpoints
Data validation and error handling

Phase 3: Frontend Interface (Weeks 7-9)

Filter interface components
Basic visualization charts
Upload interface
Data table with sorting

Phase 4: Advanced Features (Weeks 10-12)

Comparison analysis
Export functionality
Advanced visualizations
Performance optimizations

Phase 5: Production Readiness (Weeks 13-14)

Deployment automation
Monitoring and logging
Documentation
Testing suite

Success Metrics
Functional Metrics

Support for all 150+ vLLM configuration parameters
Sub-second response time for filter operations
Successfully process 1000+ experiment results
Zero data loss during ingestion process

Usability Metrics

Intuitive filter interface requiring minimal training
Comprehensive export capabilities for external analysis
Clear visualization of performance trends and comparisons
Reliable experiment execution with proper error handling

Technical Metrics

99.9% uptime for dashboard services
Efficient database queries with proper indexing
Scalable architecture supporting concurrent users
Complete test coverage for critical components

Deployment Architecture
Production Environment:
├── Compute Nodes (8x H100 each)
│   └── ProfLLM Library execution
├── Database Server
│   └── PostgreSQL with experiment data
├── Web Server
│   ├── FastAPI backend
│   └── React frontend
└── File Storage
    └── Trace files and exports
The system is designed for independent component deployment, allowing flexible scaling based on workload requirements while maintaining data consistency and analysis capabilities.

VLLM ARGS

ModelConfig
Configuration for the model.

--model
Name or path of the Hugging Face model to use. It is also used as the content for model_name tag in metrics output when served_model_name is not specified.

Default: Qwen/Qwen3-0.6B

--runner
Possible choices: auto, draft, generate, pooling

The type of model runner to use. Each vLLM instance only supports one model runner, even if the same model can be used for multiple types.

Default: auto

--convert
Possible choices: auto, classify, embed, none, reward

Convert the model using adapters defined in vllm.model_executor.models.adapters. The most common use case is to adapt a text generation model to be used for pooling tasks.

Default: auto

--task
Possible choices: auto, classify, draft, embed, embedding, generate, reward, score, transcription, None

[DEPRECATED] The task to use the model for. If the model supports more than one model runner, this is used to select which model runner to run.

Note that the model may support other tasks using the same model runner.

Default: None

--tokenizer
Name or path of the Hugging Face tokenizer to use. If unspecified, model name or path will be used.

Default: None

--tokenizer-mode
Possible choices: auto, custom, mistral, slow

Tokenizer mode:

"auto" will use the fast tokenizer if available.

"slow" will always use the slow tokenizer.

"mistral" will always use the tokenizer from mistral_common.

"custom" will use --tokenizer to select the preregistered tokenizer.

Default: auto

--trust-remote-code, --no-trust-remote-code
Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.

Default: False

--dtype
Possible choices: auto, bfloat16, float, float16, float32, half

Data type for model weights and activations:

"auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.

"half" for FP16. Recommended for AWQ quantization.

"float16" is the same as "half".

"bfloat16" for a balance between precision and range.

"float" is shorthand for FP32 precision.

"float32" for FP32 precision.

Default: auto

--seed
Random seed for reproducibility. Initialized to None in V0, but initialized to 0 in V1.

Default: None

--hf-config-path
Name or path of the Hugging Face config to use. If unspecified, model name or path will be used.

Default: None

--allowed-local-media-path
Allowing API requests to read local images or videos from directories specified by the server file system. This is a security risk. Should only be enabled in trusted environments.

Default: ``

--revision
The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

Default: None

--code-revision
The specific revision to use for the model code on the Hugging Face Hub. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

Default: None

--rope-scaling
RoPE scaling configuration. For example, {"rope_type":"dynamic","factor":2.0}.

Should either be a valid JSON string or JSON keys passed individually.

Default: {}

--rope-theta
RoPE theta. Use with rope_scaling. In some cases, changing the RoPE theta improves the performance of the scaled model.

Default: None

--tokenizer-revision
The specific revision to use for the tokenizer on the Hugging Face Hub. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

Default: None

--max-model-len
Model context length (prompt and output). If unspecified, will be automatically derived from the model config.

When passing via --max-model-len, supports k/m/g/K/M/G in human-readable format. Examples:

1k -> 1000

1K -> 1024

25.6k -> 25,600

Default: None

--quantization, -q
Method used to quantize the weights. If None, we first check the quantization_config attribute in the model config file. If that is None, we assume the model weights are not quantized and use dtype to determine the data type of the weights.

Default: None

--enforce-eager, --no-enforce-eager
Whether to always use eager-mode PyTorch. If True, we will disable CUDA graph and always execute the model in eager mode. If False, we will use CUDA graph and eager execution in hybrid for maximal performance and flexibility.

Default: False

--max-seq-len-to-capture
Maximum sequence len covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode. Additionally for encoder-decoder models, if the sequence length of the encoder input is larger than this, we fall back to the eager mode.

Default: 8192

--max-logprobs
Maximum number of log probabilities to return when logprobs is specified in SamplingParams. The default value comes the default for the OpenAI Chat Completions API. -1 means no cap, i.e. all (output_length * vocab_size) logprobs are allowed to be returned and it may cause OOM.

Default: 20

--logprobs-mode
Possible choices: raw_logits, raw_logprobs, processed_logits, processed_logprobs

Indicates the content returned in the logprobs and prompt_logprobs. Supported mode: 1) raw_logprobs, 2) processed_logprobs, 3) raw_logits, 4) processed_logits. Raw means the values before applying any logit processors, like bad words. Processed means the values after applying all processors, including temperature and top_k/top_p.

Default: LogprobsMode.RAW_LOGPROBS

--disable-sliding-window, --no-disable-sliding-window
Whether to disable sliding window. If True, we will disable the sliding window functionality of the model, capping to sliding window size. If the model does not support sliding window, this argument is ignored.

Default: False

--disable-cascade-attn, --no-disable-cascade-attn
Disable cascade attention for V1. While cascade attention does not change the mathematical correctness, disabling it could be useful for preventing potential numerical issues. Note that even if this is set to False, cascade attention will be only used when the heuristic tells that it's beneficial.

Default: False

--skip-tokenizer-init, --no-skip-tokenizer-init
Skip initialization of tokenizer and detokenizer. Expects valid prompt_token_ids and None for prompt from the input. The generated output will contain token ids.

Default: False

--enable-prompt-embeds, --no-enable-prompt-embeds
If True, enables passing text embeddings as inputs via the prompt_embeds key. Note that enabling this will double the time required for graph compilation.

Default: False

--served-model-name
The model name(s) used in the API. If multiple names are provided, the server will respond to any of the provided names. The model name in the model field of a response will be the first name in this list. If not specified, the model name will be the same as the --model argument. Noted that this name(s) will also be used in model_name tag content of prometheus metrics, if multiple names provided, metrics tag will take the first one.

Default: None

--disable-async-output-proc
Disable async output processing. This may result in lower performance.

Default: False

--config-format
Possible choices: auto, hf, mistral

The format of the model config to load:

"auto" will try to load the config in hf format if available else it will try to load in mistral format.

"hf" will load the config in hf format.

"mistral" will load the config in mistral format.

Default: auto

--hf-token
The token to use as HTTP bearer authorization for remote files . If True, will use the token generated when running huggingface-cli login (stored in ~/.huggingface).

Default: None

--hf-overrides
If a dictionary, contains arguments to be forwarded to the Hugging Face config. If a callable, it is called to update the HuggingFace config.

Default: {}

--override-neuron-config
Initialize non-default neuron config or override default neuron config that are specific to Neuron devices, this argument will be used to configure the neuron config that can not be gathered from the vllm arguments. e.g. {"cast_logits_dtype": "bfloat16"}.

Should either be a valid JSON string or JSON keys passed individually.

Default: {}

--override-pooler-config
Initialize non-default pooling config or override default pooling config for the pooling model. e.g. {"pooling_type": "mean", "normalize": false}.

Default: None

--logits-processor-pattern
Optional regex pattern specifying valid logits processor qualified names that can be passed with the logits_processors extra completion argument. Defaults to None, which allows no processors.

Default: None

--generation-config
The folder path to the generation config. Defaults to "auto", the generation config will be loaded from model path. If set to "vllm", no generation config is loaded, vLLM defaults will be used. If set to a folder path, the generation config will be loaded from the specified folder path. If max_new_tokens is specified in generation config, then it sets a server-wide limit on the number of output tokens for all requests.

Default: auto

--override-generation-config
Overrides or sets generation config. e.g. {"temperature": 0.5}. If used with --generation-config auto, the override parameters will be merged with the default config from the model. If used with --generation-config vllm, only the override parameters are used.

Should either be a valid JSON string or JSON keys passed individually.

Default: {}

--enable-sleep-mode, --no-enable-sleep-mode
Enable sleep mode for the engine (only cuda platform is supported).

Default: False

--model-impl
Possible choices: auto, vllm, transformers

Which implementation of the model to use:

"auto" will try to use the vLLM implementation, if it exists, and fall back to the Transformers implementation if no vLLM implementation is available.

"vllm" will use the vLLM model implementation.

"transformers" will use the Transformers model implementation.

Default: auto

--override-attention-dtype
Override dtype for attention

Default: None

--logits-processors
One or more logits processors' fully-qualified class names or class definitions

Default: None

LoadConfig
Configuration for loading the model weights.

--load-format
The format of the model weights to load:

"auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available.

"pt" will load the weights in the pytorch bin format.

"safetensors" will load the weights in the safetensors format.

"npcache" will load the weights in pytorch format and store a numpy cache to speed up the loading.

"dummy" will initialize the weights with random values, which is mainly for profiling.

"tensorizer" will use CoreWeave's tensorizer library for fast weight loading. See the Tensorize vLLM Model script in the Examples section for more information.

"runai_streamer" will load the Safetensors weights using Run:ai Model Streamer.

"bitsandbytes" will load the weights using bitsandbytes quantization.

"sharded_state" will load weights from pre-sharded checkpoint files, supporting efficient loading of tensor-parallel models.

"gguf" will load weights from GGUF format files (details specified in https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

"mistral" will load weights from consolidated safetensors files used by Mistral models.

Other custom values can be supported via plugins.
Default: auto

--download-dir
Directory to download and load the weights, default to the default cache directory of Hugging Face.

Default: None

--model-loader-extra-config
Extra config for model loader. This will be passed to the model loader corresponding to the chosen load_format.

Default: {}

--ignore-patterns
The list of patterns to ignore when loading the model. Default to "original/*/" to avoid repeated loading of llama's checkpoints.

Default: None

--use-tqdm-on-load, --no-use-tqdm-on-load
Whether to enable tqdm for showing progress bar when loading model weights.

Default: True

--pt-load-map-location
pt_load_map_location: the map location for loading pytorch checkpoint, to support loading checkpoints can only be loaded on certain devices like "cuda", this is equivalent to {"": "cuda"}. Another supported format is mapping from different devices like from GPU 1 to GPU 0: {"cuda:1": "cuda:0"}. Note that when passed from command line, the strings in dictionary needs to be double quoted for json parsing. For more details, see original doc for map_location in https://pytorch.org/docs/stable/generated/torch.load.html

Default: cpu

DecodingConfig
Dataclass which contains the decoding strategy of the engine.

--guided-decoding-backend
Possible choices: auto, guidance, lm-format-enforcer, outlines, xgrammar

Which engine will be used for guided decoding (JSON schema / regex etc) by default. With "auto", we will make opinionated choices based on request contents and what the backend libraries currently support, so the behavior is subject to change in each release.

Default: auto

--guided-decoding-disable-fallback, --no-guided-decoding-disable-fallback
If True, vLLM will not fallback to a different backend on error.

Default: False

--guided-decoding-disable-any-whitespace, --no-guided-decoding-disable-any-whitespace
If True, the model will not generate any whitespace during guided decoding. This is only supported for xgrammar and guidance backends.

Default: False

--guided-decoding-disable-additional-properties, --no-guided-decoding-disable-additional-properties
If True, the guidance backend will not use additionalProperties in the JSON schema. This is only supported for the guidance backend and is used to better align its behaviour with outlines and xgrammar.

Default: False

--reasoning-parser
Possible choices: deepseek_r1, glm45, GptOss, granite, hunyuan_a13b, mistral, qwen3, step3

Select the reasoning parser depending on the model that you're using. This is used to parse the reasoning content into OpenAI API format.

Default: ``

ParallelConfig
Configuration for the distributed execution.

--distributed-executor-backend
Possible choices: external_launcher, mp, ray, uni

Backend to use for distributed model workers, either "ray" or "mp" (multiprocessing). If the product of pipeline_parallel_size and tensor_parallel_size is less than or equal to the number of GPUs available, "mp" will be used to keep processing on a single host. Otherwise, this will default to "ray" if Ray is installed and fail otherwise. Note that tpu only support Ray for distributed inference.

Default: None

--pipeline-parallel-size, -pp
Number of pipeline parallel groups.

Default: 1

--tensor-parallel-size, -tp
Number of tensor parallel groups.

Default: 1

--data-parallel-size, -dp
Number of data parallel groups. MoE layers will be sharded according to the product of the tensor parallel size and data parallel size.

Default: 1

--data-parallel-rank, -dpn
Data parallel rank of this instance. When set, enables external load balancer mode.

Default: None

--data-parallel-start-rank, -dpr
Starting data parallel rank for secondary nodes.

Default: None

--data-parallel-size-local, -dpl
Number of data parallel replicas to run on this node.

Default: None

--data-parallel-address, -dpa
Address of data parallel cluster head-node.

Default: None

--data-parallel-rpc-port, -dpp
Port for data parallel RPC communication.

Default: None

--data-parallel-backend, -dpb
Backend for data parallel, either "mp" or "ray".

Default: mp

--data-parallel-hybrid-lb, --no-data-parallel-hybrid-lb
Whether to use "hybrid" DP LB mode. Applies only to online serving and when data_parallel_size > 0. Enables running an AsyncLLM and API server on a "per-node" basis where vLLM load balances between local data parallel ranks, but an external LB balances between vLLM nodes/replicas. Set explicitly in conjunction with --data-parallel-start-rank.

Default: False

--enable-expert-parallel, --no-enable-expert-parallel
Use expert parallelism instead of tensor parallelism for MoE layers.

Default: False

--enable-eplb, --no-enable-eplb
Enable expert parallelism load balancing for MoE layers.

Default: False

--eplb-config
Expert parallelism configuration.

Should either be a valid JSON string or JSON keys passed individually.

Default: EPLBConfig(window_size=1000, step_interval=3000, num_redundant_experts=0, log_balancedness=False)

--num-redundant-experts
[DEPRECATED] --num-redundant-experts will be removed in v0.12.0.

Default: None

--eplb-window-size
[DEPRECATED] --eplb-window-size will be removed in v0.12.0.

Default: None

--eplb-step-interval
[DEPRECATED] --eplb-step-interval will be removed in v0.12.0.

Default: None

--eplb-log-balancedness, --no-eplb-log-balancedness
[DEPRECATED] --eplb-log-balancedness will be removed in v0.12.0.

Default: None

--max-parallel-loading-workers
Maximum number of parallel loading workers when loading model sequentially in multiple batches. To avoid RAM OOM when using tensor parallel and large models.

Default: None

--ray-workers-use-nsight, --no-ray-workers-use-nsight
Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.

Default: False

--disable-custom-all-reduce, --no-disable-custom-all-reduce
Disable the custom all-reduce kernel and fall back to NCCL.

Default: False

--worker-cls
The full name of the worker class to use. If "auto", the worker class will be determined based on the platform.

Default: auto

--worker-extension-cls
The full name of the worker extension class to use. The worker extension class is dynamically inherited by the worker class. This is used to inject new attributes and methods to the worker class for use in collective_rpc calls.

Default: ``

--enable-multimodal-encoder-data-parallel
Default: False

CacheConfig
Configuration for the KV cache.

--block-size
Possible choices: 1, 8, 16, 32, 64, 128

Size of a contiguous cache block in number of tokens. This is ignored on neuron devices and set to --max-model-len. On CUDA devices, only block sizes up to 32 are supported. On HPU devices, block size defaults to 128.

This config has no static default. If left unspecified by the user, it will be set in Platform.check_and_update_config() based on the current platform.

Default: None

--gpu-memory-utilization
The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. For example, a value of 0.5 would imply 50%% GPU memory utilization. If unspecified, will use the default value of 0.9. This is a per-instance limit, and only applies to the current vLLM instance. It does not matter if you have another vLLM instance running on the same GPU. For example, if you have two vLLM instances running on the same GPU, you can set the GPU memory utilization to 0.5 for each instance.

Default: 0.9

--swap-space
Size of the CPU swap space per GPU (in GiB).

Default: 4

--kv-cache-dtype
Possible choices: auto, fp8, fp8_e4m3, fp8_e5m2, fp8_inc

Data type for kv cache storage. If "auto", will use model data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports fp8 (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc).

Default: auto

--num-gpu-blocks-override
Number of GPU blocks to use. This overrides the profiled num_gpu_blocks if specified. Does nothing if None. Used for testing preemption.

Default: None

--enable-prefix-caching, --no-enable-prefix-caching
Whether to enable prefix caching. Disabled by default for V0. Enabled by default for V1.

Default: None

--prefix-caching-hash-algo
Possible choices: builtin, sha256, sha256_cbor_64bit

Set the hash algorithm for prefix caching:

"builtin" is Python's built-in hash.

"sha256" is collision resistant but with certain overheads. This option uses Pickle for object serialization before hashing.

"sha256_cbor_64bit" provides a reproducible, cross-language compatible hash. It serializes objects using canonical CBOR and hashes them with SHA-256. The resulting hash consists of the lower 64 bits of the SHA-256 digest.

Default: builtin

--cpu-offload-gb
The space in GiB to offload to CPU, per GPU. Default is 0, which means no offloading. Intuitively, this argument can be seen as a virtual way to increase the GPU memory size. For example, if you have one 24 GB GPU and set this to 10, virtually you can think of it as a 34 GB GPU. Then you can load a 13B model with BF16 weight, which requires at least 26GB GPU memory. Note that this requires fast CPU-GPU interconnect, as part of the model is loaded from CPU memory to GPU memory on the fly in each model forward pass.

Default: 0

--calculate-kv-scales, --no-calculate-kv-scales
This enables dynamic calculation of k_scale and v_scale when kv_cache_dtype is fp8. If False, the scales will be loaded from the model checkpoint if available. Otherwise, the scales will default to 1.0.

Default: False

--kv-sharing-fast-prefill, --no-kv-sharing-fast-prefill
This feature is work in progress and no prefill optimization takes place with this flag enabled currently.

In some KV sharing setups, e.g. YOCO (https://arxiv.org/abs/2405.05254), some layers can skip tokens corresponding to prefill. This flag enables attention metadata for eligible layers to be overridden with metadata necessary for implementing this optimization in some models (e.g. Gemma3n)

Default: False

--mamba-cache-dtype
Possible choices: auto, float32

The data type to use for the Mamba cache (both the conv as well as the ssm state). If set to 'auto', the data type will be inferred from the model config.

Default: auto

--mamba-ssm-cache-dtype
Possible choices: auto, float32

The data type to use for the Mamba cache (ssm state only, conv state will still be controlled by mamba_cache_dtype). If set to 'auto', the data type for the ssm state will be determined by mamba_cache_dtype.

Default: auto

MultiModalConfig
Controls the behavior of multimodal models.

--limit-mm-per-prompt
The maximum number of input items allowed per prompt for each modality. Defaults to 1 (V0) or 999 (V1) for each modality.

For example, to allow up to 16 images and 2 videos per prompt: {"image": 16, "video": 2}

Should either be a valid JSON string or JSON keys passed individually.

Default: {}

--media-io-kwargs
Additional args passed to process media inputs, keyed by modalities. For example, to set num_frames for video, set --media-io-kwargs '{"video": {"num_frames": 40} }'

Should either be a valid JSON string or JSON keys passed individually.

Default: {}

--mm-processor-kwargs
Overrides for the multi-modal processor obtained from transformers.AutoProcessor.from_pretrained.

The available overrides depend on the model that is being run.

For example, for Phi-3-Vision: {"num_crops": 4}.

Should either be a valid JSON string or JSON keys passed individually.

Default: None

--mm-processor-cache-gb
The size (in GiB) of the multi-modal processor cache, which is used to

This cache is duplicated for each API process and engine core process, resulting in a total memory usage of mm_processor_cache_gb * (api_server_count + data_parallel_size).

Set to 0 to disable this cache completely (not recommended).

Default: 4

--disable-mm-preprocessor-cache
Default: False

--mm-encoder-tp-mode
Possible choices: data, weights

Indicates how to optimize multi-modal encoder inference using tensor parallelism (TP).

"weights": Within the same vLLM engine, split the weights of each layer across TP ranks. (default TP behavior)
"data": Within the same vLLM engine, split the batched input data across TP ranks to process the data in parallel, while hosting the full weights on each TP rank. This batch-level DP is not to be confused with API request-level DP (which is controlled by --data-parallel-size). This is only supported on a per-model basis and falls back to "weights" if the encoder does not support DP.
Default: weights

--interleave-mm-strings, --no-interleave-mm-strings
Enable fully interleaved support for multimodal prompts.

Default: False

--skip-mm-profiling, --no-skip-mm-profiling
When enabled, skips multimodal memory profiling and only profiles with language backbone model during engine initialization.

This reduces engine startup time but shifts the responsibility to users for estimating the peak memory usage of the activation of multimodal encoder and embedding cache.

Default: False

LoRAConfig
Configuration for LoRA.

--enable-lora, --no-enable-lora
If True, enable handling of LoRA adapters.

Default: None

--enable-lora-bias, --no-enable-lora-bias
Enable bias for LoRA adapters.

Default: False

--max-loras
Max number of LoRAs in a single batch.

Default: 1

--max-lora-rank
Max LoRA rank.

Default: 16

--lora-extra-vocab-size
(Deprecated) Maximum size of extra vocabulary that can be present in a LoRA adapter. Will be removed in v0.12.0.

Default: 256

--lora-dtype
Possible choices: auto, bfloat16, float16

Data type for LoRA. If auto, will default to base model dtype.

Default: auto

--max-cpu-loras
Maximum number of LoRAs to store in CPU memory. Must be >= than max_loras.

Default: None

--fully-sharded-loras, --no-fully-sharded-loras
By default, only half of the LoRA computation is sharded with tensor parallelism. Enabling this will use the fully sharded layers. At high sequence length, max rank or tensor parallel size, this is likely faster.

Default: False

--default-mm-loras
Dictionary mapping specific modalities to LoRA model paths; this field is only applicable to multimodal models and should be leveraged when a model always expects a LoRA to be active when a given modality is present. Note that currently, if a request provides multiple additional modalities, each of which have their own LoRA, we do NOT apply default_mm_loras because we currently only support one lora adapter per prompt. When run in offline mode, the lora IDs for n modalities will be automatically assigned to 1-n with the names of the modalities in alphabetic order.

Should either be a valid JSON string or JSON keys passed individually.

Default: None

ObservabilityConfig
Configuration for observability - metrics and tracing.

--show-hidden-metrics-for-version
Enable deprecated Prometheus metrics that have been hidden since the specified version. For example, if a previously deprecated metric has been hidden since the v0.7.0 release, you use --show-hidden-metrics-for-version=0.7 as a temporary escape hatch while you migrate to new metrics. The metric is likely to be removed completely in an upcoming release.

Default: None

--otlp-traces-endpoint
Target URL to which OpenTelemetry traces will be sent.

Default: None

--collect-detailed-traces
Possible choices: all, model, worker, None, model,worker, model,all, worker,model, worker,all, all,model, all,worker

It makes sense to set this only if --otlp-traces-endpoint is set. If set, it will collect detailed traces for the specified modules. This involves use of possibly costly and or blocking operations and hence might have a performance impact.

Note that collecting detailed timing information for each request can be expensive.

Default: None

SchedulerConfig
Scheduler configuration.

--max-num-batched-tokens
Maximum number of tokens to be processed in a single iteration.

This config has no static default. If left unspecified by the user, it will be set in EngineArgs.create_engine_config based on the usage context.

Default: None

--max-num-seqs
Maximum number of sequences to be processed in a single iteration.

This config has no static default. If left unspecified by the user, it will be set in EngineArgs.create_engine_config based on the usage context.

Default: None

--max-num-partial-prefills
For chunked prefill, the maximum number of sequences that can be partially prefilled concurrently.

Default: 1

--max-long-partial-prefills
For chunked prefill, the maximum number of prompts longer than long_prefill_token_threshold that will be prefilled concurrently. Setting this less than max_num_partial_prefills will allow shorter prompts to jump the queue in front of longer prompts in some cases, improving latency.

Default: 1

--cuda-graph-sizes
Cuda graph capture sizes 1. if none provided, then default set to [min(max_num_seqs * 2, 512)] 2. if one value is provided, then the capture list would follow the pattern: [1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)] 3. more than one value (e.g. 1 2 128) is provided, then the capture list will follow the provided list.

Default: []

--long-prefill-token-threshold
For chunked prefill, a request is considered long if the prompt is longer than this number of tokens.

Default: 0

--num-lookahead-slots
The number of slots to allocate per sequence per step, beyond the known token ids. This is used in speculative decoding to store KV activations of tokens which may or may not be accepted.

NOTE: This will be replaced by speculative config in the future; it is present to enable correctness tests until then.

Default: 0

--scheduler-delay-factor
Apply a delay (of delay factor multiplied by previous prompt latency) before scheduling next prompt.

Default: 0.0

--preemption-mode
Possible choices: recompute, swap, None

Whether to perform preemption by swapping or recomputation. If not specified, we determine the mode as follows: We use recomputation by default since it incurs lower overhead than swapping. However, when the sequence group has multiple sequences (e.g., beam search), recomputation is not currently supported. In such a case, we use swapping instead.

Default: None

--scheduling-policy
Possible choices: fcfs, priority

The scheduling policy to use:

"fcfs" means first come first served, i.e. requests are handled in order of arrival.

"priority" means requests are handled based on given priority (lower value means earlier handling) and time of arrival deciding any ties).

Default: fcfs

--enable-chunked-prefill, --no-enable-chunked-prefill
If True, prefill requests can be chunked based on the remaining max_num_batched_tokens.

Default: None

--disable-chunked-mm-input, --no-disable-chunked-mm-input
If set to true and chunked prefill is enabled, we do not want to partially schedule a multimodal item. Only used in V1 This ensures that if a request has a mixed prompt (like text tokens TTTT followed by image tokens IIIIIIIIII) where only some image tokens can be scheduled (like TTTTIIIII, leaving IIIII), it will be scheduled as TTTT in one step and IIIIIIIIII in the next.

Default: False

--scheduler-cls
The scheduler class to use. "vllm.core.scheduler.Scheduler" is the default scheduler. Can be a class directly or the path to a class of form "mod.custom_class".

Default: vllm.core.scheduler.Scheduler

--disable-hybrid-kv-cache-manager, --no-disable-hybrid-kv-cache-manager
If set to True, KV cache manager will allocate the same size of KV cache for all attention layers even if there are multiple type of attention layers like full attention and sliding window attention.

Default: False

--async-scheduling, --no-async-scheduling
EXPERIMENTAL: If set to True, perform async scheduling. This may help reduce the CPU overheads, leading to better latency and throughput. However, async scheduling is currently not supported with some features such as structured outputs, speculative decoding, and pipeline parallelism.

Default: False

VllmConfig
Dataclass which contains all vllm-related configuration. This simplifies passing around the distinct configurations in the codebase.

--speculative-config
Speculative decoding configuration.

Should either be a valid JSON string or JSON keys passed individually.

Default: None

--kv-transfer-config
The configurations for distributed KV cache transfer.

Should either be a valid JSON string or JSON keys passed individually.

Default: None

--kv-events-config
The configurations for event publishing.

Should either be a valid JSON string or JSON keys passed individually.

Default: None

--compilation-config, -O
torch.compile and cudagraph capture configuration for the model.

As a shorthand, -O<n> can be used to directly specify the compilation level n: -O3 is equivalent to -O.level=3 (same as -O='{"level":3}'). Currently, -O and -O= are supported as well but this will likely be removed in favor of clearer -O syntax in the future.

NOTE: level 0 is the default level without any optimization. level 1 and 2 are for internal testing only. level 3 is the recommended level for production, also default in V1.

You can specify the full compilation config like so: {"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}

Should either be a valid JSON string or JSON keys passed individually.

Default: {"level":null,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":null,"use_inductor":true,"compile_sizes":null,"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":null,"use_cudagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":null,"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":null,"local_cache_dir":null}

--additional-config
Additional config for specified platform. Different platforms may support different configs. Make sure the configs are valid for the platform you are using. Contents must be hashable.

Default: {}

AsyncEngineArgs
--enable-log-requests, --no-enable-log-requests
Enable logging requests.

Default: False

--disable-log-requests, --no-disable-log-requests
[DEPRECATED] Disable logging requests.

Default: True


Profiling vLLM
We support tracing vLLM workers using the torch.profiler module. You can enable tracing by setting the VLLM_TORCH_PROFILER_DIR environment variable to the directory where you want to save the traces: VLLM_TORCH_PROFILER_DIR=/mnt/traces/

The OpenAI server also needs to be started with the VLLM_TORCH_PROFILER_DIR environment variable set.

When using benchmarks/benchmark_serving.py, you can enable profiling by passing the --profile flag.

Warning

Only enable profiling in a development environment.

Traces can be visualized using https://ui.perfetto.dev/.

Tip

Only send a few requests through vLLM when profiling, as the traces can get quite large. Also, no need to untar the traces, they can be viewed directly.

Example commands:

OpenAI Server:

VLLM_TORCH_PROFILER_DIR=/mnt/traces/ python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B
benchmark_serving.py:

python benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-70B --dataset-name sharegpt --dataset-p