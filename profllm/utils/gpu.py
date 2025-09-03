"""GPU resource management utilities"""

import logging
from typing import List, Dict, Any, Optional

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)

async def allocate_gpu_devices(num_gpus: int) -> List[int]:
    """Allocate GPU devices for tensor parallelism"""
    if not NVML_AVAILABLE:
        logger.warning("NVML not available, cannot allocate specific GPU devices")
        return list(range(num_gpus))
    
    try:
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        
        if num_gpus > device_count:
            raise ValueError(f"Requested {num_gpus} GPUs but only {device_count} available")
        
        # Get GPU utilization and memory usage
        gpu_info = []
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get memory info
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_free = memory_info.free
            
            # Get utilization
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except:
                gpu_util = 0
            
            gpu_info.append({
                "index": i,
                "memory_free": memory_free,
                "utilization": gpu_util,
                "score": memory_free * (100 - gpu_util)  # Prefer free memory and low utilization
            })
        
        # Sort by score (higher is better)
        gpu_info.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top GPUs
        selected_gpus = [gpu["index"] for gpu in gpu_info[:num_gpus]]
        selected_gpus.sort()  # Return in ascending order
        
        logger.info(f"Allocated GPUs: {selected_gpus}")
        return selected_gpus
        
    except Exception as e:
        logger.error(f"Error allocating GPU devices: {str(e)}")
        # Fallback to sequential allocation
        return list(range(num_gpus))

def get_gpu_info() -> List[Dict[str, Any]]:
    """Get information about all available GPUs"""
    if not NVML_AVAILABLE:
        return []
    
    gpu_info = []
    try:
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            
            # Basic info
            name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Utilization
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                memory_util = util.memory
            except:
                gpu_util = 0
                memory_util = 0
            
            # Temperature
            try:
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0
            
            # Power
            try:
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power = 0
            
            gpu_info.append({
                "index": i,
                "name": name,
                "memory_total": memory_info.total,
                "memory_free": memory_info.free,
                "memory_used": memory_info.used,
                "utilization_gpu": gpu_util,
                "utilization_memory": memory_util,
                "temperature": temp,
                "power_usage_watts": power
            })
            
    except Exception as e:
        logger.error(f"Error getting GPU info: {str(e)}")
    
    return gpu_info

def check_gpu_compatibility(
    tensor_parallel_size: int, 
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    model_size_gb: Optional[float] = None
) -> Dict[str, Any]:
    """Check if GPU configuration is compatible with requirements"""
    result = {
        "compatible": False,
        "total_gpus": 0,
        "total_memory_gb": 0.0,
        "memory_per_gpu_gb": 0.0,
        "required_gpus": 0,
        "warnings": [],
        "errors": []
    }
    
    if not NVML_AVAILABLE:
        result["warnings"].append("NVML not available, cannot check GPU compatibility")
        return result
    
    try:
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        result["total_gpus"] = device_count
        
        # Calculate total required GPUs
        total_required_gpus = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        result["required_gpus"] = total_required_gpus
        
        # Check if we have enough GPUs
        if total_required_gpus > device_count:
            error_msg = f"Not enough GPUs: required {total_required_gpus} (tensor_parallel_size={tensor_parallel_size} × pipeline_parallel_size={pipeline_parallel_size} × data_parallel_size={data_parallel_size}) but only {device_count} available"
            result["errors"].append(error_msg)
            return result
        
        # Check memory
        total_memory = 0
        min_memory = float('inf')
        
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gb = memory_info.total / (1024**3)
            
            total_memory += memory_gb
            min_memory = min(min_memory, memory_gb)
        
        result["total_memory_gb"] = total_memory
        result["memory_per_gpu_gb"] = min_memory
        
        # Estimate memory requirements
        if model_size_gb:
            # For tensor parallelism, model is split across GPUs
            # For pipeline parallelism, each stage needs the full model
            memory_per_gpu = model_size_gb / tensor_parallel_size
            memory_overhead = memory_per_gpu * 0.2  # 20% overhead estimate
            required_memory = memory_per_gpu + memory_overhead
            
            if required_memory > min_memory:
                result["warnings"].append(f"Estimated memory requirement ({required_memory:.1f}GB) > available memory per GPU ({min_memory:.1f}GB)")
            else:
                result["compatible"] = True
        else:
            result["compatible"] = True
            
    except Exception as e:
        result["errors"].append(f"Error checking GPU compatibility: {str(e)}")
    
    return result

def estimate_model_size_gb(model_name: str) -> Optional[float]:
    """Estimate model size in GB based on model name and common patterns"""
    model_name_lower = model_name.lower()
    
    # Common model size patterns
    if "7b" in model_name_lower or "7b-" in model_name_lower:
        return 14.0  # ~14GB for 7B models in FP16/BF16
    elif "8b" in model_name_lower or "8b-" in model_name_lower:
        return 16.0  # ~16GB for 8B models in FP16/BF16
    elif "13b" in model_name_lower or "13b-" in model_name_lower:
        return 26.0  # ~26GB for 13B models in FP16/BF16
    elif "30b" in model_name_lower or "30b-" in model_name_lower:
        return 60.0  # ~60GB for 30B models in FP16/BF16
    elif "65b" in model_name_lower or "65b-" in model_name_lower:
        return 130.0  # ~130GB for 65B models in FP16/BF16
    elif "70b" in model_name_lower or "70b-" in model_name_lower:
        return 140.0  # ~140GB for 70B models in FP16/BF16
    else:
        # Try to extract number from model name
        import re
        numbers = re.findall(r'\d+', model_name)
        if numbers:
            # Assume the largest number is the parameter count in billions
            param_count = max([int(n) for n in numbers])
            if param_count > 100:  # Likely billions
                return param_count * 2.0  # Rough estimate: 2GB per billion parameters in FP16/BF16
        
        logger.warning(f"Could not estimate model size for {model_name}")
        return None

# GPU utilities
