# System utilities
"""System information and monitoring utilities"""

import platform
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    system_info = {
        "platform": {
            "system": platform.system(),
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "min_frequency": psutil.cpu_freq().min if psutil.cpu_freq() else None,
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent,
            "used": psutil.virtual_memory().used,
            "free": psutil.virtual_memory().free,
        },
        "disk": {
            "total": psutil.disk_usage('/').total,
            "used": psutil.disk_usage('/').used,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent,
        }
    }
    
    # Add GPU information if available
    try:
        from .gpu import get_gpu_info
        system_info["gpus"] = get_gpu_info()
    except Exception as e:
        logger.debug(f"Could not get GPU info: {str(e)}")
        system_info["gpus"] = []
    
    return system_info

def format_bytes(bytes_value: int) -> str:
    """Format bytes as human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def get_process_info(pid: int) -> Dict[str, Any]:
    """Get information about a specific process"""
    try:
        process = psutil.Process(pid)
        return {
            "pid": pid,
            "name": process.name(),
            "status": process.status(),
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "memory_info": process.memory_info()._asdict(),
            "create_time": process.create_time(),
            "cmdline": process.cmdline(),
        }
    except psutil.NoSuchProcess:
        return {"error": f"Process {pid} not found"}
    except Exception as e:
        return {"error": str(e)}