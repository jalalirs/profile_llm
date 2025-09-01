
"""System metrics collection"""

import asyncio
import logging
import psutil
import time
from typing import Dict, Any, List, Optional

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from ..config.models import SystemConfig

logger = logging.getLogger(__name__)

class SystemMetrics:
    """Collects system resource metrics during benchmark execution"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitoring = False
        self.metrics_data: List[Dict[str, Any]] = []
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Initialize NVML if available and GPU monitoring is enabled
        self.nvml_initialized = False
        if NVML_AVAILABLE and config.monitor_gpu:
            try:
                nvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("NVML initialized for GPU monitoring")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {str(e)}")
    
    async def start_monitoring(self):
        """Start system metrics collection"""
        if self.monitoring:
            return
        
        logger.info("Starting system metrics monitoring...")
        self.monitoring = True
        self.metrics_data = []
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics"""
        if not self.monitoring:
            return {}
        
        logger.info("Stopping system metrics monitoring...")
        self.monitoring = False
        
        if self.monitor_task:
            await self.monitor_task
        
        # Calculate summary statistics
        summary = self._calculate_summary()
        
        return {
            "hardware_info": self._get_hardware_info(),
            "peak_gpu_memory_gb": summary.get("peak_gpu_memory_gb", 0.0),
            "avg_gpu_utilization": summary.get("avg_gpu_utilization", 0.0),
            "peak_cpu_percent": summary.get("peak_cpu_percent", 0.0),
            "avg_cpu_percent": summary.get("avg_cpu_percent", 0.0),
            "peak_memory_gb": summary.get("peak_memory_gb", 0.0),
            "avg_memory_gb": summary.get("avg_memory_gb", 0.0),
            "raw_metrics": self.metrics_data
        }
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                timestamp = time.time()
                metrics = {"timestamp": timestamp}
                
                # CPU metrics
                if self.config.monitor_cpu:
                    metrics.update(self._collect_cpu_metrics())
                
                # Memory metrics
                if self.config.monitor_memory:
                    metrics.update(self._collect_memory_metrics())
                
                # GPU metrics
                if self.config.monitor_gpu and self.nvml_initialized:
                    metrics.update(self._collect_gpu_metrics())
                
                self.metrics_data.append(metrics)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
            
            await asyncio.sleep(self.config.monitoring_interval)
    
    def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """Collect CPU metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    
    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory metrics"""
        memory = psutil.virtual_memory()
        return {
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_percent": memory.percent
        }
    
    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU metrics using NVML"""
        gpu_metrics = {
            "gpu_count": 0,
            "gpu_memory_used_gb": 0.0,
            "gpu_memory_total_gb": 0.0,
            "gpu_utilization": 0.0,
            "gpu_temperature": 0.0
        }
        
        try:
            device_count = nvml.nvmlDeviceGetCount()
            gpu_metrics["gpu_count"] = device_count
            
            total_memory_used = 0
            total_memory_total = 0
            total_utilization = 0
            total_temperature = 0
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # Memory info
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory_used += memory_info.used
                total_memory_total += memory_info.total
                
                # Utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                total_utilization += util.gpu
                
                # Temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                total_temperature += temp
            
            if device_count > 0:
                gpu_metrics.update({
                    "gpu_memory_used_gb": total_memory_used / (1024**3),
                    "gpu_memory_total_gb": total_memory_total / (1024**3),
                    "gpu_utilization": total_utilization / device_count,
                    "gpu_temperature": total_temperature / device_count
                })
                
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {str(e)}")
        
        return gpu_metrics
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from collected metrics"""
        if not self.metrics_data:
            return {}
        
        summary = {}
        
        # GPU metrics
        gpu_memory_values = [m.get("gpu_memory_used_gb", 0) for m in self.metrics_data]
        gpu_util_values = [m.get("gpu_utilization", 0) for m in self.metrics_data]
        
        if gpu_memory_values:
            summary["peak_gpu_memory_gb"] = max(gpu_memory_values)
            summary["avg_gpu_memory_gb"] = sum(gpu_memory_values) / len(gpu_memory_values)
        
        if gpu_util_values:
            summary["avg_gpu_utilization"] = sum(gpu_util_values) / len(gpu_util_values)
            summary["peak_gpu_utilization"] = max(gpu_util_values)
        
        # CPU metrics
        cpu_values = [m.get("cpu_percent", 0) for m in self.metrics_data]
        if cpu_values:
            summary["avg_cpu_percent"] = sum(cpu_values) / len(cpu_values)
            summary["peak_cpu_percent"] = max(cpu_values)
        
        # Memory metrics
        memory_values = [m.get("memory_used_gb", 0) for m in self.metrics_data]
        if memory_values:
            summary["avg_memory_gb"] = sum(memory_values) / len(memory_values)
            summary["peak_memory_gb"] = max(memory_values)
        
        return summary
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get static hardware information"""
        hardware_info = {
            "cpu_model": "Unknown",
            "cpu_cores": psutil.cpu_count(),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_devices": []
        }
        
        # Get CPU model
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        hardware_info["cpu_model"] = line.split(':')[1].strip()
                        break
        except:
            pass
        
        # Get GPU information
        if self.nvml_initialized:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_info = {
                        "index": i,
                        "name": name,
                        "memory_total_gb": memory_info.total / (1024**3)
                    }
                    hardware_info["gpu_devices"].append(gpu_info)
            except Exception as e:
                logger.error(f"Error getting GPU hardware info: {str(e)}")
        
        return hardware_info# Metrics collection
