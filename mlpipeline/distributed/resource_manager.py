"""
Resource management for distributed computing backends.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import psutil

from ..core.errors import MLPipelineError, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"


@dataclass
class ResourceRequirement:
    """Resource requirement specification."""
    cpu_cores: Optional[float] = None
    memory_gb: Optional[float] = None
    gpu_count: Optional[int] = None
    disk_gb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "gpu_count": self.gpu_count,
            "disk_gb": self.disk_gb
        }


@dataclass
class ResourceUsage:
    """Current resource usage information."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_count: int = 0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "disk_percent": self.disk_percent,
            "disk_used_gb": self.disk_used_gb,
            "disk_total_gb": self.disk_total_gb,
            "gpu_count": self.gpu_count,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "gpu_memory_total_gb": self.gpu_memory_total_gb
        }


class ResourceManager:
    """Manages computational resources across distributed backends."""
    
    def __init__(self, 
                 cpu_limit_percent: float = 80.0,
                 memory_limit_percent: float = 80.0,
                 disk_limit_percent: float = 90.0,
                 monitoring_interval: int = 30):
        """
        Initialize resource manager.
        
        Args:
            cpu_limit_percent: CPU usage limit percentage
            memory_limit_percent: Memory usage limit percentage
            disk_limit_percent: Disk usage limit percentage
            monitoring_interval: Resource monitoring interval in seconds
        """
        self.cpu_limit_percent = cpu_limit_percent
        self.memory_limit_percent = memory_limit_percent
        self.disk_limit_percent = disk_limit_percent
        self.monitoring_interval = monitoring_interval
        
        self.backends: Dict[str, Any] = {}
        self.resource_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        
    def register_backend(self, name: str, backend: Any) -> None:
        """Register a distributed computing backend."""
        self.backends[name] = backend
        logger.info(f"Registered backend: {name}")
    
    def unregister_backend(self, name: str) -> None:
        """Unregister a distributed computing backend."""
        if name in self.backends:
            del self.backends[name]
            logger.info(f"Unregistered backend: {name}")
    
    def get_system_resources(self) -> ResourceUsage:
        """Get current system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
            
            # GPU usage (if available)
            gpu_count = 0
            gpu_memory_used_gb = 0.0
            gpu_memory_total_gb = 0.0
            
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used_gb += mem_info.used / (1024**3)
                    gpu_memory_total_gb += mem_info.total / (1024**3)
                    
            except (ImportError, Exception):
                # GPU monitoring not available
                pass
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                gpu_count=gpu_count,
                gpu_memory_used_gb=gpu_memory_used_gb,
                gpu_memory_total_gb=gpu_memory_total_gb
            )
            
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            raise MLPipelineError(f"Resource monitoring failed: {e}", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
    
    def get_backend_resources(self, backend_name: str) -> Dict[str, Any]:
        """Get resource usage for a specific backend."""
        if backend_name not in self.backends:
            raise MLPipelineError(f"Backend not registered: {backend_name}", ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM)
        
        backend = self.backends[backend_name]
        
        if hasattr(backend, 'get_resource_usage'):
            return backend.get_resource_usage()
        else:
            logger.warning(f"Backend {backend_name} does not support resource monitoring")
            return {}
    
    def check_resource_availability(self, requirements: ResourceRequirement) -> bool:
        """Check if required resources are available."""
        current_usage = self.get_system_resources()
        
        # Check CPU availability
        if requirements.cpu_cores:
            available_cpu = psutil.cpu_count() * (1 - current_usage.cpu_percent / 100)
            if available_cpu < requirements.cpu_cores:
                return False
        
        # Check memory availability
        if requirements.memory_gb:
            available_memory = current_usage.memory_total_gb - current_usage.memory_used_gb
            if available_memory < requirements.memory_gb:
                return False
        
        # Check GPU availability
        if requirements.gpu_count:
            if current_usage.gpu_count < requirements.gpu_count:
                return False
        
        # Check disk availability
        if requirements.disk_gb:
            available_disk = current_usage.disk_total_gb - current_usage.disk_used_gb
            if available_disk < requirements.disk_gb:
                return False
        
        return True
    
    def wait_for_resources(self, 
                          requirements: ResourceRequirement,
                          timeout: int = 300,
                          check_interval: int = 10) -> bool:
        """Wait for required resources to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.check_resource_availability(requirements):
                return True
            
            logger.info(f"Waiting for resources: {requirements.to_dict()}")
            time.sleep(check_interval)
        
        logger.warning(f"Timeout waiting for resources: {requirements.to_dict()}")
        return False
    
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor resources across all backends."""
        monitoring_data = {
            "timestamp": time.time(),
            "system": self.get_system_resources().to_dict(),
            "backends": {}
        }
        
        # Get backend-specific resource usage
        for name, backend in self.backends.items():
            try:
                backend_resources = self.get_backend_resources(name)
                monitoring_data["backends"][name] = backend_resources
            except Exception as e:
                logger.warning(f"Failed to get resources for backend {name}: {e}")
                monitoring_data["backends"][name] = {"error": str(e)}
        
        # Store in history
        self.resource_history.append(monitoring_data)
        
        # Keep only recent history (last 100 entries)
        if len(self.resource_history) > 100:
            self.resource_history = self.resource_history[-100:]
        
        # Check for resource alerts
        self._check_resource_alerts(monitoring_data)
        
        return monitoring_data
    
    def _check_resource_alerts(self, monitoring_data: Dict[str, Any]) -> None:
        """Check for resource usage alerts."""
        system_data = monitoring_data["system"]
        timestamp = monitoring_data["timestamp"]
        
        alerts = []
        
        # CPU alert
        if system_data["cpu_percent"] > self.cpu_limit_percent:
            alerts.append({
                "type": "cpu_high",
                "message": f"CPU usage is {system_data['cpu_percent']:.1f}% (limit: {self.cpu_limit_percent}%)",
                "timestamp": timestamp,
                "severity": "warning"
            })
        
        # Memory alert
        if system_data["memory_percent"] > self.memory_limit_percent:
            alerts.append({
                "type": "memory_high",
                "message": f"Memory usage is {system_data['memory_percent']:.1f}% (limit: {self.memory_limit_percent}%)",
                "timestamp": timestamp,
                "severity": "warning"
            })
        
        # Disk alert
        if system_data["disk_percent"] > self.disk_limit_percent:
            alerts.append({
                "type": "disk_high",
                "message": f"Disk usage is {system_data['disk_percent']:.1f}% (limit: {self.disk_limit_percent}%)",
                "timestamp": timestamp,
                "severity": "critical"
            })
        
        # Add alerts to history
        self.alerts.extend(alerts)
        
        # Keep only recent alerts (last 50)
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
        
        # Log alerts
        for alert in alerts:
            if alert["severity"] == "critical":
                logger.critical(alert["message"])
            else:
                logger.warning(alert["message"])
    
    def get_resource_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent resource usage history."""
        return self.resource_history[-limit:]
    
    def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent resource alerts."""
        return self.alerts[-limit:]
    
    def optimize_resource_allocation(self, task_requirements: List[ResourceRequirement]) -> Dict[str, Any]:
        """Optimize resource allocation for multiple tasks."""
        current_usage = self.get_system_resources()
        
        # Calculate total requirements
        total_cpu = sum(req.cpu_cores or 0 for req in task_requirements)
        total_memory = sum(req.memory_gb or 0 for req in task_requirements)
        total_gpu = sum(req.gpu_count or 0 for req in task_requirements)
        
        # Check if requirements can be satisfied
        available_cpu = psutil.cpu_count() * (1 - current_usage.cpu_percent / 100)
        available_memory = current_usage.memory_total_gb - current_usage.memory_used_gb
        available_gpu = current_usage.gpu_count
        
        recommendations = {
            "can_satisfy": True,
            "recommendations": [],
            "resource_allocation": {}
        }
        
        if total_cpu > available_cpu:
            recommendations["can_satisfy"] = False
            recommendations["recommendations"].append(
                f"Insufficient CPU: need {total_cpu:.1f}, available {available_cpu:.1f}"
            )
        
        if total_memory > available_memory:
            recommendations["can_satisfy"] = False
            recommendations["recommendations"].append(
                f"Insufficient memory: need {total_memory:.1f}GB, available {available_memory:.1f}GB"
            )
        
        if total_gpu > available_gpu:
            recommendations["can_satisfy"] = False
            recommendations["recommendations"].append(
                f"Insufficient GPU: need {total_gpu}, available {available_gpu}"
            )
        
        if recommendations["can_satisfy"]:
            # Suggest optimal backend allocation
            if len(self.backends) > 0:
                # Simple round-robin allocation
                backend_names = list(self.backends.keys())
                for i, req in enumerate(task_requirements):
                    backend_name = backend_names[i % len(backend_names)]
                    if backend_name not in recommendations["resource_allocation"]:
                        recommendations["resource_allocation"][backend_name] = []
                    recommendations["resource_allocation"][backend_name].append(req.to_dict())
        
        return recommendations
    
    def scale_backend(self, backend_name: str, scale_factor: float) -> bool:
        """Scale a backend up or down based on resource usage."""
        if backend_name not in self.backends:
            raise MLPipelineError(f"Backend not registered: {backend_name}", ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM)
        
        backend = self.backends[backend_name]
        
        # Check if backend supports scaling
        if hasattr(backend, 'scale'):
            try:
                backend.scale(scale_factor)
                logger.info(f"Scaled backend {backend_name} by factor {scale_factor}")
                return True
            except Exception as e:
                logger.error(f"Failed to scale backend {backend_name}: {e}")
                return False
        else:
            logger.warning(f"Backend {backend_name} does not support scaling")
            return False