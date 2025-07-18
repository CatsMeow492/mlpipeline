"""
Ray backend for distributed model training and hyperparameter optimization.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union
import pandas as pd
import numpy as np
import time

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    from ray.air import session
    import ray.data
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from ..core.interfaces import DistributedBackend
from ..core.errors import MLPipelineError, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class RayBackend(DistributedBackend):
    """Ray-based distributed computing backend."""
    
    def __init__(self,
                 ray_address: Optional[str] = None,
                 num_cpus: Optional[int] = None,
                 num_gpus: Optional[int] = None,
                 memory: Optional[int] = None,
                 dashboard_host: str = "127.0.0.1",
                 dashboard_port: int = 8265):
        """
        Initialize Ray backend.
        
        Args:
            ray_address: Address of existing Ray cluster
            num_cpus: Number of CPUs to use
            num_gpus: Number of GPUs to use
            memory: Memory limit in bytes
            dashboard_host: Dashboard host
            dashboard_port: Dashboard port
        """
        if not RAY_AVAILABLE:
            raise MLPipelineError(
                "Ray is not available. Install with: pip install ray[tune]",
                ErrorCategory.DEPENDENCY,
                ErrorSeverity.HIGH
            )
        
        self.ray_address = ray_address
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.memory = memory
        self.dashboard_host = dashboard_host
        self.dashboard_port = dashboard_port
        self.initialized = False
        
    def initialize(self) -> None:
        """Initialize Ray cluster."""
        try:
            if ray.is_initialized():
                logger.info("Ray is already initialized")
                self.initialized = True
                return
            
            init_kwargs = {
                "dashboard_host": self.dashboard_host,
                "dashboard_port": self.dashboard_port,
                "ignore_reinit_error": True
            }
            
            if self.ray_address:
                init_kwargs["address"] = self.ray_address
            else:
                # Local cluster configuration
                if self.num_cpus:
                    init_kwargs["num_cpus"] = self.num_cpus
                if self.num_gpus:
                    init_kwargs["num_gpus"] = self.num_gpus
                if self.memory:
                    init_kwargs["object_store_memory"] = self.memory
            
            ray.init(**init_kwargs)
            self.initialized = True
            
            cluster_resources = ray.cluster_resources()
            logger.info(f"Ray cluster initialized with resources: {cluster_resources}")
            logger.info(f"Ray dashboard available at: http://{self.dashboard_host}:{self.dashboard_port}")
            
        except Exception as e:
            raise MLPipelineError(f"Failed to initialize Ray backend: {e}", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
    
    def shutdown(self) -> None:
        """Shutdown Ray cluster."""
        if self.initialized and ray.is_initialized():
            ray.shutdown()
            self.initialized = False
            logger.info("Ray cluster shutdown")
    
    def is_available(self) -> bool:
        """Check if Ray backend is available."""
        return RAY_AVAILABLE and self.initialized and ray.is_initialized()
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the Ray cluster."""
        if not self.is_available():
            return {}
        
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        
        return {
            "cluster_resources": cluster_resources,
            "available_resources": available_resources,
            "nodes": len(ray.nodes()),
            "dashboard_url": f"http://{self.dashboard_host}:{self.dashboard_port}",
            "total_cpus": cluster_resources.get("CPU", 0),
            "total_gpus": cluster_resources.get("GPU", 0),
            "total_memory": cluster_resources.get("memory", 0),
        }
    
    def create_dataset(self, data: Union[pd.DataFrame, np.ndarray, List[str]]) -> Any:
        """Create Ray Dataset from various data sources."""
        if not self.is_available():
            raise MLPipelineError("Ray backend not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        if isinstance(data, pd.DataFrame):
            return ray.data.from_pandas(data)
        elif isinstance(data, np.ndarray):
            return ray.data.from_numpy(data)
        elif isinstance(data, list) and all(isinstance(x, str) for x in data):
            # Assume list of file paths
            return ray.data.read_parquet(data)
        else:
            raise MLPipelineError(f"Unsupported data type: {type(data)}", ErrorCategory.DATA, ErrorSeverity.MEDIUM)
    
    def distributed_hyperparameter_tuning(self,
                                         train_func: Callable,
                                         config: Dict[str, Any],
                                         num_samples: int = 10,
                                         max_concurrent_trials: int = 4,
                                         scheduler: str = "asha",
                                         search_algorithm: str = "optuna",
                                         metric: str = "accuracy",
                                         mode: str = "max") -> Any:
        """Perform distributed hyperparameter tuning using Ray Tune."""
        if not self.is_available():
            raise MLPipelineError("Ray backend not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        # Configure scheduler
        if scheduler == "asha":
            tune_scheduler = ASHAScheduler(
                metric=metric,
                mode=mode,
                max_t=100,
                grace_period=10,
                reduction_factor=2
            )
        else:
            tune_scheduler = None
        
        # Configure search algorithm
        if search_algorithm == "optuna":
            search_alg = OptunaSearch(metric=metric, mode=mode)
        else:
            search_alg = None
        
        # Run hyperparameter tuning
        tuner = tune.Tuner(
            train_func,
            param_space=config,
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                max_concurrent_trials=max_concurrent_trials,
                scheduler=tune_scheduler,
                search_alg=search_alg
            )
        )
        
        results = tuner.fit()
        return results
    
    def distributed_data_processing(self,
                                  dataset: Any,
                                  processing_func: Callable,
                                  batch_size: int = 1000) -> Any:
        """Apply distributed data processing using Ray Data."""
        if not self.is_available():
            raise MLPipelineError("Ray backend not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        return dataset.map_batches(processing_func, batch_size=batch_size)
    
    def parallel_model_training(self,
                              train_func: Callable,
                              data_splits: List[Any],
                              model_configs: List[Dict[str, Any]]) -> List[Any]:
        """Train multiple models in parallel using Ray."""
        if not self.is_available():
            raise MLPipelineError("Ray backend not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        @ray.remote
        def train_model_remote(data, config):
            return train_func(data, config)
        
        # Submit training tasks
        futures = []
        for data, config in zip(data_splits, model_configs):
            future = train_model_remote.remote(data, config)
            futures.append(future)
        
        # Wait for completion and return results
        return ray.get(futures)
    
    def distributed_inference(self,
                            model: Any,
                            dataset: Any,
                            batch_size: int = 1000) -> Any:
        """Perform distributed inference using Ray."""
        if not self.is_available():
            raise MLPipelineError("Ray backend not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        def predict_batch(batch):
            # Convert batch to appropriate format for model
            if hasattr(batch, 'to_pandas'):
                batch_df = batch.to_pandas()
            else:
                batch_df = batch
            
            predictions = model.predict(batch_df)
            return {"predictions": predictions}
        
        return dataset.map_batches(predict_batch, batch_size=batch_size)
    
    def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a task to Ray cluster."""
        if not self.is_available():
            raise MLPipelineError("Ray backend not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        @ray.remote
        def remote_func(*args, **kwargs):
            return func(*args, **kwargs)
        
        return remote_func.remote(*args, **kwargs)
    
    def get_task_result(self, task_id: Any) -> Any:
        """Get result from a Ray task."""
        if not self.is_available():
            raise MLPipelineError("Ray backend not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        return ray.get(task_id)
    
    def wait_for_tasks(self, task_ids: List[Any], timeout: Optional[float] = None) -> tuple:
        """Wait for Ray tasks to complete."""
        if not self.is_available():
            raise MLPipelineError("Ray backend not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        return ray.wait(task_ids, timeout=timeout)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage of the Ray cluster."""
        if not self.is_available():
            return {}
        
        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            # Calculate usage
            cpu_total = cluster_resources.get("CPU", 0)
            cpu_available = available_resources.get("CPU", 0)
            cpu_used = cpu_total - cpu_available
            
            gpu_total = cluster_resources.get("GPU", 0)
            gpu_available = available_resources.get("GPU", 0)
            gpu_used = gpu_total - gpu_available
            
            memory_total = cluster_resources.get("memory", 0)
            memory_available = available_resources.get("memory", 0)
            memory_used = memory_total - memory_available
            
            return {
                "cpu_total": cpu_total,
                "cpu_used": cpu_used,
                "cpu_usage_percent": (cpu_used / cpu_total * 100) if cpu_total > 0 else 0,
                "gpu_total": gpu_total,
                "gpu_used": gpu_used,
                "gpu_usage_percent": (gpu_used / gpu_total * 100) if gpu_total > 0 else 0,
                "memory_total_bytes": memory_total,
                "memory_used_bytes": memory_used,
                "memory_usage_percent": (memory_used / memory_total * 100) if memory_total > 0 else 0,
                "active_nodes": len(ray.nodes()),
            }
            
        except Exception as e:
            logger.warning(f"Failed to get resource usage: {e}")
            return {}


def create_ray_backend(config: Dict[str, Any]) -> RayBackend:
    """Factory function to create Ray backend from configuration."""
    return RayBackend(
        ray_address=config.get('ray_address'),
        num_cpus=config.get('num_cpus'),
        num_gpus=config.get('num_gpus'),
        memory=config.get('memory'),
        dashboard_host=config.get('dashboard_host', '127.0.0.1'),
        dashboard_port=config.get('dashboard_port', 8265)
    )


# Example training function for Ray Tune
def example_train_func(config):
    """Example training function for Ray Tune hyperparameter optimization."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Create model with hyperparameters from config
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        random_state=42
    )
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    accuracy = scores.mean()
    
    # Report results to Ray Tune
    session.report({"accuracy": accuracy})