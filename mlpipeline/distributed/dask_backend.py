"""
Dask backend for distributed data processing and model training.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import dask
    import dask.dataframe as dd
    import dask.array as da
    from dask.distributed import Client, as_completed, wait
    from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
    from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from ..core.interfaces import DistributedBackend
from ..core.errors import MLPipelineError, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class DaskBackend(DistributedBackend):
    """Dask-based distributed computing backend."""
    
    def __init__(self, 
                 scheduler_address: Optional[str] = None,
                 n_workers: Optional[int] = None,
                 threads_per_worker: int = 2,
                 memory_limit: str = "2GB",
                 dashboard_address: Optional[str] = ":8787"):
        """
        Initialize Dask backend.
        
        Args:
            scheduler_address: Address of existing Dask scheduler
            n_workers: Number of workers to start (if creating local cluster)
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
            dashboard_address: Dashboard address
        """
        if not DASK_AVAILABLE:
            raise MLPipelineError(
                "Dask is not available. Install with: pip install dask[complete] dask-ml",
                ErrorCategory.DEPENDENCY,
                ErrorSeverity.HIGH
            )
        
        self.scheduler_address = scheduler_address
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.dashboard_address = dashboard_address
        self.client: Optional[Client] = None
        
    def initialize(self) -> None:
        """Initialize Dask client and cluster."""
        try:
            if self.scheduler_address:
                # Connect to existing cluster
                self.client = Client(self.scheduler_address)
                logger.info(f"Connected to Dask cluster at {self.scheduler_address}")
            else:
                # Create local cluster
                from dask.distributed import LocalCluster
                
                cluster = LocalCluster(
                    n_workers=self.n_workers or 2,
                    threads_per_worker=self.threads_per_worker,
                    memory_limit=self.memory_limit,
                    dashboard_address=self.dashboard_address
                )
                self.client = Client(cluster)
                logger.info(f"Created local Dask cluster with {cluster.workers} workers")
                
            logger.info(f"Dask dashboard available at: {self.client.dashboard_link}")
            
        except Exception as e:
            raise MLPipelineError(f"Failed to initialize Dask backend: {e}", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
    
    def shutdown(self) -> None:
        """Shutdown Dask client and cluster."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Dask client closed")
    
    def is_available(self) -> bool:
        """Check if Dask backend is available."""
        return DASK_AVAILABLE and self.client is not None
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the Dask cluster."""
        if not self.client:
            return {}
            
        return {
            "scheduler_address": self.client.scheduler.address,
            "dashboard_link": self.client.dashboard_link,
            "workers": len(self.client.scheduler.workers),
            "total_cores": sum(w.nthreads for w in self.client.scheduler.workers.values()),
            "total_memory": sum(w.memory_limit for w in self.client.scheduler.workers.values()),
        }
    
    def parallelize_dataframe(self, df: pd.DataFrame, npartitions: Optional[int] = None) -> Any:
        """Convert pandas DataFrame to Dask DataFrame."""
        if not self.client:
            raise MLPipelineError("Dask client not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
            
        npartitions = npartitions or self.client.nthreads()
        return dd.from_pandas(df, npartitions=npartitions)
    
    def parallelize_array(self, array: np.ndarray, chunks: Optional[tuple] = None) -> Any:
        """Convert numpy array to Dask array."""
        if not self.client:
            raise MLPipelineError("Dask client not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
            
        if chunks is None:
            # Auto-determine chunk size
            chunk_size = max(1000, len(array) // (self.client.nthreads() * 2))
            chunks = (chunk_size,) + array.shape[1:]
            
        return da.from_array(array, chunks=chunks)
    
    def compute(self, *args, **kwargs) -> Any:
        """Compute Dask collections."""
        if not self.client:
            raise MLPipelineError("Dask client not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
            
        return dask.compute(*args, **kwargs)
    
    def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a task to the Dask cluster."""
        if not self.client:
            raise MLPipelineError("Dask client not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
            
        return self.client.submit(func, *args, **kwargs)
    
    def map_tasks(self, func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
        """Map a function over an iterable using Dask."""
        if not self.client:
            raise MLPipelineError("Dask client not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
            
        futures = self.client.map(func, iterable, **kwargs)
        return self.client.gather(futures)
    
    def distributed_grid_search(self, 
                               estimator: Any,
                               param_grid: Dict[str, List[Any]],
                               X: Union[pd.DataFrame, np.ndarray],
                               y: Union[pd.Series, np.ndarray],
                               cv: int = 5,
                               scoring: str = 'accuracy',
                               **kwargs) -> Any:
        """Perform distributed grid search using Dask-ML."""
        if not self.client:
            raise MLPipelineError("Dask client not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        # Convert to Dask arrays if needed
        if isinstance(X, pd.DataFrame):
            X_dask = self.parallelize_dataframe(X).to_dask_array(lengths=True)
        elif isinstance(X, np.ndarray):
            X_dask = self.parallelize_array(X)
        else:
            X_dask = X
            
        if isinstance(y, (pd.Series, np.ndarray)):
            y_dask = da.from_array(y, chunks=X_dask.chunks[0])
        else:
            y_dask = y
        
        # Perform distributed grid search
        grid_search = DaskGridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            **kwargs
        )
        
        grid_search.fit(X_dask, y_dask)
        return grid_search
    
    def distributed_preprocessing(self, 
                                df: pd.DataFrame,
                                transformations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply preprocessing transformations using Dask."""
        if not self.client:
            raise MLPipelineError("Dask client not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        # Convert to Dask DataFrame
        ddf = self.parallelize_dataframe(df)
        
        for transform in transformations:
            transform_type = transform.get('type')
            columns = transform.get('columns', [])
            
            if transform_type == 'standard_scaler':
                # Use Dask-ML StandardScaler
                scaler = DaskStandardScaler()
                if columns:
                    ddf[columns] = scaler.fit_transform(ddf[columns])
                else:
                    numeric_cols = ddf.select_dtypes(include=[np.number]).columns
                    ddf[numeric_cols] = scaler.fit_transform(ddf[numeric_cols])
                    
            elif transform_type == 'fillna':
                fill_value = transform.get('value', 0)
                if columns:
                    ddf[columns] = ddf[columns].fillna(fill_value)
                else:
                    ddf = ddf.fillna(fill_value)
                    
            elif transform_type == 'drop_columns':
                ddf = ddf.drop(columns=columns)
                
            # Add more transformations as needed
        
        # Compute and return pandas DataFrame
        return ddf.compute()
    
    def batch_predict(self, 
                     model: Any,
                     X: Union[pd.DataFrame, np.ndarray],
                     batch_size: int = 1000) -> np.ndarray:
        """Perform batch prediction using Dask."""
        if not self.client:
            raise MLPipelineError("Dask client not initialized", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        
        # Convert to Dask array
        if isinstance(X, pd.DataFrame):
            X_dask = self.parallelize_dataframe(X).to_dask_array(lengths=True)
        else:
            X_dask = self.parallelize_array(X)
        
        # Define prediction function
        def predict_batch(X_batch):
            return model.predict(X_batch)
        
        # Apply prediction to each chunk
        predictions = X_dask.map_blocks(
            predict_batch,
            dtype=np.float64,
            drop_axis=1
        )
        
        return predictions.compute()
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage of the Dask cluster."""
        if not self.client:
            return {}
        
        try:
            # Get worker information
            workers_info = self.client.scheduler_info()['workers']
            
            total_memory_used = 0
            total_memory_limit = 0
            total_cpu_usage = 0
            
            for worker_info in workers_info.values():
                total_memory_used += worker_info.get('memory', 0)
                total_memory_limit += worker_info.get('memory_limit', 0)
                total_cpu_usage += worker_info.get('cpu', 0)
            
            return {
                "memory_used_bytes": total_memory_used,
                "memory_limit_bytes": total_memory_limit,
                "memory_usage_percent": (total_memory_used / total_memory_limit * 100) if total_memory_limit > 0 else 0,
                "cpu_usage_percent": total_cpu_usage,
                "active_workers": len(workers_info),
                "task_queue_length": len(self.client.scheduler.tasks)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get resource usage: {e}")
            return {}


def create_dask_backend(config: Dict[str, Any]) -> DaskBackend:
    """Factory function to create Dask backend from configuration."""
    return DaskBackend(
        scheduler_address=config.get('scheduler_address'),
        n_workers=config.get('n_workers'),
        threads_per_worker=config.get('threads_per_worker', 2),
        memory_limit=config.get('memory_limit', '2GB'),
        dashboard_address=config.get('dashboard_address', ':8787')
    )