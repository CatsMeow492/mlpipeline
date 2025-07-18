"""
Distributed computing support for ML Pipeline.

This module provides distributed computing capabilities using Dask and Ray
for scaling data processing, model training, and hyperparameter optimization.
"""

from .dask_backend import DaskBackend
from .ray_backend import RayBackend
from .resource_manager import ResourceManager
from .scheduler import DistributedScheduler

__all__ = [
    'DaskBackend',
    'RayBackend', 
    'ResourceManager',
    'DistributedScheduler'
]