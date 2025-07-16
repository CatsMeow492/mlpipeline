"""
ML Pipeline - A comprehensive machine learning pipeline framework using open source tools.
"""

__version__ = "0.1.0"
__author__ = "ML Pipeline Team"

from .core import PipelineOrchestrator
from .config import ConfigManager

__all__ = ["PipelineOrchestrator", "ConfigManager"]