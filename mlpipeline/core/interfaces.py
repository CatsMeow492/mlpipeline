"""Core interfaces and abstract base classes for pipeline components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging


class PipelineStatus(Enum):
    """Pipeline execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ComponentType(Enum):
    """Types of pipeline components."""
    DATA_INGESTION = "data_ingestion"
    DATA_PREPROCESSING = "data_preprocessing"
    DATA_VALIDATION = "data_validation"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_INFERENCE = "model_inference"
    DRIFT_DETECTION = "drift_detection"
    FEW_SHOT_LEARNING = "few_shot_learning"


@dataclass
class ExecutionContext:
    """Context information for pipeline execution."""
    experiment_id: str
    stage_name: str
    component_type: ComponentType
    config: Dict[str, Any]
    artifacts_path: str
    logger: logging.Logger
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionResult:
    """Result of component execution."""
    success: bool
    artifacts: List[str]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


class PipelineComponent(ABC):
    """Abstract base class for all pipeline components."""
    
    def __init__(self, component_type: ComponentType):
        self.component_type = component_type
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute the component with given context."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate component-specific configuration."""
        pass
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup component before execution (optional override)."""
        pass
    
    def cleanup(self, context: ExecutionContext) -> None:
        """Cleanup after component execution (optional override)."""
        pass


@dataclass
class PipelineStage:
    """Represents a stage in the pipeline with its components."""
    name: str
    components: List[PipelineComponent]
    dependencies: List[str] = None
    parallel: bool = False
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class PipelineOrchestrator(ABC):
    """Abstract base class for pipeline orchestrators."""
    
    @abstractmethod
    def execute_pipeline(self, stages: List[PipelineStage], context: ExecutionContext) -> ExecutionResult:
        """Execute a complete pipeline."""
        pass
    
    @abstractmethod
    def execute_stage(self, stage: PipelineStage, context: ExecutionContext) -> ExecutionResult:
        """Execute a single pipeline stage."""
        pass