"""Configuration schema definitions using Pydantic models."""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class DataSourceType(str, Enum):
    """Supported data source types."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    SQL = "sql"


class ModelType(str, Enum):
    """Supported model types."""
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    PYTORCH = "pytorch"
    HUGGINGFACE = "huggingface"


class HyperparameterMethod(str, Enum):
    """Hyperparameter optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    OPTUNA = "optuna"


class DriftMethod(str, Enum):
    """Drift detection methods."""
    KL_DIVERGENCE = "kl_divergence"
    PSI = "psi"
    WASSERSTEIN = "wasserstein"
    EVIDENTLY = "evidently"


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""
    type: DataSourceType
    path: str = Field(..., description="Path to data source")
    schema_path: Optional[str] = Field(None, description="Path to schema file")
    connection_string: Optional[str] = Field(None, description="Database connection string")
    table_name: Optional[str] = Field(None, description="Table name for SQL sources")
    query: Optional[str] = Field(None, description="SQL query for data extraction")
    options: Dict[str, Any] = Field(default_factory=dict, description="Source-specific options (e.g., CSV separator)")
    
    @model_validator(mode='after')
    def validate_sql_config(self):
        """Validate SQL-specific configuration."""
        if self.type == DataSourceType.SQL and not self.connection_string:
            raise ValueError("connection_string is required for SQL data sources")
        return self


class PreprocessingStep(BaseModel):
    """Configuration for preprocessing steps."""
    type: str = Field(..., description="Type of preprocessing step")
    columns: Optional[List[str]] = Field(None, description="Columns to apply preprocessing to")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step-specific parameters")


class DataConfig(BaseModel):
    """Data configuration section."""
    sources: List[DataSourceConfig] = Field(..., description="List of data sources")
    preprocessing: List[PreprocessingStep] = Field(default_factory=list, description="Preprocessing steps")
    train_split: float = Field(0.7, ge=0.1, le=0.9, description="Training data split ratio")
    validation_split: float = Field(0.15, ge=0.05, le=0.4, description="Validation data split ratio")
    test_split: float = Field(0.15, ge=0.05, le=0.4, description="Test data split ratio")
    stratify: bool = Field(True, description="Whether to stratify splits")
    random_state: int = Field(42, description="Random state for reproducibility")
    
    @model_validator(mode='after')
    def validate_splits(self):
        """Validate that splits sum to 1.0."""
        train = self.train_split
        val = self.validation_split
        test = self.test_split
        
        total = train + val + test
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Data splits must sum to 1.0, got {total}")
        
        return self


class HyperparameterConfig(BaseModel):
    """Hyperparameter optimization configuration."""
    method: HyperparameterMethod = Field(HyperparameterMethod.OPTUNA, description="Optimization method")
    n_trials: int = Field(50, ge=1, description="Number of optimization trials")
    timeout: Optional[int] = Field(None, ge=1, description="Timeout in seconds")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameter search space")
    cv_folds: int = Field(5, ge=2, le=10, description="Cross-validation folds")
    scoring: str = Field("accuracy", description="Scoring metric for optimization")


class ModelConfig(BaseModel):
    """Model configuration section."""
    type: ModelType = Field(..., description="Type of model to train")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    hyperparameter_tuning: Optional[HyperparameterConfig] = Field(None, description="Hyperparameter tuning config")
    early_stopping: bool = Field(False, description="Enable early stopping")
    early_stopping_patience: int = Field(10, ge=1, description="Early stopping patience")
    
    @field_validator('parameters')
    @classmethod
    def validate_model_parameters(cls, v, info):
        """Validate model-specific parameters."""
        model_type = info.data.get('type')
        
        # Add model-specific validation logic here
        if model_type == ModelType.XGBOOST:
            if 'n_estimators' in v and v['n_estimators'] <= 0:
                raise ValueError("n_estimators must be positive for XGBoost")
        
        return v


class EvaluationConfig(BaseModel):
    """Model evaluation configuration."""
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "f1_score"], description="Evaluation metrics")
    cross_validation: bool = Field(True, description="Enable cross-validation")
    cv_folds: int = Field(5, ge=2, le=10, description="Cross-validation folds")
    stratify: bool = Field(True, description="Stratify cross-validation")
    generate_plots: bool = Field(True, description="Generate evaluation plots")
    plot_types: List[str] = Field(default_factory=lambda: ["confusion_matrix", "roc_curve"], description="Types of plots to generate")


class DriftThresholds(BaseModel):
    """Drift detection thresholds."""
    data_drift: float = Field(0.1, ge=0.0, le=1.0, description="Data drift threshold")
    prediction_drift: float = Field(0.05, ge=0.0, le=1.0, description="Prediction drift threshold")
    feature_drift: float = Field(0.1, ge=0.0, le=1.0, description="Individual feature drift threshold")


class DriftConfig(BaseModel):
    """Drift detection configuration."""
    enabled: bool = Field(False, description="Enable drift detection")
    baseline_data: Optional[str] = Field(None, description="Path to baseline data")
    methods: List[DriftMethod] = Field(default_factory=lambda: [DriftMethod.EVIDENTLY], description="Drift detection methods")
    thresholds: DriftThresholds = Field(default_factory=DriftThresholds, description="Drift thresholds")
    monitoring_window: int = Field(100, ge=10, description="Number of samples in monitoring window")
    alert_channels: List[str] = Field(default_factory=list, description="Alert notification channels")


class FewShotConfig(BaseModel):
    """Few-shot learning configuration."""
    enabled: bool = Field(False, description="Enable few-shot learning")
    prompt_template: Optional[str] = Field(None, description="Path to prompt template file")
    max_examples: int = Field(5, ge=1, le=20, description="Maximum number of examples to include")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold for example selection")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Model for computing embeddings")
    example_store_path: Optional[str] = Field(None, description="Path to example store")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field("INFO", description="Logging level")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file_path: Optional[str] = Field(None, description="Log file path")
    max_file_size: str = Field("10MB", description="Maximum log file size")
    backup_count: int = Field(5, ge=1, description="Number of backup log files")


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    tags: List[str] = Field(default_factory=list, description="Experiment tags")
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow tracking server URI")
    artifact_location: Optional[str] = Field(None, description="Artifact storage location")


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    pipeline: ExperimentConfig = Field(..., description="Pipeline metadata")
    data: DataConfig = Field(..., description="Data configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    drift_detection: DriftConfig = Field(default_factory=DriftConfig, description="Drift detection configuration")
    few_shot: FewShotConfig = Field(default_factory=FewShotConfig, description="Few-shot learning configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    
    model_config = {
        "extra": "forbid",  # Forbid extra fields
        "validate_assignment": True,  # Validate on assignment
        "use_enum_values": True,  # Use enum values in serialization
    }
    
    @model_validator(mode='after')
    def validate_config_consistency(self):
        """Validate cross-field consistency."""
        # Validate few-shot learning dependencies
        if self.few_shot.enabled and not self.few_shot.prompt_template:
            raise ValueError("prompt_template is required when few-shot learning is enabled")
        
        # Validate drift detection dependencies
        if self.drift_detection.enabled and not self.drift_detection.baseline_data:
            raise ValueError("baseline_data is required when drift detection is enabled")
        
        return self


class ValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, message: str, errors: List[Dict[str, Any]] = None):
        self.message = message
        self.errors = errors or []
        super().__init__(self.message)


class ValidationResult(BaseModel):
    """Result of configuration validation."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    config: Optional[PipelineConfig] = None