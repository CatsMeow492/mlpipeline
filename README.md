# ML Pipeline Framework

A comprehensive machine learning pipeline framework built with open source tools, designed for production-ready ML workflows with advanced error handling, drift detection, and few-shot learning capabilities.

## Features

### 🚀 Core Pipeline Orchestration
- **Robust Pipeline Execution**: Advanced orchestrator with dependency resolution, parallel execution, and comprehensive error handling
- **Component Registry**: Dynamic component loading and management system
- **Structured Logging**: Correlation ID tracking and detailed execution logs
- **Checkpointing**: Resume pipeline execution from failure points

### 📊 Data Management
- **Multi-Source Support**: CSV, JSON, Parquet, and SQL database connectors
- **Advanced Preprocessing**: Scikit-learn based transformations with metadata tracking
- **Data Validation**: Schema validation and type checking
- **Version Control**: DVC integration for data versioning and lineage

### 🤖 Model Training & Evaluation
- **Multi-Framework Support**: Scikit-learn, XGBoost, PyTorch, and Hugging Face
- **Hyperparameter Optimization**: Optuna integration with grid search, random search, and Bayesian optimization
- **Comprehensive Evaluation**: Standard metrics, visualizations, and model comparison
- **Experiment Tracking**: MLflow integration for parameter, metric, and artifact logging

### 🔍 Monitoring & Drift Detection
- **Evidently AI Integration**: Data and prediction drift detection
- **Multiple Algorithms**: KL divergence, PSI, Wasserstein distance
- **Alerting System**: Configurable thresholds and notification channels
- **Drift Visualization**: Automated report generation

### 🎯 Few-Shot Learning
- **Prompt Management**: Template system with variable substitution
- **Example Store**: Similarity-based example selection using embeddings
- **LLM Integration**: Hugging Face transformers and OpenAI-compatible APIs
- **Context Injection**: Automatic few-shot example context generation

### ⚡ Inference & Deployment
- **Batch Processing**: Scalable batch inference with progress tracking
- **Real-time API**: Fast inference endpoints with confidence scoring
- **Model Validation**: Compatibility checks and preprocessing consistency
- **Distributed Computing**: Dask and Ray support for scaling

## Installation

```bash
pip install mlpipeline
```

### Optional Dependencies

```bash
# For distributed computing
pip install mlpipeline[distributed]

# For GPU support
pip install mlpipeline[gpu]

# For development
pip install mlpipeline[dev]
```

## Quick Start

### 1. Create Configuration

```yaml
# config.yaml
pipeline:
  name: "my_experiment"
  description: "Example ML pipeline"
  tags: ["classification", "demo"]

data:
  sources:
    - type: csv
      path: "data/train.csv"
  preprocessing:
    - type: standard_scaler
      columns: ["feature1", "feature2"]
  train_split: 0.7
  validation_split: 0.15
  test_split: 0.15

model:
  type: sklearn
  parameters:
    algorithm: RandomForestClassifier
    n_estimators: 100
    random_state: 42
  hyperparameter_tuning:
    method: optuna
    n_trials: 50
    parameters:
      n_estimators: [50, 100, 200]
      max_depth: [3, 5, 10]

evaluation:
  metrics: ["accuracy", "f1_score", "precision", "recall"]
  cross_validation: true
  cv_folds: 5

drift_detection:
  enabled: true
  baseline_data: "data/baseline.csv"
  methods: ["evidently"]
  thresholds:
    data_drift: 0.1
    prediction_drift: 0.05

few_shot:
  enabled: false
  max_examples: 5
  similarity_threshold: 0.8
```

### 2. Run Pipeline

```python
from mlpipeline import PipelineOrchestrator, ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("config.yaml")

# Create orchestrator
orchestrator = PipelineOrchestrator(
    max_workers=4,
    enable_checkpointing=True
)

# Execute pipeline
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
import logging

context = ExecutionContext(
    experiment_id="exp_001",
    stage_name="training",
    component_type=ComponentType.MODEL_TRAINING,
    config=config.model_dump(),
    artifacts_path="./artifacts",
    logger=logging.getLogger("pipeline")
)

# Create stages from config and execute
stages = orchestrator.create_stages_from_config(config)
result = orchestrator.execute_pipeline(stages, context)

if result.success:
    print(f"Pipeline completed successfully in {result.execution_time:.2f}s")
    print(f"Artifacts: {result.artifacts}")
    print(f"Metrics: {result.metrics}")
else:
    print(f"Pipeline failed: {result.error_message}")
```

### 3. CLI Usage

```bash
# Train model
mlpipeline train --config config.yaml --experiment-name my_experiment

# Run inference
mlpipeline predict --model-path ./models/best_model.pkl --data ./data/test.csv

# Monitor for drift
mlpipeline monitor --config config.yaml --baseline ./data/baseline.csv

# Evaluate model
mlpipeline evaluate --model-path ./models/best_model.pkl --test-data ./data/test.csv
```

## Architecture

### Component System
The framework uses a component-based architecture where each pipeline stage consists of reusable components:

```python
from mlpipeline.core.interfaces import PipelineComponent, ComponentType, ExecutionContext, ExecutionResult

class CustomPreprocessor(PipelineComponent):
    def __init__(self):
        super().__init__(ComponentType.DATA_PREPROCESSING)
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        # Your preprocessing logic here
        return ExecutionResult(
            success=True,
            artifacts=["preprocessed_data.csv"],
            metrics={"rows_processed": 1000},
            metadata={"preprocessing_time": 5.2}
        )
    
    def validate_config(self, config: dict) -> bool:
        return "input_path" in config

# Register component
from mlpipeline.core.registry import component_registry
component_registry.register_component("custom_preprocessor", CustomPreprocessor)
```

### Error Handling
Advanced error handling with automatic recovery:

```python
from mlpipeline.core.errors import ErrorHandler, RecoveryStrategy, RecoveryAction, ErrorCategory

# Configure custom recovery strategy
error_handler = ErrorHandler()
error_handler.register_recovery_strategy(
    ErrorCategory.DATA,
    RecoveryStrategy(
        action=RecoveryAction.RETRY,
        max_retries=3,
        retry_delay=5.0,
        exponential_backoff=True
    )
)
```

## Configuration Reference

### Data Sources
- **CSV**: `type: csv, path: string, schema_path?: string`
- **JSON**: `type: json, path: string`
- **Parquet**: `type: parquet, path: string`
- **SQL**: `type: sql, connection_string: string, table_name: string, query?: string`

### Model Types
- **Scikit-learn**: `type: sklearn, parameters: {algorithm: string, ...}`
- **XGBoost**: `type: xgboost, parameters: {n_estimators: int, ...}`
- **PyTorch**: `type: pytorch, parameters: {model_class: string, ...}`
- **Hugging Face**: `type: huggingface, parameters: {model_name: string, ...}`

### Hyperparameter Optimization
- **Grid Search**: `method: grid_search, parameters: {param: [values]}`
- **Random Search**: `method: random_search, n_trials: int`
- **Optuna**: `method: optuna, n_trials: int, timeout?: int`

## Development

### Setup Development Environment

```bash
git clone https://github.com/your-org/mlpipeline.git
cd mlpipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev]
```

### Run Tests

```bash
pytest tests/ -v --cov=mlpipeline --cov-report=html
```

### Code Quality

```bash
# Format code
black mlpipeline tests

# Lint code
flake8 mlpipeline tests

# Type checking
mypy mlpipeline
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Kubernetes deployment support
- [ ] Advanced AutoML capabilities
- [ ] Real-time streaming data support
- [ ] Enhanced few-shot learning with retrieval augmentation
- [ ] Integration with more cloud providers
- [ ] Advanced model interpretability features

## Support

- 📖 [Documentation](https://mlpipeline.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/your-org/mlpipeline/issues)
- 💬 [Discussions](https://github.com/your-org/mlpipeline/discussions)
- 📧 [Email Support](mailto:support@mlpipeline.dev)