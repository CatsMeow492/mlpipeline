# MLflow Integration

This document describes the MLflow integration capabilities of the ML Pipeline system, which provides comprehensive experiment tracking, model registry, and artifact management.

## Overview

The MLflow integration enables:
- **Experiment Tracking**: Automatic logging of parameters, metrics, and artifacts
- **Model Registry**: Versioned model storage with lifecycle management
- **Run Comparison**: Compare multiple experiments and their results
- **Artifact Management**: Store and retrieve model artifacts, plots, and data
- **Hyperparameter Optimization Tracking**: Track individual optimization trials

## Components

### MLflowConfig

Configuration class for MLflow integration settings.

```python
from mlpipeline.models.mlflow_integration import MLflowConfig

config = MLflowConfig(
    tracking_uri="http://localhost:5000",
    experiment_name="my-ml-experiment",
    log_params=True,
    log_metrics=True,
    log_artifacts=True,
    log_model=True,
    register_model=True,
    model_name="production-classifier",
    tags={"environment": "production", "version": "1.0"}
)
```

#### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tracking_uri` | str | None | MLflow tracking server URI |
| `experiment_name` | str | "ml-pipeline-experiment" | Name of the MLflow experiment |
| `run_name` | str | None | Name for individual runs |
| `log_params` | bool | True | Whether to log parameters |
| `log_metrics` | bool | True | Whether to log metrics |
| `log_artifacts` | bool | True | Whether to log artifacts |
| `log_model` | bool | True | Whether to log the trained model |
| `log_input_example` | bool | False | Whether to log input examples |
| `log_model_signature` | bool | True | Whether to log model signature |
| `register_model` | bool | False | Whether to register model in registry |
| `model_name` | str | None | Name for model registry |
| `model_stage` | str | "None" | Model stage (None, Staging, Production, Archived) |
| `tags` | dict | {} | Custom tags for runs |

### MLflowTracker

Core MLflow tracking functionality.

```python
from mlpipeline.models.mlflow_integration import MLflowTracker, MLflowConfig

# Create tracker
config = MLflowConfig(experiment_name="my-experiment")
tracker = MLflowTracker(config)

# Start a run
run_id = tracker.start_run(
    run_name="experiment_1",
    tags={"model_type": "random_forest"}
)

# Log parameters
tracker.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
})

# Log metrics
tracker.log_metrics({
    "accuracy": 0.95,
    "f1_score": 0.93
})

# Log artifacts
tracker.log_artifacts(["model.pkl", "plots/"])

# End run
tracker.end_run()
```

### MLflowIntegratedTrainer

Model trainer with built-in MLflow tracking.

```python
from mlpipeline.models.mlflow_integration import MLflowIntegratedTrainer, MLflowConfig
from mlpipeline.core.interfaces import ExecutionContext, ComponentType

# Configure MLflow
mlflow_config = MLflowConfig(
    experiment_name="model-training-experiment",
    log_model=True,
    register_model=True,
    model_name="my-classifier"
)

# Create trainer
trainer = MLflowIntegratedTrainer(mlflow_config)

# Configure training
config = {
    'mlflow': {
        'enabled': True,
        'experiment_name': 'model-training-experiment'
    },
    'training': {
        'model': {
            'framework': 'sklearn',
            'model_type': 'random_forest_classifier',
            'task_type': 'classification',
            'parameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        'target_column': 'target'
    }
}

# Create execution context
context = ExecutionContext(
    experiment_id="exp_001",
    stage_name="training",
    component_type=ComponentType.MODEL_TRAINING,
    config=config,
    artifacts_path="/path/to/artifacts",
    logger=trainer.logger
)

# Execute training with MLflow tracking
result = trainer.execute(context)

if result.success:
    print(f"MLflow run ID: {result.metadata['mlflow_run_id']}")
    print(f"Model metrics: {result.metrics}")
```

### MLflowIntegratedEvaluator

Model evaluator with MLflow tracking.

```python
from mlpipeline.models.mlflow_integration import MLflowIntegratedEvaluator

evaluator = MLflowIntegratedEvaluator(mlflow_config)

config = {
    'mlflow': {
        'enabled': True,
        'experiment_name': 'model-evaluation-experiment'
    },
    'evaluation': {
        'metrics': ['accuracy', 'f1_score', 'roc_auc']
    }
}

result = evaluator.execute(context)
```

### MLflowIntegratedHyperparameterTrainer

Hyperparameter optimization with MLflow tracking for individual trials.

```python
from mlpipeline.models.mlflow_integration import MLflowIntegratedHyperparameterTrainer

trainer = MLflowIntegratedHyperparameterTrainer(mlflow_config)

config = {
    'mlflow': {
        'enabled': True,
        'experiment_name': 'hyperparameter-optimization'
    },
    'training': {
        'model': {
            'framework': 'sklearn',
            'model_type': 'random_forest_classifier',
            'task_type': 'classification'
        },
        'hyperparameter_optimization': {
            'enabled': True,
            'n_trials': 50,
            'metric': 'accuracy',
            'parameter_space': {
                'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20}
            }
        }
    }
}

result = trainer.execute(context)

# Access optimization results
print(f"Parent run ID: {result.metadata['mlflow_parent_run_id']}")
print(f"Child runs: {result.metadata['mlflow_child_runs']}")
print(f"Best parameters: {result.metadata['best_hyperparameters']}")
```

## Configuration Examples

### Basic Configuration

```yaml
mlflow:
  enabled: true
  experiment_name: "basic-ml-experiment"
  tracking_uri: "http://localhost:5000"
  
training:
  model:
    framework: sklearn
    model_type: random_forest_classifier
    task_type: classification
    parameters:
      n_estimators: 100
      max_depth: 10
```

### Production Configuration with Model Registry

```yaml
mlflow:
  enabled: true
  experiment_name: "production-model-training"
  tracking_uri: "https://mlflow.company.com"
  registry_uri: "https://mlflow.company.com"
  
  # Model registry settings
  register_model: true
  model_name: "customer-churn-classifier"
  model_stage: "Staging"
  model_description: "Customer churn prediction model v2.0"
  
  # Logging settings
  log_model: true
  log_input_example: true
  log_model_signature: true
  log_artifacts: true
  
  # Tags
  tags:
    environment: "production"
    team: "data-science"
    version: "2.0"
    
training:
  model:
    framework: sklearn
    model_type: gradient_boosting_classifier
    task_type: classification
    parameters:
      n_estimators: 200
      learning_rate: 0.1
      max_depth: 8
```

### Hyperparameter Optimization Configuration

```yaml
mlflow:
  enabled: true
  experiment_name: "hyperparameter-optimization"
  
training:
  model:
    framework: sklearn
    model_type: random_forest_classifier
    task_type: classification
    
  hyperparameter_optimization:
    enabled: true
    method: optuna
    n_trials: 100
    metric: f1_score
    direction: maximize
    cv_folds: 5
    
    parameter_space:
      n_estimators:
        type: int
        low: 50
        high: 300
      max_depth:
        type: int
        low: 5
        high: 20
      min_samples_split:
        type: int
        low: 2
        high: 10
      max_features:
        type: categorical
        choices: ["sqrt", "log2", null]
```

## Model Registry Usage

### Registering Models

```python
# During training
config = MLflowConfig(
    register_model=True,
    model_name="production-classifier",
    model_stage="Staging",
    model_description="Latest version of the classifier"
)

trainer = MLflowIntegratedTrainer(config)
result = trainer.execute(context)

# Model is automatically registered
model_version = result.metadata.get('mlflow_model_version')
print(f"Registered model version: {model_version}")
```

### Managing Model Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition model to production
client.transition_model_version_stage(
    name="production-classifier",
    version="3",
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="production-classifier", 
    version="2",
    stage="Archived"
)
```

## Run Comparison and Analysis

### Comparing Runs

```python
# Search for runs
tracker = MLflowTracker(config)
runs = tracker.search_runs(
    filter_string="metrics.accuracy > 0.9",
    max_results=10
)

# Compare specific runs
run_ids = [run.run_id for run in runs[:3]]
comparison_df = tracker.compare_runs(run_ids)
print(comparison_df)
```

### Querying Experiments

```python
# Get run information
run_info = tracker.get_run_info("run-id-here")
print(f"Run: {run_info.run_name}")
print(f"Status: {run_info.status}")
print(f"Metrics: {run_info.metrics}")
print(f"Parameters: {run_info.params}")
```

## Best Practices

### 1. Experiment Organization

- Use descriptive experiment names that reflect the purpose
- Group related experiments together
- Use consistent naming conventions

```python
# Good experiment names
"customer-churn-baseline-models"
"customer-churn-feature-engineering"
"customer-churn-hyperparameter-tuning"
"customer-churn-production-candidates"
```

### 2. Tagging Strategy

- Use tags to categorize and filter runs
- Include metadata like environment, team, and version

```python
tags = {
    "environment": "development",
    "team": "data-science",
    "model_type": "ensemble",
    "feature_set": "v2",
    "data_version": "2023-12"
}
```

### 3. Model Registry Workflow

1. **Development**: Train models without registration
2. **Staging**: Register promising models in "Staging"
3. **Testing**: Validate staged models thoroughly
4. **Production**: Promote to "Production" stage
5. **Retirement**: Archive old models

### 4. Artifact Management

- Log all relevant artifacts (plots, reports, preprocessors)
- Use consistent artifact paths
- Include model explanations and documentation

```python
# Organize artifacts
artifacts = [
    "model/trained_model.pkl",
    "plots/confusion_matrix.png", 
    "plots/feature_importance.png",
    "reports/model_evaluation.html",
    "data/preprocessing_pipeline.pkl"
]
```

## Troubleshooting

### Common Issues

1. **MLflow Server Connection**
   ```python
   # Test connection
   import mlflow
   mlflow.set_tracking_uri("http://localhost:5000")
   print(mlflow.get_tracking_uri())
   ```

2. **Experiment Not Found**
   ```python
   # Create experiment if it doesn't exist
   try:
       experiment_id = mlflow.create_experiment("my-experiment")
   except mlflow.exceptions.MlflowException:
       experiment = mlflow.get_experiment_by_name("my-experiment")
       experiment_id = experiment.experiment_id
   ```

3. **Model Registration Failures**
   ```python
   # Check model registry URI
   print(mlflow.get_registry_uri())
   
   # Verify model exists in run
   run = mlflow.get_run("run-id")
   print(run.data.tags)
   ```

### Error Handling

The MLflow integration includes comprehensive error handling:

- Connection failures are logged as warnings
- Missing dependencies are detected early
- Failed operations don't crash the pipeline
- Detailed error messages for debugging

## Performance Considerations

### Large Models

- Use `log_model=False` for very large models
- Log model metadata instead of the full model
- Use external storage for large artifacts

### High-Frequency Logging

- Batch metric logging when possible
- Use asynchronous logging for better performance
- Consider sampling for very frequent updates

### Storage Management

- Regularly clean up old experiments
- Archive completed experiments
- Monitor storage usage

## Integration with Other Tools

### CI/CD Pipelines

```yaml
# GitHub Actions example
- name: Train Model with MLflow
  run: |
    export MLFLOW_TRACKING_URI=${{ secrets.MLFLOW_URI }}
    python train_model.py --config production.yaml
    
- name: Register Model
  if: github.ref == 'refs/heads/main'
  run: |
    python register_model.py --run-id ${{ env.RUN_ID }}
```

### Monitoring Systems

- Export MLflow metrics to monitoring dashboards
- Set up alerts for model performance degradation
- Track model usage and performance in production

## Security Considerations

### Authentication

```python
# Set up authentication
import os
os.environ['MLFLOW_TRACKING_USERNAME'] = 'username'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'

# Or use token-based auth
os.environ['MLFLOW_TRACKING_TOKEN'] = 'your-token'
```

### Access Control

- Use MLflow's built-in authentication
- Implement role-based access control
- Secure model registry access
- Audit model deployments

## Examples

See `examples/mlflow_integration_example.py` for comprehensive usage examples including:

- Basic MLflow tracking
- Integrated model training
- Hyperparameter optimization with tracking
- Model registry operations
- Run comparison and analysis

## Dependencies

Required packages:
- `mlflow>=2.0.0`
- `scikit-learn` (for sklearn model logging)
- `xgboost` (for XGBoost model logging)
- `torch` (for PyTorch model logging)

Install with:
```bash
pip install mlflow scikit-learn xgboost torch
```