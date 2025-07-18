# MLflow Integration Guide

This guide explains how to use MLflow with the ML Pipeline framework for experiment tracking, model registry, and artifact management.

## Overview

MLflow integration provides:
- **Experiment Tracking**: Automatic logging of parameters, metrics, and artifacts
- **Model Registry**: Version control for trained models
- **Artifact Storage**: Centralized storage for model artifacts and outputs
- **Comparison Tools**: Compare experiments and model performance

## Setup

### Local MLflow Server

For development and testing:

```bash
# Install MLflow
pip install mlflow

# Start local tracking server
mlflow server --host 0.0.0.0 --port 5000

# Access UI at http://localhost:5000
```

### Remote MLflow Server

For production environments, configure a remote MLflow server:

```yaml
pipeline:
  mlflow_tracking_uri: "http://mlflow.company.com:5000"
  artifact_location: "s3://ml-artifacts/experiments"
```

### Database Backend

For persistent storage, configure a database backend:

```bash
# PostgreSQL example
mlflow server \
  --backend-store-uri postgresql://user:password@localhost:5432/mlflow \
  --default-artifact-root s3://ml-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

## Configuration

### Basic Configuration

```yaml
pipeline:
  name: "customer_segmentation"
  description: "Customer segmentation using clustering"
  tags: ["clustering", "customer", "production"]
  mlflow_tracking_uri: "http://localhost:5000"  # MLflow server URL
  artifact_location: "s3://ml-artifacts"        # Optional: custom artifact location
```

### Environment Variables

You can also configure MLflow using environment variables:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://minio:9000  # For MinIO
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

## Automatic Logging

The ML Pipeline framework automatically logs:

### Parameters
- All model parameters
- Preprocessing configuration
- Data split ratios
- Hyperparameter tuning settings

### Metrics
- Training metrics (accuracy, loss, etc.)
- Validation metrics
- Cross-validation scores
- Hyperparameter optimization results

### Artifacts
- Trained model files
- Preprocessing pipelines
- Evaluation plots and reports
- Configuration files
- Feature importance data

### Tags
- Pipeline name and version
- Model type
- Dataset information
- Custom tags from configuration

## Example Logged Information

When you run a training pipeline, MLflow automatically logs:

```python
# Parameters logged
{
    "model.type": "xgboost",
    "model.n_estimators": 1000,
    "model.max_depth": 6,
    "model.learning_rate": 0.1,
    "data.train_split": 0.7,
    "preprocessing.standard_scaler.columns": ["feature1", "feature2"]
}

# Metrics logged
{
    "train_accuracy": 0.95,
    "val_accuracy": 0.87,
    "test_accuracy": 0.89,
    "train_f1": 0.94,
    "val_f1": 0.86,
    "test_f1": 0.88,
    "training_time": 45.2
}

# Artifacts logged
[
    "model/model.pkl",
    "preprocessor/preprocessor.pkl",
    "plots/confusion_matrix.png",
    "plots/roc_curve.png",
    "plots/feature_importance.png",
    "config/pipeline_config.yaml",
    "reports/evaluation_report.json"
]
```

## Model Registry

### Automatic Model Registration

Models are automatically registered when training completes successfully:

```yaml
# Configuration enables automatic registration
pipeline:
  name: "fraud_detection_v2"
  mlflow_tracking_uri: "http://localhost:5000"

# Model will be registered as "fraud_detection_v2" with version number
```

### Manual Model Management

You can also manage models manually using MLflow CLI:

```bash
# Register a model
mlflow models register \
  --model-uri runs:/abc123/model \
  --name "fraud_detection"

# Transition model to staging
mlflow models transition \
  --name "fraud_detection" \
  --version 2 \
  --stage "Staging"

# Transition to production
mlflow models transition \
  --name "fraud_detection" \
  --version 2 \
  --stage "Production"
```

## Experiment Organization

### Experiment Naming

Use descriptive experiment names:

```yaml
pipeline:
  name: "customer_churn_v3_xgboost_tuned"  # Version, model type, optimization
  description: "XGBoost with Optuna tuning for customer churn prediction"
  tags: ["churn", "xgboost", "tuned", "v3"]
```

### Hierarchical Organization

Organize experiments by project:

```yaml
# Project: Customer Analytics
pipeline:
  name: "customer_analytics/churn_prediction"
  tags: ["customer_analytics", "churn"]

# Project: Fraud Detection  
pipeline:
  name: "fraud_detection/transaction_classifier"
  tags: ["fraud_detection", "transactions"]
```

## Hyperparameter Tracking

MLflow automatically tracks hyperparameter optimization:

```yaml
model:
  hyperparameter_tuning:
    method: "optuna"
    n_trials: 100
    parameters:
      n_estimators: [100, 500, 1000]
      max_depth: [3, 5, 7, 10]
      learning_rate: [0.01, 0.1, 0.2]
```

Each trial is logged as a child run with:
- Trial parameters
- Trial metrics
- Trial duration
- Trial status (completed, failed, pruned)

## Artifact Management

### Custom Artifacts

Log additional artifacts in your custom components:

```python
import mlflow

# In your custom component
def execute(self, context):
    # Your processing logic
    result = process_data()
    
    # Log custom artifacts
    mlflow.log_artifact("custom_report.html")
    mlflow.log_dict(result, "results.json")
    mlflow.log_figure(plot, "custom_plot.png")
    
    return ExecutionResult(...)
```

### Artifact Organization

Artifacts are organized by type:

```
artifacts/
├── model/
│   ├── model.pkl
│   └── model_metadata.json
├── preprocessor/
│   └── preprocessor.pkl
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── reports/
│   ├── evaluation_report.json
│   └── data_quality_report.html
└── config/
    └── pipeline_config.yaml
```

## Comparing Experiments

### Using MLflow UI

1. Navigate to the MLflow UI
2. Select experiments to compare
3. Use the comparison view to analyze:
   - Parameter differences
   - Metric trends
   - Artifact differences

### Using MLflow API

```python
import mlflow

# Get experiment by name
experiment = mlflow.get_experiment_by_name("customer_churn")

# Search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.8",
    order_by=["metrics.accuracy DESC"]
)

# Compare top runs
top_runs = runs.head(5)
print(top_runs[['run_id', 'metrics.accuracy', 'params.model.n_estimators']])
```

## Production Deployment

### Model Serving

Deploy models directly from MLflow:

```bash
# Serve model locally
mlflow models serve \
  --model-uri models:/fraud_detection/Production \
  --port 5001

# Deploy to cloud (example with AWS SageMaker)
mlflow deployments create \
  --target sagemaker \
  --name fraud-detection-prod \
  --model-uri models:/fraud_detection/Production
```

### Model Loading

Load models in production code:

```python
import mlflow.pyfunc

# Load latest production model
model = mlflow.pyfunc.load_model("models:/fraud_detection/Production")

# Make predictions
predictions = model.predict(new_data)
```

## Monitoring and Alerts

### Model Performance Monitoring

Track model performance over time:

```python
# Log production metrics
with mlflow.start_run():
    mlflow.log_metric("production_accuracy", current_accuracy)
    mlflow.log_metric("data_drift_score", drift_score)
    mlflow.set_tag("environment", "production")
```

### Automated Alerts

Set up alerts for model degradation:

```python
# Example alert logic
if current_accuracy < baseline_accuracy * 0.95:
    mlflow.log_metric("alert_triggered", 1)
    send_alert("Model performance degraded")
```

## Best Practices

### Experiment Naming
- Use consistent naming conventions
- Include version numbers
- Add descriptive tags

### Parameter Logging
- Log all relevant parameters
- Include data preprocessing settings
- Track environment information

### Artifact Management
- Organize artifacts by type
- Include model metadata
- Store evaluation reports

### Model Registry
- Use semantic versioning
- Document model changes
- Maintain staging/production stages

### Performance
- Use batch logging for large experiments
- Configure appropriate artifact storage
- Monitor storage costs

## Troubleshooting

### Common Issues

**Connection Errors:**
```bash
# Check MLflow server status
curl http://localhost:5000/health

# Verify tracking URI
echo $MLFLOW_TRACKING_URI
```

**Storage Issues:**
```bash
# Check artifact storage permissions
aws s3 ls s3://ml-artifacts/

# Verify credentials
aws sts get-caller-identity
```

**Performance Issues:**
- Use local artifact storage for development
- Configure appropriate database backend
- Monitor disk space usage

### Debugging

Enable debug logging:

```yaml
logging:
  level: "DEBUG"
  file_path: "logs/mlflow_debug.log"
```

Check MLflow logs:
```bash
# MLflow server logs
tail -f mlflow_server.log

# Application logs
tail -f logs/pipeline.log
```

## Integration Examples

### Complete Training Pipeline

```yaml
pipeline:
  name: "recommendation_system_v1"
  description: "Collaborative filtering recommendation system"
  tags: ["recommendations", "collaborative_filtering", "v1"]
  mlflow_tracking_uri: "http://mlflow.company.com:5000"
  artifact_location: "s3://ml-artifacts/recommendations"

data:
  sources:
    - type: "sql"
      connection_string: "${DB_CONNECTION}"
      query: "SELECT * FROM user_interactions WHERE date >= '2023-01-01'"

model:
  type: "sklearn"
  parameters:
    algorithm: "NMF"
    n_components: 50
    random_state: 42
  
  hyperparameter_tuning:
    method: "optuna"
    n_trials: 50
    parameters:
      n_components: [20, 50, 100, 200]
      alpha: [0.0, 0.1, 0.5]
      l1_ratio: [0.0, 0.5, 1.0]

evaluation:
  metrics: ["rmse", "mae", "precision_at_k", "recall_at_k"]
  generate_plots: true
  plot_types: ["learning_curve", "validation_curve"]
```

This configuration will automatically:
1. Create an MLflow experiment named "recommendation_system_v1"
2. Log all parameters, metrics, and artifacts
3. Track hyperparameter optimization trials
4. Register the best model in the model registry
5. Generate and store evaluation plots

The MLflow UI will show all this information organized and searchable, making it easy to track progress and compare different approaches.