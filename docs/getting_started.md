# Getting Started with ML Pipeline

This guide will help you get started with the ML Pipeline framework, from installation to running your first experiment.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install ML Pipeline

```bash
pip install mlpipeline
```

For development installation:

```bash
git clone https://github.com/your-org/mlpipeline.git
cd mlpipeline
pip install -e .
```

### Optional Dependencies

For distributed computing:
```bash
pip install mlpipeline[distributed]
```

For GPU support:
```bash
pip install mlpipeline[gpu]
```

## Quick Start

### 1. Create a Configuration File

Generate a basic configuration template:

```bash
mlpipeline init --output my_config.yaml --use-case classification
```

This creates a YAML configuration file with sensible defaults for a classification task.

### 2. Prepare Your Data

Organize your data files:
```
data/
├── train.csv
├── test.csv
└── validation.csv
```

Your CSV files should have:
- Feature columns (numerical or categorical)
- A target column for supervised learning

### 3. Configure Your Pipeline

Edit the generated `my_config.yaml` file:

```yaml
pipeline:
  name: "my_first_experiment"
  description: "My first ML pipeline experiment"

data:
  sources:
    - type: "csv"
      path: "data/train.csv"
  
  preprocessing:
    - type: "standard_scaler"
      columns: ["feature1", "feature2", "feature3"]
    - type: "label_encoder"
      columns: ["target"]

model:
  type: "sklearn"
  parameters:
    algorithm: "RandomForestClassifier"
    n_estimators: 100
    random_state: 42

evaluation:
  metrics: ["accuracy", "f1_score", "precision", "recall"]
```

### 4. Validate Your Configuration

Before running, validate your configuration:

```bash
mlpipeline validate --config my_config.yaml
```

### 5. Train Your Model

Run the training pipeline:

```bash
mlpipeline train --config my_config.yaml
```

This will:
- Load and preprocess your data
- Train the specified model
- Evaluate performance
- Save artifacts and logs

### 6. Monitor Progress

The CLI provides real-time progress updates and will display:
- Configuration validation results
- Training progress
- Performance metrics
- Artifact locations

## Example Workflows

### Basic Classification

```bash
# Create configuration
mlpipeline init --output classification.yaml --use-case classification

# Edit configuration file (add your data paths)
# ...

# Validate configuration
mlpipeline validate --config classification.yaml

# Train model
mlpipeline train --config classification.yaml

# Evaluate on test data
mlpipeline evaluate --config classification.yaml \
  --model-path artifacts/model.pkl \
  --test-data data/test.csv
```

### Regression with Hyperparameter Tuning

```bash
# Create configuration
mlpipeline init --output regression.yaml --use-case regression

# Edit configuration to enable hyperparameter tuning
# Set model.hyperparameter_tuning.enabled = true

# Train with optimization
mlpipeline train --config regression.yaml
```

### Few-Shot Learning

```bash
# Create configuration
mlpipeline init --output few_shot.yaml --use-case few-shot

# Edit configuration to specify prompt template and examples

# Train few-shot model
mlpipeline train --config few_shot.yaml

# Run inference
mlpipeline inference --config few_shot.yaml \
  --model-path artifacts/model \
  --input-data data/new_examples.json
```

## Configuration Examples

The framework includes several example configurations:

- `examples/configs/classification_basic.yaml` - Basic classification
- `examples/configs/regression_advanced.yaml` - Advanced regression with drift detection
- `examples/configs/few_shot_learning.yaml` - Few-shot learning setup

## Next Steps

1. **Explore Advanced Features**: Learn about drift detection, hyperparameter optimization, and few-shot learning
2. **Customize Components**: Create custom preprocessing steps and model adapters
3. **Set Up Monitoring**: Configure drift detection and alerting
4. **Scale Up**: Use distributed computing for large datasets

## Common Issues

### Data Loading Errors
- Ensure file paths are correct and accessible
- Check data format matches the specified type (CSV, JSON, Parquet)
- Verify column names match preprocessing configuration

### Model Training Failures
- Check that model parameters are valid for the chosen algorithm
- Ensure sufficient memory for large datasets
- Verify target column format for supervised learning

### Configuration Validation Errors
- Use `mlpipeline validate` to identify specific issues
- Check that required fields are present
- Ensure data splits sum to 1.0

## Getting Help

- Run `mlpipeline --help` for command-line help
- Use `mlpipeline status` to check system information
- Check the logs directory for detailed error information
- Refer to the configuration reference documentation

## Interactive Configuration

For beginners, use the interactive configuration wizard:

```bash
mlpipeline train --interactive
```

This will guide you through:
- Choosing model type and task
- Specifying data paths
- Configuring preprocessing steps
- Setting up evaluation metrics
- Enabling advanced features

## Advanced Features

### Hyperparameter Optimization

Enable automatic hyperparameter tuning in your configuration:

```yaml
model:
  hyperparameter_tuning:
    method: "optuna"
    n_trials: 50
    parameters:
      n_estimators: [50, 100, 200]
      max_depth: [5, 10, 15, 20]
```

### Drift Detection

Monitor your model in production:

```yaml
drift_detection:
  enabled: true
  baseline_data: "data/training_baseline.csv"
  thresholds:
    data_drift: 0.1
    prediction_drift: 0.05
  alert_channels: ["email:team@company.com"]
```

### Progress Monitoring

Monitor long-running experiments:

```bash
# Check progress of a running experiment
mlpipeline progress --experiment-id exp_123 --follow

# Analyze configuration before running
mlpipeline analyze --config my_config.yaml --check-data --estimate-time
```

## Configuration Examples by Use Case

The framework includes several ready-to-use configuration examples:

| Use Case | Configuration File | Description |
|----------|-------------------|-------------|
| Basic Classification | `examples/configs/classification_basic.yaml` | Simple binary/multi-class classification |
| Advanced Regression | `examples/configs/regression_advanced.yaml` | Regression with drift detection |
| Deep Learning | `examples/configs/deep_learning_pytorch.yaml` | PyTorch neural networks |
| NLP/Text Analysis | `examples/configs/nlp_sentiment_analysis.yaml` | Text classification with BERT |
| Few-Shot Learning | `examples/configs/few_shot_learning.yaml` | Few-shot learning setup |
| Time Series | `examples/configs/time_series_forecasting.yaml` | Time series forecasting |
| Minimal Setup | `examples/configs/minimal_example.yaml` | Simplest possible configuration |

Copy and modify these examples for your specific needs:

```bash
cp examples/configs/classification_basic.yaml my_project_config.yaml
# Edit my_project_config.yaml with your data paths and parameters
```

## Sample Data

Create synthetic data for testing:

```python
from sklearn.datasets import make_classification, make_regression
import pandas as pd
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Classification data
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_classes=2, 
    n_informative=8,
    n_redundant=2,
    random_state=42
)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y
df.to_csv('data/sample_classification.csv', index=False)

# Regression data
X, y = make_regression(
    n_samples=1000, 
    n_features=10, 
    noise=0.1, 
    random_state=42
)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y
df.to_csv('data/sample_regression.csv', index=False)

print("Sample data created in data/ directory")
```

## Troubleshooting Guide

### Common Configuration Issues

**Problem**: "Configuration validation failed"
```bash
# Solution: Use the validate command to see specific errors
mlpipeline validate --config my_config.yaml
```

**Problem**: "Data file not found"
```bash
# Solution: Check file paths and permissions
ls -la data/
# Ensure paths in config match actual file locations
```

**Problem**: "Model training fails with memory error"
```yaml
# Solution: Reduce batch size or use data chunking
model:
  parameters:
    batch_size: 16  # Reduce from 32 or 64
```

### Performance Optimization

**For Large Datasets:**
```yaml
# Enable data chunking
data:
  chunk_size: 10000
  
# Use distributed processing
distributed:
  enabled: true
  n_workers: 4
```

**For Slow Training:**
```yaml
# Enable early stopping
model:
  early_stopping: true
  early_stopping_patience: 10

# Reduce hyperparameter search space
model:
  hyperparameter_tuning:
    n_trials: 20  # Reduce from 100
```

### Debugging Tips

1. **Start with minimal configuration**: Use `examples/configs/minimal_example.yaml` as a base
2. **Enable verbose logging**: Add `--verbose` flag to CLI commands
3. **Check system status**: Run `mlpipeline status --detailed`
4. **Validate step by step**: Use `--dry-run` flag to test without execution
5. **Monitor resources**: Check memory and disk usage during training

## Production Deployment

### Environment Setup

```bash
# Set environment variables for production
export MLFLOW_TRACKING_URI="http://mlflow.company.com:5000"
export DATA_PATH="/prod/data"
export MODEL_REGISTRY_URI="s3://ml-models"
```

### Automated Pipelines

```bash
# Schedule regular retraining
crontab -e
# Add: 0 2 * * 0 /path/to/mlpipeline train --config /path/to/prod_config.yaml

# Set up drift monitoring
mlpipeline monitor --config prod_config.yaml \
  --current-data /prod/data/latest.csv \
  --output-path /prod/reports/drift_$(date +%Y%m%d).json
```

### Model Serving

```bash
# Export trained model for serving
mlpipeline inference --config prod_config.yaml \
  --model-path artifacts/best_model.pkl \
  --input-data /prod/data/batch_input.csv \
  --output-path /prod/predictions/batch_output.json
```

## Getting Help

- **CLI Help**: `mlpipeline --help` or `mlpipeline <command> --help`
- **Configuration Validation**: `mlpipeline validate --config <file>`
- **System Information**: `mlpipeline status --detailed`
- **Configuration Analysis**: `mlpipeline analyze --config <file> --suggest-improvements`
- **Log Files**: Check the `logs/` directory for detailed execution logs
- **Example Configurations**: Browse `examples/configs/` for templates

## Next Steps

1. **Try the Examples**: Start with `examples/configs/minimal_example.yaml`
2. **Read the Configuration Reference**: See `docs/configuration_reference.md`
3. **Explore Advanced Features**: Learn about drift detection and few-shot learning
4. **Join the Community**: Contribute examples and improvements
5. **Scale Up**: Use distributed computing for production workloads

This comprehensive guide should get you started with the ML Pipeline framework. The combination of simple examples and advanced features makes it suitable for both beginners and experienced practitioners.