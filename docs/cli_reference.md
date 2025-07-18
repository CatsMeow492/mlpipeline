# CLI Reference Guide

This document provides a comprehensive reference for all command-line interface (CLI) commands available in the ML Pipeline framework.

## Global Options

All commands support these global options:

```bash
mlpipeline [GLOBAL_OPTIONS] <command> [COMMAND_OPTIONS]
```

**Global Options:**
- `--verbose, -v`: Enable verbose logging output
- `--log-file PATH`: Specify custom log file path
- `--help`: Show help message and exit

## Commands Overview

| Command | Description |
|---------|-------------|
| `train` | Train a machine learning model |
| `inference` | Perform inference with a trained model |
| `evaluate` | Evaluate model performance on test data |
| `monitor` | Monitor for data and model drift |
| `init` | Create configuration template |
| `validate` | Validate configuration file |
| `status` | Show system and component status |
| `experiments` | List and manage experiments |
| `progress` | Monitor experiment progress |
| `analyze` | Analyze configuration and provide insights |

## Command Details

### train

Train a machine learning model using the specified configuration.

```bash
mlpipeline train [OPTIONS]
```

**Options:**
- `--config, -c PATH`: Path to pipeline configuration file
- `--experiment-id TEXT`: Custom experiment ID
- `--artifacts-path PATH`: Path to store artifacts (default: ./artifacts)
- `--resume`: Resume from checkpoint if available
- `--dry-run`: Validate configuration without execution
- `--interactive, -i`: Run in interactive mode

**Examples:**

```bash
# Basic training
mlpipeline train --config my_config.yaml

# Training with custom experiment ID
mlpipeline train --config my_config.yaml --experiment-id "exp_2024_01_15"

# Interactive training setup
mlpipeline train --interactive

# Dry run to validate configuration
mlpipeline train --config my_config.yaml --dry-run

# Resume from checkpoint
mlpipeline train --config my_config.yaml --resume
```

**Interactive Mode:**
When using `--interactive`, the CLI will guide you through:
1. Pipeline name and description
2. Model type selection (sklearn, xgboost, pytorch, huggingface)
3. Task type (classification, regression, few-shot)
4. Data file paths
5. Advanced options (hyperparameter tuning, drift detection)

### inference

Perform inference using a trained model on new data.

```bash
mlpipeline inference [OPTIONS]
```

**Options:**
- `--config, -c PATH`: Path to pipeline configuration file (required)
- `--model-path PATH`: Path to trained model (required)
- `--input-data PATH`: Path to input data for inference (required)
- `--output-path PATH`: Path to save predictions (default: ./predictions.json)
- `--batch-size INTEGER`: Batch size for processing (default: 1000)
- `--confidence-threshold FLOAT`: Confidence threshold for predictions

**Examples:**

```bash
# Basic inference
mlpipeline inference \
  --config my_config.yaml \
  --model-path artifacts/model.pkl \
  --input-data data/new_data.csv

# Inference with custom output and batch size
mlpipeline inference \
  --config my_config.yaml \
  --model-path artifacts/model.pkl \
  --input-data data/large_dataset.csv \
  --output-path results/predictions.json \
  --batch-size 500

# Inference with confidence filtering
mlpipeline inference \
  --config my_config.yaml \
  --model-path artifacts/model.pkl \
  --input-data data/uncertain_cases.csv \
  --confidence-threshold 0.8
```

### evaluate

Evaluate a trained model's performance on test data.

```bash
mlpipeline evaluate [OPTIONS]
```

**Options:**
- `--config, -c PATH`: Path to pipeline configuration file (required)
- `--model-path PATH`: Path to trained model (required)
- `--test-data PATH`: Path to test data (required)
- `--output-path PATH`: Path to save evaluation results (default: ./evaluation_results.json)
- `--metrics TEXT`: Specific metrics to compute (can be used multiple times)

**Examples:**

```bash
# Basic evaluation
mlpipeline evaluate \
  --config my_config.yaml \
  --model-path artifacts/model.pkl \
  --test-data data/test.csv

# Evaluation with specific metrics
mlpipeline evaluate \
  --config my_config.yaml \
  --model-path artifacts/model.pkl \
  --test-data data/test.csv \
  --metrics accuracy \
  --metrics f1_score \
  --metrics precision

# Evaluation with custom output path
mlpipeline evaluate \
  --config my_config.yaml \
  --model-path artifacts/model.pkl \
  --test-data data/test.csv \
  --output-path reports/model_evaluation.json
```

### monitor

Monitor for data and model drift in production data.

```bash
mlpipeline monitor [OPTIONS]
```

**Options:**
- `--config, -c PATH`: Path to pipeline configuration file (required)
- `--baseline-data PATH`: Path to baseline data for drift detection
- `--current-data PATH`: Path to current data to check for drift (required)
- `--output-path PATH`: Path to save drift report (default: ./drift_report.json)
- `--threshold FLOAT`: Custom drift detection threshold

**Examples:**

```bash
# Basic drift monitoring
mlpipeline monitor \
  --config my_config.yaml \
  --current-data data/production_data.csv

# Drift monitoring with custom baseline
mlpipeline monitor \
  --config my_config.yaml \
  --baseline-data data/training_baseline.csv \
  --current-data data/production_data.csv

# Drift monitoring with custom threshold
mlpipeline monitor \
  --config my_config.yaml \
  --current-data data/production_data.csv \
  --threshold 0.15 \
  --output-path reports/drift_$(date +%Y%m%d).json
```

### init

Create a new pipeline configuration template.

```bash
mlpipeline init [OPTIONS]
```

**Options:**
- `--output, -o PATH`: Output path for configuration template (default: ./config_template.yaml)
- `--format CHOICE`: Output format (yaml|json, default: yaml)
- `--use-case CHOICE`: Use case template (classification|regression|few-shot, default: classification)

**Examples:**

```bash
# Create basic classification template
mlpipeline init --output my_project.yaml

# Create regression template
mlpipeline init --output regression_config.yaml --use-case regression

# Create few-shot learning template
mlpipeline init --output few_shot_config.yaml --use-case few-shot

# Create JSON format template
mlpipeline init --output config.json --format json
```

### validate

Validate a pipeline configuration file for correctness.

```bash
mlpipeline validate [OPTIONS]
```

**Options:**
- `--config, -c PATH`: Path to pipeline configuration file to validate (required)

**Examples:**

```bash
# Validate configuration
mlpipeline validate --config my_config.yaml

# Validate with verbose output
mlpipeline validate --config my_config.yaml --verbose
```

**Validation Checks:**
- Required fields presence
- Data type correctness
- Value range validation
- Cross-field consistency
- File path existence (where applicable)

### status

Show pipeline status and system information.

```bash
mlpipeline status [OPTIONS]
```

**Options:**
- `--detailed`: Show detailed component and dependency information

**Examples:**

```bash
# Basic status
mlpipeline status

# Detailed status with dependency versions
mlpipeline status --detailed
```

**Status Information:**
- Registered components
- System information (Python version, platform)
- Key dependency versions (when --detailed)
- Available commands summary

### experiments

List and manage ML experiments.

```bash
mlpipeline experiments [OPTIONS]
```

**Options:**
- `--experiment-id TEXT`: Filter by specific experiment ID
- `--limit INTEGER`: Limit number of results (default: 10)
- `--format CHOICE`: Output format (table|json, default: table)
- `--sort-by CHOICE`: Sort experiments by field (created|name|accuracy, default: created)
- `--status CHOICE`: Filter by experiment status (running|completed|failed)

**Examples:**

```bash
# List recent experiments
mlpipeline experiments

# List experiments with specific status
mlpipeline experiments --status completed --limit 20

# Get specific experiment details
mlpipeline experiments --experiment-id exp_123

# Export experiments to JSON
mlpipeline experiments --format json --limit 50 > experiments.json

# Sort by performance
mlpipeline experiments --sort-by accuracy --limit 5
```

### progress

Monitor the progress of running experiments.

```bash
mlpipeline progress [OPTIONS]
```

**Options:**
- `--experiment-id TEXT`: Experiment ID to monitor (required)
- `--follow, -f`: Follow progress in real-time
- `--refresh-interval INTEGER`: Refresh interval in seconds for follow mode (default: 5)

**Examples:**

```bash
# Check current progress
mlpipeline progress --experiment-id exp_123

# Follow progress in real-time
mlpipeline progress --experiment-id exp_123 --follow

# Follow with custom refresh rate
mlpipeline progress --experiment-id exp_123 --follow --refresh-interval 10
```

**Progress Information:**
- Stage completion status
- Current progress percentages
- Estimated time remaining
- Error information (if any)

### analyze

Analyze pipeline configuration and provide insights.

```bash
mlpipeline analyze [OPTIONS]
```

**Options:**
- `--config, -c PATH`: Configuration file to analyze (required)
- `--check-data`: Check if data files exist and show sizes
- `--estimate-time`: Estimate training time based on configuration
- `--suggest-improvements`: Suggest configuration improvements

**Examples:**

```bash
# Basic configuration analysis
mlpipeline analyze --config my_config.yaml

# Comprehensive analysis
mlpipeline analyze --config my_config.yaml \
  --check-data \
  --estimate-time \
  --suggest-improvements

# Check data availability only
mlpipeline analyze --config my_config.yaml --check-data
```

**Analysis Features:**
- Configuration complexity assessment
- Data file validation and size reporting
- Training time estimation
- Performance optimization suggestions
- Best practice recommendations

## Common Usage Patterns

### Development Workflow

```bash
# 1. Create initial configuration
mlpipeline init --output dev_config.yaml --use-case classification

# 2. Validate configuration
mlpipeline validate --config dev_config.yaml

# 3. Analyze configuration
mlpipeline analyze --config dev_config.yaml --check-data --suggest-improvements

# 4. Test with dry run
mlpipeline train --config dev_config.yaml --dry-run

# 5. Train model
mlpipeline train --config dev_config.yaml

# 6. Evaluate results
mlpipeline evaluate --config dev_config.yaml \
  --model-path artifacts/model.pkl \
  --test-data data/test.csv
```

### Production Monitoring

```bash
# Daily drift monitoring
mlpipeline monitor --config prod_config.yaml \
  --current-data /prod/data/daily_$(date +%Y%m%d).csv \
  --output-path /prod/reports/drift_$(date +%Y%m%d).json

# Weekly model retraining
mlpipeline train --config prod_config.yaml \
  --experiment-id "weekly_retrain_$(date +%Y%m%d)"

# Batch inference
mlpipeline inference --config prod_config.yaml \
  --model-path /prod/models/latest_model.pkl \
  --input-data /prod/data/batch_input.csv \
  --output-path /prod/predictions/batch_$(date +%Y%m%d).json
```

### Experiment Management

```bash
# List recent experiments
mlpipeline experiments --limit 20

# Monitor running experiment
mlpipeline progress --experiment-id exp_$(date +%Y%m%d) --follow

# Compare experiment results
mlpipeline experiments --sort-by accuracy --format json > results.json
```

## Environment Variables

The CLI supports several environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `MLPIPELINE_CONFIG_PATH` | Default configuration file path | None |
| `MLPIPELINE_LOG_LEVEL` | Default logging level | INFO |
| `MLPIPELINE_ARTIFACTS_PATH` | Default artifacts directory | ./artifacts |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | file:./mlruns |

**Example:**
```bash
export MLPIPELINE_CONFIG_PATH="configs/production.yaml"
export MLFLOW_TRACKING_URI="http://mlflow.company.com:5000"
mlpipeline train  # Uses environment variables
```

## Exit Codes

The CLI uses standard exit codes:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Data error |
| 4 | Model error |
| 5 | System error |

## Tips and Best Practices

1. **Always validate configurations** before running expensive operations
2. **Use dry-run mode** to test configurations without execution
3. **Enable verbose logging** for debugging: `--verbose`
4. **Use meaningful experiment IDs** for better tracking
5. **Set up environment variables** for production deployments
6. **Monitor long-running experiments** with the progress command
7. **Analyze configurations** before training to catch issues early
8. **Use interactive mode** for learning and quick setups

## Troubleshooting

### Common CLI Issues

**Command not found:**
```bash
# Ensure mlpipeline is installed and in PATH
pip install mlpipeline
which mlpipeline
```

**Permission errors:**
```bash
# Check file permissions
ls -la config.yaml
chmod 644 config.yaml
```

**Memory errors during training:**
```bash
# Use smaller batch sizes or enable distributed processing
mlpipeline train --config config.yaml --verbose
```

**Configuration validation failures:**
```bash
# Get detailed validation errors
mlpipeline validate --config config.yaml --verbose
```

For more help, use `mlpipeline <command> --help` or refer to the configuration reference documentation.