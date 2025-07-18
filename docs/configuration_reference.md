# Configuration Reference

This document provides a comprehensive reference for all configuration options available in the ML Pipeline framework.

## Configuration File Format

Configuration files can be written in YAML or JSON format. YAML is recommended for better readability.

### Environment Variable Substitution

The framework supports environment variable substitution in configuration files:

```yaml
# Using ${VAR_NAME} syntax with optional default values
data:
  sources:
    - type: "csv"
      path: "${DATA_PATH:/default/path/data.csv}"

# Using $VAR_NAME syntax
model:
  parameters:
    random_state: $RANDOM_SEED
```

## Configuration Schema

### Pipeline Section

The `pipeline` section contains metadata about your experiment.

```yaml
pipeline:
  name: string                    # Required: Experiment name
  description: string             # Optional: Experiment description
  tags: [string]                  # Optional: List of tags
  mlflow_tracking_uri: string     # Optional: MLflow tracking server URI
  artifact_location: string       # Optional: Artifact storage location
```

**Example:**
```yaml
pipeline:
  name: "customer_churn_prediction"
  description: "Predict customer churn using historical data"
  tags: ["classification", "churn", "production"]
  mlflow_tracking_uri: "http://mlflow.company.com:5000"
  artifact_location: "s3://ml-artifacts/experiments"
```

### Data Section

The `data` section configures data sources, preprocessing, and splitting.

#### Data Sources

```yaml
data:
  sources:
    - type: "csv" | "json" | "parquet" | "sql"
      path: string                # Required: Path to data file
      schema_path: string         # Optional: Path to schema file
      connection_string: string   # Required for SQL: Database connection
      table_name: string          # Optional for SQL: Table name
      query: string              # Optional for SQL: Custom query
```

**Supported Data Source Types:**

| Type | Description | Required Fields | Optional Fields |
|------|-------------|----------------|-----------------|
| `csv` | Comma-separated values | `path` | `schema_path` |
| `json` | JSON format | `path` | `schema_path` |
| `parquet` | Apache Parquet | `path` | `schema_path` |
| `sql` | SQL database | `connection_string` | `table_name`, `query` |

**Examples:**

```yaml
# CSV file
- type: "csv"
  path: "data/customers.csv"
  schema_path: "schemas/customer_schema.json"

# SQL database
- type: "sql"
  connection_string: "postgresql://user:pass@localhost:5432/db"
  table_name: "customers"
  # OR use custom query
  query: "SELECT * FROM customers WHERE created_date > '2023-01-01'"

# Multiple sources
sources:
  - type: "csv"
    path: "data/features.csv"
  - type: "parquet"
    path: "data/additional_features.parquet"
```

#### Preprocessing Steps

```yaml
data:
  preprocessing:
    - type: string              # Required: Preprocessing step type
      columns: [string]         # Optional: Columns to apply to
      parameters: object        # Optional: Step-specific parameters
```

**Available Preprocessing Types:**

| Type | Description | Parameters |
|------|-------------|------------|
| `standard_scaler` | Standardize features | `with_mean`, `with_std` |
| `robust_scaler` | Scale using robust statistics | `quantile_range` |
| `min_max_scaler` | Scale to [0,1] range | `feature_range` |
| `label_encoder` | Encode categorical labels | None |
| `one_hot_encoder` | One-hot encode categories | `drop`, `sparse` |
| `target_encoder` | Target-based encoding | `smoothing` |
| `polynomial_features` | Generate polynomial features | `degree`, `include_bias` |
| `missing_value_imputer` | Impute missing values | `strategy`, `fill_value` |

**Examples:**

```yaml
preprocessing:
  # Standardize numerical features
  - type: "standard_scaler"
    columns: ["age", "income", "score"]
  
  # One-hot encode categorical features
  - type: "one_hot_encoder"
    columns: ["category", "region"]
    parameters:
      drop: "first"
      sparse: false
  
  # Impute missing values
  - type: "missing_value_imputer"
    columns: ["optional_field"]
    parameters:
      strategy: "median"
```

#### Data Splitting

```yaml
data:
  train_split: float            # Training data ratio (0.1-0.9)
  validation_split: float       # Validation data ratio (0.05-0.4)
  test_split: float            # Test data ratio (0.05-0.4)
  stratify: boolean            # Whether to stratify splits
  random_state: integer        # Random seed for reproducibility
```

**Note:** The three split ratios must sum to 1.0.

### Model Section

The `model` section configures the machine learning model and training parameters.

```yaml
model:
  type: "sklearn" | "xgboost" | "pytorch" | "huggingface"
  parameters: object            # Model-specific parameters
  hyperparameter_tuning: object # Optional: Hyperparameter optimization
  early_stopping: boolean       # Enable early stopping
  early_stopping_patience: integer # Early stopping patience
```

#### Model Types and Parameters

**Scikit-learn (`sklearn`):**
```yaml
model:
  type: "sklearn"
  parameters:
    algorithm: string           # Algorithm name (e.g., "RandomForestClassifier")
    # Algorithm-specific parameters
    n_estimators: 100
    max_depth: 10
    random_state: 42
```

**XGBoost (`xgboost`):**
```yaml
model:
  type: "xgboost"
  parameters:
    objective: string           # Objective function
    n_estimators: 1000
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
```

**PyTorch (`pytorch`):**
```yaml
model:
  type: "pytorch"
  parameters:
    model_class: string         # Custom model class name
    hidden_layers: [128, 64]    # Hidden layer sizes
    dropout: 0.2
    learning_rate: 0.001
    batch_size: 32
    epochs: 100
    device: "auto"              # "cpu", "cuda", or "auto"
```

**Hugging Face (`huggingface`):**
```yaml
model:
  type: "huggingface"
  parameters:
    model_name: string          # Pre-trained model name
    task: string               # Task type
    num_labels: integer        # Number of output labels
    learning_rate: 2e-5
    num_train_epochs: 3
    per_device_train_batch_size: 8
    device: "auto"
```

#### Hyperparameter Tuning

```yaml
model:
  hyperparameter_tuning:
    method: "grid_search" | "random_search" | "optuna"
    n_trials: integer           # Number of trials (for optuna/random_search)
    timeout: integer           # Timeout in seconds
    parameters: object         # Parameter search space
    cv_folds: integer         # Cross-validation folds
    scoring: string           # Scoring metric
```

**Parameter Search Space Examples:**

```yaml
# Grid search - all combinations
parameters:
  n_estimators: [50, 100, 200]
  max_depth: [5, 10, 15]

# Random search / Optuna - sample from ranges
parameters:
  n_estimators: [50, 100, 200, 300]
  max_depth: [3, 4, 5, 6, 7, 8, 9, 10]
  learning_rate: [0.01, 0.05, 0.1, 0.2]
```

### Evaluation Section

```yaml
evaluation:
  metrics: [string]             # List of evaluation metrics
  cross_validation: boolean     # Enable cross-validation
  cv_folds: integer            # Number of CV folds
  stratify: boolean            # Stratify CV splits
  generate_plots: boolean      # Generate evaluation plots
  plot_types: [string]         # Types of plots to generate
```

**Available Metrics:**

**Classification:**
- `accuracy` - Classification accuracy
- `precision` - Precision score
- `recall` - Recall score
- `f1_score` - F1 score
- `f1_macro` - Macro-averaged F1
- `f1_micro` - Micro-averaged F1
- `roc_auc` - ROC AUC score
- `precision_recall_auc` - Precision-Recall AUC

**Regression:**
- `mse` - Mean Squared Error
- `mae` - Mean Absolute Error
- `rmse` - Root Mean Squared Error
- `r2_score` - R² Score
- `mape` - Mean Absolute Percentage Error

**Available Plot Types:**
- `confusion_matrix` - Confusion matrix heatmap
- `roc_curve` - ROC curve
- `precision_recall_curve` - Precision-Recall curve
- `feature_importance` - Feature importance plot
- `residuals` - Residual plots (regression)
- `prediction_vs_actual` - Prediction vs actual scatter plot
- `learning_curve` - Learning curve
- `classification_report` - Classification report

### Drift Detection Section

```yaml
drift_detection:
  enabled: boolean              # Enable drift detection
  baseline_data: string         # Path to baseline data
  methods: [string]            # Drift detection methods
  thresholds: object           # Drift thresholds
  monitoring_window: integer   # Monitoring window size
  alert_channels: [string]     # Alert notification channels
```

**Drift Detection Methods:**
- `evidently` - Evidently AI drift detection
- `kl_divergence` - KL divergence
- `psi` - Population Stability Index
- `wasserstein` - Wasserstein distance

**Thresholds:**
```yaml
thresholds:
  data_drift: 0.1              # Overall data drift threshold
  prediction_drift: 0.05       # Prediction drift threshold
  feature_drift: 0.1           # Individual feature drift threshold
```

**Alert Channels:**
```yaml
alert_channels:
  - "email:alerts@company.com"
  - "slack:#ml-alerts"
  - "webhook:https://hooks.company.com/ml-alerts"
```

### Few-Shot Learning Section

```yaml
few_shot:
  enabled: boolean              # Enable few-shot learning
  prompt_template: string       # Path to prompt template
  max_examples: integer        # Maximum examples to include
  similarity_threshold: float  # Similarity threshold
  embedding_model: string      # Model for embeddings
  example_store_path: string   # Path to example store
```

### Logging Section

```yaml
logging:
  level: "DEBUG" | "INFO" | "WARNING" | "ERROR"
  format: string               # Log format string
  file_path: string           # Log file path
  max_file_size: string       # Maximum log file size
  backup_count: integer       # Number of backup files
```

## Complete Configuration Example

```yaml
# Complete configuration example
pipeline:
  name: "production_model_v2"
  description: "Production model with full monitoring"
  tags: ["production", "v2", "monitored"]
  mlflow_tracking_uri: "http://mlflow.company.com:5000"

data:
  sources:
    - type: "sql"
      connection_string: "${DB_CONNECTION_STRING}"
      query: "SELECT * FROM features WHERE date >= '2023-01-01'"
  
  preprocessing:
    - type: "standard_scaler"
      columns: ["numerical_feature_1", "numerical_feature_2"]
    - type: "one_hot_encoder"
      columns: ["categorical_feature"]
      parameters:
        drop: "first"
    - type: "missing_value_imputer"
      columns: ["optional_feature"]
      parameters:
        strategy: "median"
  
  train_split: 0.7
  validation_split: 0.15
  test_split: 0.15
  stratify: true
  random_state: 42

model:
  type: "xgboost"
  parameters:
    objective: "binary:logistic"
    n_estimators: 1000
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
  
  hyperparameter_tuning:
    method: "optuna"
    n_trials: 100
    timeout: 3600
    parameters:
      n_estimators: [500, 1000, 1500]
      max_depth: [4, 5, 6, 7, 8]
      learning_rate: [0.05, 0.1, 0.15, 0.2]
    cv_folds: 5
    scoring: "f1"
  
  early_stopping: true
  early_stopping_patience: 50

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
  cross_validation: true
  cv_folds: 5
  stratify: true
  generate_plots: true
  plot_types: ["confusion_matrix", "roc_curve", "feature_importance"]

drift_detection:
  enabled: true
  baseline_data: "data/baseline.csv"
  methods: ["evidently", "wasserstein"]
  thresholds:
    data_drift: 0.1
    prediction_drift: 0.05
    feature_drift: 0.15
  monitoring_window: 1000
  alert_channels:
    - "email:ml-team@company.com"
    - "slack:#ml-alerts"

few_shot:
  enabled: false

logging:
  level: "INFO"
  file_path: "logs/production_model.log"
  max_file_size: "100MB"
  backup_count: 10
```

## Validation

Use the CLI to validate your configuration:

```bash
mlpipeline validate --config your_config.yaml
```

The validator will check:
- Required fields are present
- Data types are correct
- Value ranges are valid
- Cross-field consistency
- File paths exist (where applicable)

## Best Practices

1. **Use Environment Variables**: Store sensitive information like database credentials in environment variables
2. **Version Control**: Keep configuration files in version control
3. **Validate Early**: Always validate configurations before running experiments
4. **Document Changes**: Use descriptive experiment names and tags
5. **Monitor Resources**: Set appropriate timeouts for long-running operations
6. **Test Configurations**: Use smaller datasets for testing configuration changes