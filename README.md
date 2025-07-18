# ML Pipeline Framework

A comprehensive, production-ready machine learning pipeline framework built with open source tools. Designed for scalable ML workflows with advanced error handling, drift detection, few-shot learning capabilities, and enterprise-grade deployment options.

## 🚀 Features

### Core Pipeline Orchestration
- **Advanced Pipeline Execution**: Robust orchestrator with dependency resolution, parallel execution, and comprehensive error handling
- **Component Registry**: Dynamic component loading and management system with custom component support
- **Structured Logging**: Correlation ID tracking and detailed execution logs with configurable levels
- **Checkpointing & Recovery**: Resume pipeline execution from failure points with automatic state preservation
- **Interactive Configuration**: Guided setup with intelligent defaults and validation

### 📊 Data Management
- **Multi-Source Support**: CSV, JSON, Parquet, and SQL database connectors with schema validation
- **Advanced Preprocessing**: Scikit-learn based transformations with metadata tracking and pipeline caching
- **Data Validation**: Schema validation, type checking, and data quality assessments
- **Version Control**: DVC integration for data versioning and lineage tracking
- **Environment Variable Support**: Flexible configuration with `${VAR:default}` syntax

### 🤖 Model Training & Evaluation
- **Multi-Framework Support**: Scikit-learn, XGBoost, PyTorch, and Hugging Face Transformers
- **Hyperparameter Optimization**: Optuna integration with grid search, random search, and Bayesian optimization
- **Comprehensive Evaluation**: Standard metrics, visualizations, model comparison, and cross-validation
- **Experiment Tracking**: MLflow integration for parameter, metric, and artifact logging
- **Early Stopping**: Configurable early stopping with patience and performance monitoring

### 🔍 Monitoring & Drift Detection
- **Evidently AI Integration**: Data and prediction drift detection with automated reports
- **Multiple Algorithms**: KL divergence, PSI, Wasserstein distance, and statistical tests
- **Alerting System**: Configurable thresholds with email, Slack, and webhook notifications
- **Real-time Monitoring**: Continuous monitoring with customizable windows and alert suppression

### 🎯 Few-Shot Learning
- **Prompt Management**: Template system with variable substitution and versioning
- **Example Store**: Similarity-based example selection using sentence embeddings
- **LLM Integration**: Hugging Face transformers and OpenAI-compatible APIs
- **Context Injection**: Automatic few-shot example context generation with similarity thresholds

### ⚡ Inference & Deployment
- **Batch Processing**: Scalable batch inference with progress tracking and chunking
- **Real-time API**: Fast inference endpoints with confidence scoring and validation
- **Model Validation**: Compatibility checks and preprocessing consistency verification
- **Distributed Computing**: Dask and Ray support for horizontal scaling

### 🐳 Container & Deployment Support
- **Multi-stage Docker Builds**: Optimized production, GPU, and development images
- **Docker Compose Orchestration**: Complete stack deployment with monitoring
- **Kubernetes Ready**: Helm charts and deployment configurations
- **GPU Support**: NVIDIA Docker runtime integration with CUDA optimization
- **Monitoring Stack**: Prometheus, Grafana, and custom metrics collection

## 📦 Installation

### Quick Install

```bash
# Basic installation
pip install mlpipeline

# With all optional dependencies
pip install mlpipeline[distributed,gpu,dev]
```

### Development Installation

```bash
git clone https://github.com/your-org/mlpipeline.git
cd mlpipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev]
```

### Optional Dependencies

```bash
# For distributed computing (Dask + Ray)
pip install mlpipeline[distributed]

# For GPU support (CUDA PyTorch)
pip install mlpipeline[gpu]

# For development tools
pip install mlpipeline[dev]
```

## 🚀 Quick Start

### 1. Initialize Configuration

```bash
# Create basic classification template
mlpipeline init --output my_config.yaml --use-case classification

# Create regression template
mlpipeline init --output regression.yaml --use-case regression

# Create few-shot learning template
mlpipeline init --output few_shot.yaml --use-case few-shot

# Interactive configuration setup
mlpipeline train --interactive
```

### 2. Validate Configuration

```bash
# Validate configuration
mlpipeline validate --config my_config.yaml

# Analyze configuration and get suggestions
mlpipeline analyze --config my_config.yaml --check-data --suggest-improvements
```

### 3. Train Model

```bash
# Basic training
mlpipeline train --config my_config.yaml

# Training with custom experiment ID
mlpipeline train --config my_config.yaml --experiment-id "exp_2024_01_15"

# Dry run (validate without execution)
mlpipeline train --config my_config.yaml --dry-run

# Resume from checkpoint
mlpipeline train --config my_config.yaml --resume
```

### 4. Monitor Progress

```bash
# Monitor running experiment
mlpipeline progress --experiment-id exp_2024_01_15 --follow

# List recent experiments
mlpipeline experiments --limit 10 --sort-by accuracy

# Check system status
mlpipeline status --detailed
```

## 📋 Configuration Reference

### Basic Configuration Structure

```yaml
pipeline:
  name: "my_experiment"
  description: "Example ML pipeline"
  tags: ["classification", "production"]
  mlflow_tracking_uri: "http://localhost:5000"

data:
  sources:
    - type: csv
      path: "${DATA_PATH:/default/path/data.csv}"
      schema_path: "schemas/data_schema.json"
  
  preprocessing:
    - type: standard_scaler
      columns: ["feature1", "feature2"]
    - type: one_hot_encoder
      columns: ["category"]
      parameters:
        drop: "first"
  
  train_split: 0.7
  validation_split: 0.15
  test_split: 0.15
  stratify: true
  random_state: 42

model:
  type: sklearn
  parameters:
    algorithm: RandomForestClassifier
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
  hyperparameter_tuning:
    method: optuna
    n_trials: 50
    timeout: 3600
    parameters:
      n_estimators: [50, 100, 200]
      max_depth: [5, 10, 15, 20]
    cv_folds: 5
    scoring: "f1_weighted"

evaluation:
  metrics: ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
  cross_validation: true
  cv_folds: 5
  generate_plots: true
  plot_types: ["confusion_matrix", "roc_curve", "feature_importance"]

drift_detection:
  enabled: true
  baseline_data: "data/baseline.csv"
  methods: ["evidently"]
  thresholds:
    data_drift: 0.1
    prediction_drift: 0.05
  alert_channels:
    - "email:alerts@company.com"
    - "slack:#ml-alerts"

few_shot:
  enabled: false
  prompt_template: "templates/classification.txt"
  max_examples: 5
  similarity_threshold: 0.8

logging:
  level: "INFO"
  file_path: "logs/pipeline.log"
  max_file_size: "100MB"
  backup_count: 5
```

### Data Sources

| Type | Description | Required Fields | Optional Fields |
|------|-------------|----------------|-----------------|
| `csv` | CSV files | `path` | `schema_path` |
| `json` | JSON files | `path` | `schema_path` |
| `parquet` | Parquet files | `path` | `schema_path` |
| `sql` | SQL databases | `connection_string` | `table_name`, `query` |

### Model Types

| Type | Framework | Use Cases | Key Parameters |
|------|-----------|-----------|----------------|
| `sklearn` | Scikit-learn | Classification, Regression | `algorithm`, `n_estimators`, `max_depth` |
| `xgboost` | XGBoost | Gradient Boosting | `objective`, `learning_rate`, `n_estimators` |
| `pytorch` | PyTorch | Deep Learning | `model_class`, `epochs`, `learning_rate` |
| `huggingface` | Transformers | NLP, Few-shot | `model_name`, `task`, `num_labels` |

### Preprocessing Steps

| Type | Description | Parameters |
|------|-------------|------------|
| `standard_scaler` | Standardize features | `with_mean`, `with_std` |
| `robust_scaler` | Robust scaling | `quantile_range` |
| `min_max_scaler` | Min-max scaling | `feature_range` |
| `one_hot_encoder` | One-hot encoding | `drop`, `sparse` |
| `label_encoder` | Label encoding | None |
| `missing_value_imputer` | Handle missing values | `strategy`, `fill_value` |

## 🖥️ CLI Reference

### Core Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `train` | Train ML model | `--config`, `--experiment-id`, `--interactive`, `--dry-run` |
| `inference` | Run inference | `--model-path`, `--input-data`, `--batch-size` |
| `evaluate` | Evaluate model | `--test-data`, `--metrics`, `--output-path` |
| `monitor` | Monitor drift | `--current-data`, `--baseline`, `--output-path` |
| `init` | Create config template | `--output`, `--use-case`, `--format` |
| `validate` | Validate config | `--config` |
| `analyze` | Analyze config | `--check-data`, `--estimate-time`, `--suggest-improvements` |

### Management Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `experiments` | List/manage experiments | `--limit`, `--sort-by`, `--status`, `--format` |
| `progress` | Monitor progress | `--experiment-id`, `--follow`, `--refresh-interval` |
| `status` | System status | `--detailed` |
| `help-guide` | Detailed help | `--topic` |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLPIPELINE_CONFIG_PATH` | Default config path | None |
| `MLPIPELINE_LOG_LEVEL` | Logging level | INFO |
| `MLPIPELINE_ARTIFACTS_PATH` | Artifacts directory | ./artifacts |
| `MLFLOW_TRACKING_URI` | MLflow server | file:./mlruns |

### Example Workflows

```bash
# Development workflow
mlpipeline init --output dev_config.yaml --use-case classification
mlpipeline validate --config dev_config.yaml
mlpipeline analyze --config dev_config.yaml --check-data --suggest-improvements
mlpipeline train --config dev_config.yaml --dry-run
mlpipeline train --config dev_config.yaml
mlpipeline evaluate --config dev_config.yaml --model-path artifacts/model.pkl --test-data data/test.csv

# Production monitoring
mlpipeline monitor --config prod_config.yaml \
  --current-data /prod/data/daily_$(date +%Y%m%d).csv \
  --output-path /prod/reports/drift_$(date +%Y%m%d).json

# Experiment management
mlpipeline experiments --limit 20 --sort-by accuracy
mlpipeline progress --experiment-id exp_$(date +%Y%m%d) --follow
```

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Development environment with Jupyter
make up-dev
# Access Jupyter at http://localhost:8888 (token: mlpipeline-dev-token)

# Production environment
make up

# GPU-enabled environment
make up-gpu

# Full monitoring stack
make up-monitor

# Distributed computing
make up-distributed
```

### Available Docker Services

| Service | Description | Ports | Profile |
|---------|-------------|-------|---------|
| `mlpipeline` | Main application (CPU) | - | default |
| `mlpipeline-gpu` | GPU-enabled app | - | gpu |
| `mlpipeline-dev` | Development + Jupyter | 8888, 8080 | development |
| `postgres` | PostgreSQL database | 5432 | default |
| `mlflow` | MLflow tracking server | 5000 | default |
| `redis` | Caching and queues | 6379 | default |
| `prometheus` | Metrics collection | 9090 | monitoring |
| `grafana` | Visualization | 3000 | monitoring |
| `dask-scheduler` | Dask coordinator | 8786, 8787 | distributed |
| `ray-head` | Ray cluster head | 8265, 10001 | distributed |

### Docker Build Targets

```bash
# Production (CPU-optimized)
docker build --target production -t mlpipeline:prod .

# Production with GPU support
docker build --target production-gpu -t mlpipeline:prod-gpu .

# Development with tools
docker build --target development -t mlpipeline:dev .
```

### Docker Environment Variables

```bash
# Core configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
POSTGRES_HOST=postgres
POSTGRES_DB=mlpipeline
POSTGRES_USER=mlpipeline
POSTGRES_PASSWORD=mlpipeline123

# GPU configuration
CUDA_VISIBLE_DEVICES=0

# Development
DEBUG=true
LOG_LEVEL=DEBUG
JUPYTER_TOKEN=mlpipeline-dev-token
```

### Production Deployment

```bash
# Build and deploy production stack
make build
make up

# Scale distributed workers
make scale-dask-workers  # Scale to 4 Dask workers
make scale-ray-workers   # Scale to 4 Ray workers

# Monitor services
make logs
docker-compose ps
```

## 🔧 Development Setup

### Local Development

```bash
# Clone and setup
git clone https://github.com/your-org/mlpipeline.git
cd mlpipeline
python -m venv venv
source venv/bin/activate
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=mlpipeline --cov-report=html

# Code quality checks
black mlpipeline tests
flake8 mlpipeline tests
mypy mlpipeline
```

### Docker Development

```bash
# Start development environment
make up-dev

# Access development container
make shell

# Run tests in container
make test

# Test Docker setup
make test-docker
```

### Project Structure

```
mlpipeline/
├── mlpipeline/           # Main package
│   ├── cli.py           # CLI interface
│   ├── config/          # Configuration management
│   ├── core/            # Core orchestration
│   ├── data/            # Data processing
│   ├── models/          # Model training/inference
│   ├── monitoring/      # Drift detection
│   ├── few_shot/        # Few-shot learning
│   └── distributed/     # Distributed computing
├── docker/              # Docker configurations
├── examples/            # Example configurations
├── docs/                # Documentation
├── tests/               # Test suite
└── requirements*.txt    # Dependencies
```

## 📊 Monitoring & Observability

### MLflow Integration

```yaml
pipeline:
  mlflow_tracking_uri: "http://localhost:5000"
  artifact_location: "s3://ml-artifacts"
```

- **Experiment Tracking**: Automatic parameter, metric, and artifact logging
- **Model Registry**: Version control for trained models with stage management
- **Artifact Storage**: Centralized storage with S3/MinIO support
- **Comparison Tools**: Compare experiments and model performance

### Prometheus Metrics

- Custom application metrics for pipeline performance
- System resource monitoring (CPU, memory, GPU)
- Model inference latency and throughput
- Data drift detection alerts

### Grafana Dashboards

- Pre-configured dashboards for ML pipeline monitoring
- Real-time experiment tracking visualization
- Resource utilization and performance metrics
- Alert management and notification history

## 🔄 Distributed Computing

### Dask Integration

```bash
# Start Dask cluster
make up-distributed

# Access Dask dashboard
open http://localhost:8787
```

**Features:**
- Distributed data processing with automatic chunking
- Parallel hyperparameter optimization
- Scalable feature engineering
- Memory-efficient large dataset handling

### Ray Integration

```bash
# Access Ray dashboard
open http://localhost:8265
```

**Features:**
- Distributed model training and inference
- Advanced hyperparameter tuning with Ray Tune
- Distributed few-shot learning workflows
- GPU-accelerated distributed computing

### Configuration

```yaml
# Enable distributed processing
distributed:
  backend: "dask"  # or "ray"
  scheduler_address: "dask-scheduler:8786"
  n_workers: 4
  memory_per_worker: "2GB"
```

## 📚 Examples

### Minimal Example

```yaml
# examples/configs/minimal_example.yaml
pipeline:
  name: "minimal_example"

data:
  sources:
    - type: "csv"
      path: "data/sample_data.csv"
  train_split: 0.8
  test_split: 0.2

model:
  type: "sklearn"
  parameters:
    algorithm: "RandomForestClassifier"

evaluation:
  metrics: ["accuracy", "f1_score"]
```

### Advanced Classification

See `examples/configs/classification_basic.yaml` for:
- Comprehensive preprocessing pipeline
- Hyperparameter optimization with Optuna
- Cross-validation and evaluation metrics
- Model comparison and selection

### NLP Sentiment Analysis

See `examples/configs/nlp_sentiment_analysis.yaml` for:
- Hugging Face transformer integration
- Text preprocessing and tokenization
- Few-shot learning capabilities
- Attention visualization

### Time Series Forecasting

See `examples/configs/time_series_forecasting.yaml` for:
- Time series specific preprocessing
- Seasonal and trend decomposition
- Forecast horizon configuration
- Confidence interval prediction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Run quality checks (`black`, `flake8`, `mypy`, `pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Use type hints throughout the codebase
- Ensure Docker compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🗺️ Roadmap

- [ ] **Kubernetes Deployment**: Helm charts and operator for production deployment
- [ ] **Advanced AutoML**: Automated model selection and architecture search
- [ ] **Real-time Streaming**: Kafka integration for real-time data processing
- [ ] **Enhanced Few-shot Learning**: Retrieval augmentation and prompt optimization
- [ ] **Cloud Integration**: Native AWS, GCP, and Azure support
- [ ] **Model Interpretability**: SHAP, LIME, and custom explainability tools
- [ ] **Edge Deployment**: ONNX and TensorRT model optimization
- [ ] **Advanced Monitoring**: Custom drift detection algorithms and AutoML monitoring

## 📞 Support & Community

- 📖 **Documentation**: [Complete Documentation](https://mlpipeline.readthedocs.io)
- 🐛 **Issues**: [Bug Reports & Feature Requests](https://github.com/your-org/mlpipeline/issues)
- 💬 **Discussions**: [Community Forum](https://github.com/your-org/mlpipeline/discussions)
- 📧 **Email**: [support@mlpipeline.dev](mailto:support@mlpipeline.dev)
- 💼 **Enterprise**: [enterprise@mlpipeline.dev](mailto:enterprise@mlpipeline.dev)

### Getting Help

1. **Check Documentation**: Start with the [getting started guide](docs/getting_started.md)
2. **Validate Configuration**: Use `mlpipeline validate --config your_config.yaml`
3. **Analyze Setup**: Run `mlpipeline analyze --config your_config.yaml --check-data`
4. **Check System Status**: Use `mlpipeline status --detailed`
5. **Enable Verbose Logging**: Add `--verbose` to any command for detailed output
6. **Search Issues**: Check existing [GitHub issues](https://github.com/your-org/mlpipeline/issues)
7. **Ask Community**: Post in [discussions](https://github.com/your-org/mlpipeline/discussions)

### Quick Links

- [Configuration Reference](docs/configuration_reference.md)
- [CLI Reference](docs/cli_reference.md)
- [MLflow Integration](docs/mlflow_integration.md)
- [Docker Setup Guide](docker/README.md)
- [Example Configurations](examples/configs/)

---

**Built with ❤️ by the ML Pipeline Team** | **Star ⭐ this repo if it helps you!**