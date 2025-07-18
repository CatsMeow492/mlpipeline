# 🎯 ML Pipeline Framework Demos

This directory contains comprehensive demonstrations of the ML Pipeline framework capabilities using real-world datasets. Each demo showcases different aspects of the framework and provides practical examples of machine learning workflows.

## 📁 Directory Structure

```
demos/
├── README.md                 # This file
├── data/                     # Real datasets downloaded via wget
│   ├── telco_customer_churn.csv      # Telecom customer churn data
│   ├── wine_quality_red.csv          # Wine quality ratings
│   ├── adult_income.csv              # Census income data
│   └── titanic.csv                   # Titanic passenger data
├── configs/                  # Demo configurations
│   ├── telco_churn_demo.yaml         # Customer churn classification
│   ├── wine_quality_demo.yaml        # Wine quality regression
│   ├── titanic_survival_demo.yaml    # Survival prediction
│   └── adult_income_demo.yaml        # Income prediction
├── schemas/                  # Data schemas (optional)
└── results/                  # Demo outputs and logs
```

## 🚀 Available Demos

### 1. 🛒 **Telco Customer Churn Classification**
**File:** `configs/telco_churn_demo.yaml`  
**Dataset:** Telecom customer data (7,043 customers)

**What it demonstrates:**
- ✅ Binary classification with imbalanced classes
- ✅ Feature engineering with mixed data types
- ✅ Handling categorical variables with one-hot encoding
- ✅ Class balancing techniques
- ✅ Hyperparameter optimization with Optuna
- ✅ Comprehensive evaluation metrics
- ✅ Drift detection and monitoring

**Business Context:** Predict which customers are likely to churn so marketing can proactively retain them.

**Key Features:**
- 20+ features including tenure, charges, services
- Binary target: Churn (Yes/No)
- Class imbalance handling with `class_weight="balanced"`
- F1-score optimization for imbalanced data

### 2. 🍷 **Wine Quality Regression**
**File:** `configs/wine_quality_demo.yaml`  
**Dataset:** Portuguese red wine quality (1,599 wines)

**What it demonstrates:**
- ✅ Regression with continuous target variable
- ✅ Feature scaling for numerical data
- ✅ Regression metrics (MSE, MAE, R², MAPE)
- ✅ Residual analysis and prediction plots
- ✅ Feature importance analysis

**Business Context:** Predict wine quality scores based on physicochemical properties to optimize wine production.

**Key Features:**
- 11 numerical features (acidity, sugar, alcohol, etc.)
- Target: Quality score (0-10)
- All numerical features requiring standardization
- Perfect for demonstrating regression capabilities

### 3. ⚓ **Titanic Survival Prediction**
**File:** `configs/titanic_survival_demo.yaml`  
**Dataset:** Titanic passenger data (891 passengers)

**What it demonstrates:**
- ✅ Feature engineering with missing data
- ✅ Handling mixed data types (numerical + categorical)
- ✅ Missing value imputation strategies
- ✅ Feature selection and engineering
- ✅ Classic binary classification

**Business Context:** Historical analysis of survival factors to understand passenger demographics and safety factors.

**Key Features:**
- Mixed data types: Age, Fare (numerical), Sex, Class (categorical)
- Significant missing data requiring imputation
- Classic ML problem for learning and comparison
- Well-balanced target variable

### 4. 💰 **Adult Income Prediction**
**File:** `configs/adult_income_demo.yaml`  
**Dataset:** US Census data (32,561 individuals)

**What it demonstrates:**
- ✅ Large dataset handling
- ✅ Complex categorical encoding
- ✅ XGBoost for structured data
- ✅ High-cardinality categorical features
- ✅ Early stopping and advanced optimization

**Business Context:** Predict income levels for socioeconomic analysis and policy planning.

**Key Features:**
- Large dataset (32K+ samples)
- Many categorical features with high cardinality
- Income prediction (>50K vs ≤50K)
- Demonstrates scalability and performance

## 🏃‍♂️ Quick Start

### Prerequisites

1. **Ensure ML Pipeline is installed:**
```bash
cd /path/to/mlpipeline
pip install -e .
```

2. **Start MLflow tracking server:**
```bash
# Option 1: Using Docker (recommended)
make up-dev
# MLflow UI available at http://localhost:5000

# Option 2: Local MLflow
mlflow server --host 0.0.0.0 --port 5000
```

### Running a Demo

#### 1. **Validate Configuration**
```bash
# From project root directory
mlpipeline validate --config demos/configs/telco_churn_demo.yaml
```

#### 2. **Analyze Setup**
```bash
mlpipeline analyze --config demos/configs/telco_churn_demo.yaml --check-data --suggest-improvements
```

#### 3. **Run Training**
```bash
# Quick demo run (reduced trials for speed)
mlpipeline train --config demos/configs/telco_churn_demo.yaml --experiment-id "demo_$(date +%Y%m%d)"

# Production-quality run (full hyperparameter optimization)
mlpipeline train --config demos/configs/telco_churn_demo.yaml --experiment-id "full_demo_$(date +%Y%m%d)"
```

#### 4. **Monitor Progress**
```bash
# Real-time monitoring
mlpipeline progress --experiment-id demo_20240118 --follow

# Check experiments
mlpipeline experiments --limit 5 --sort-by accuracy
```

#### 5. **View Results**
- **MLflow UI**: http://localhost:5000
- **Logs**: `demos/results/telco_churn_demo.log`
- **Artifacts**: Saved in MLflow (models, plots, metrics)

## 📊 Demo Comparison Matrix

| Demo | Dataset Size | Problem Type | Complexity | Time to Run | Key Learning |
|------|-------------|--------------|------------|-------------|--------------|
| **Telco Churn** | 7K rows | Binary Classification | Medium | ~20 min | Imbalanced data, business impact |
| **Wine Quality** | 1.6K rows | Regression | Low | ~15 min | Regression metrics, feature scaling |
| **Titanic** | 891 rows | Binary Classification | Medium | ~10 min | Feature engineering, missing data |
| **Adult Income** | 32K rows | Binary Classification | High | ~40 min | Large data, complex categories |

## 🎓 Learning Path

### **Beginner** → Start with Wine Quality
- Simple regression problem
- All numerical features
- Clear interpretation
- Fast execution

### **Intermediate** → Try Titanic
- Feature engineering
- Missing data handling
- Mixed data types
- Classic ML problem

### **Advanced** → Run Telco Churn
- Business-relevant problem
- Class imbalance handling
- Comprehensive evaluation
- Drift detection

### **Expert** → Adult Income
- Large dataset challenges
- Complex categorical features
- Advanced optimization
- Production considerations

## 🔧 Customizing Demos

### Modify Training Parameters

```yaml
# Reduce training time for quick tests
model:
  hyperparameter_tuning:
    n_trials: 10        # Instead of 50
    timeout: 300        # 5 minutes instead of 30
```

### Enable Distributed Computing

```bash
# Start distributed stack
make up-distributed

# Update config to use distributed backend
# Add to config.yaml:
distributed:
  backend: "dask"
  scheduler_address: "localhost:8786"
```

### Test Different Models

```yaml
# Try XGBoost instead of Random Forest
model:
  type: xgboost
  parameters:
    objective: "binary:logistic"
    n_estimators: 100
    max_depth: 6
```

## 📈 Expected Results

### Telco Customer Churn
- **Accuracy**: ~80-85%
- **F1-Score**: ~60-65% (due to class imbalance)
- **ROC-AUC**: ~85-90%
- **Training Time**: 15-30 minutes

### Wine Quality Regression
- **R² Score**: ~0.35-0.45
- **RMSE**: ~0.65-0.75
- **MAE**: ~0.50-0.60
- **Training Time**: 10-20 minutes

### Titanic Survival
- **Accuracy**: ~80-85%
- **F1-Score**: ~75-80%
- **ROC-AUC**: ~85-88%
- **Training Time**: 5-15 minutes

### Adult Income Prediction
- **Accuracy**: ~85-87%
- **F1-Score**: ~70-75%
- **ROC-AUC**: ~90-92%
- **Training Time**: 30-60 minutes

## 🐛 Troubleshooting

### Common Issues

1. **"Dataset not found"**
   ```bash
   # Ensure you're running from project root
   pwd  # Should end with /mlpipeline
   ls demos/data/  # Should show CSV files
   ```

2. **"MLflow connection error"**
   ```bash
   # Start MLflow server
   make up-dev
   # Or manually: mlflow server --host 0.0.0.0 --port 5000
   ```

3. **"Configuration validation failed"**
   ```bash
   # Check configuration syntax
   mlpipeline validate --config demos/configs/your_demo.yaml
   ```

4. **"Memory/performance issues"**
   ```bash
   # Reduce hyperparameter trials
   # Modify n_trials in config from 50 to 10
   
   # Use distributed computing
   make up-distributed
   ```

### Getting Help

1. **Check logs**: `demos/results/*.log`
2. **Enable verbose mode**: Add `--verbose` to any command
3. **Validate setup**: `mlpipeline status --detailed`
4. **Check system**: `mlpipeline analyze --config your_config.yaml`

## 🎯 Next Steps

After running the demos:

1. **Explore MLflow UI** - Compare experiments and models
2. **Modify configurations** - Try different algorithms and parameters
3. **Add custom features** - Implement your own preprocessing steps
4. **Scale up** - Test with distributed computing
5. **Deploy models** - Use inference capabilities
6. **Monitor drift** - Set up real-time monitoring

## 📚 Additional Resources

- [Configuration Reference](../docs/configuration_reference.md)
- [CLI Reference](../docs/cli_reference.md)
- [Docker Setup Guide](../docker/README.md)
- [MLflow Integration](../docs/mlflow_integration.md)

---

**Happy experimenting! 🚀** These demos showcase the full power of the ML Pipeline framework. Start with any demo that matches your interest and skill level!