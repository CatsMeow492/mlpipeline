"""Simple tests for model training framework."""

import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

from mlpipeline.models.training import ModelConfig, TrainingMetrics, SklearnAdapter, ModelTrainer
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
from mlpipeline.core.errors import ModelError, ConfigurationError


class TestModelConfig:
    """Test model configuration dataclass."""
    
    def test_basic_config(self):
        """Test basic model configuration."""
        config = ModelConfig(
            model_type="logistic_regression",
            framework="sklearn",
            parameters={"C": 1.0},
            task_type="classification"
        )
        
        assert config.model_type == "logistic_regression"
        assert config.framework == "sklearn"
        assert config.parameters["C"] == 1.0
        assert config.task_type == "classification"
        assert config.random_state is None


class TestTrainingMetrics:
    """Test training metrics dataclass."""
    
    def test_basic_metrics(self):
        """Test basic training metrics."""
        metrics = TrainingMetrics(
            train_metrics={"accuracy": 0.85},
            val_metrics={"accuracy": 0.80},
            training_time=120.5
        )
        
        assert metrics.train_metrics["accuracy"] == 0.85
        assert metrics.val_metrics["accuracy"] == 0.80
        assert metrics.training_time == 120.5
        assert metrics.test_metrics is None


class TestSklearnAdapter:
    """Test scikit-learn model adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            model_type="logistic_regression",
            framework="sklearn",
            parameters={"C": 1.0},
            task_type="classification",
            random_state=42
        )
        self.adapter = SklearnAdapter(self.config)
        
        # Create sample data
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.randn(100, 4), columns=[f'feature_{i}' for i in range(4)])
        self.y_train = pd.Series(np.random.randint(0, 2, 100))
        self.X_val = pd.DataFrame(np.random.randn(20, 4), columns=[f'feature_{i}' for i in range(4)])
        self.y_val = pd.Series(np.random.randint(0, 2, 20))
    
    def test_create_model(self):
        """Test model creation."""
        model = self.adapter.create_model()
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert self.adapter.model is not None
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        self.adapter.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        
        assert self.adapter.is_fitted
        
        predictions = self.adapter.predict(self.X_val)
        assert len(predictions) == len(self.y_val)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_without_fit(self):
        """Test prediction without fitting."""
        with pytest.raises(ModelError, match="Model must be fitted"):
            self.adapter.predict(self.X_val)


class TestModelTrainer:
    """Test main model trainer component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = ModelTrainer()
        
        # Create sample training data files
        np.random.seed(42)
        train_data = pd.DataFrame({
            'feature_0': np.random.randn(100),
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        val_data = pd.DataFrame({
            'feature_0': np.random.randn(20),
            'feature_1': np.random.randn(20),
            'feature_2': np.random.randn(20),
            'target': np.random.randint(0, 2, 20)
        })
        
        # Save data files
        train_data.to_parquet(Path(self.temp_dir) / "train_preprocessed.parquet", index=False)
        val_data.to_parquet(Path(self.temp_dir) / "val_preprocessed.parquet", index=False)
        
        # Create execution context
        self.context = ExecutionContext(
            experiment_id="test_exp",
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config={
                'training': {
                    'model': {
                        'framework': 'sklearn',
                        'model_type': 'logistic_regression',
                        'task_type': 'classification',
                        'parameters': {'C': 1.0},
                        'random_state': 42
                    },
                    'target_column': 'target'
                }
            },
            artifacts_path=str(self.temp_dir),
            logger=None,
            metadata={}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.component_type == ComponentType.MODEL_TRAINING
        assert len(self.trainer.adapter_registry) == 3
        assert 'sklearn' in self.trainer.adapter_registry
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = {
            'model': {
                'framework': 'sklearn',
                'model_type': 'logistic_regression',
                'task_type': 'classification',
                'parameters': {'C': 1.0}
            }
        }
        
        assert self.trainer.validate_config(config) is True
    
    def test_validate_config_missing_framework(self):
        """Test configuration validation with missing framework."""
        config = {
            'model': {
                'model_type': 'logistic_regression',
                'task_type': 'classification'
            }
        }
        
        assert self.trainer.validate_config(config) is False
    
    def test_setup_success(self):
        """Test successful setup."""
        self.trainer.setup(self.context)
        
        assert self.trainer.model_adapter is not None
        assert isinstance(self.trainer.model_adapter, SklearnAdapter)
        assert self.trainer.model_adapter.config.framework == 'sklearn'
    
    def test_execute_success(self):
        """Test successful model training execution."""
        result = self.trainer.execute(self.context)
        
        assert result.success is True
        assert len(result.artifacts) >= 3  # model, metrics, config
        assert 'training_time_seconds' in result.metrics
        assert 'train_accuracy' in result.metrics
        assert 'val_accuracy' in result.metrics
        
        # Check that files were created
        model_file = Path(self.temp_dir) / "trained_model.joblib"
        assert model_file.exists()
    
    def test_execute_no_training_data(self):
        """Test execution with no training data."""
        # Remove training data file
        (Path(self.temp_dir) / "train_preprocessed.parquet").unlink()
        
        result = self.trainer.execute(self.context)
        
        assert result.success is False
        assert "Training data not found" in result.error_message