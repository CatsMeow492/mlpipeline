"""Tests for hyperparameter optimization functionality."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mlpipeline.models.hyperparameter_optimization import (
    HyperparameterOptimizer,
    HyperparameterOptimizedTrainer,
    HyperparameterConfig,
    OptimizationResult
)
from mlpipeline.models.training import ModelTrainer, ModelConfig
from mlpipeline.core.interfaces import ExecutionContext
from mlpipeline.core.errors import ModelError, ConfigurationError


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple binary classification
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    return df


@pytest.fixture
def temp_artifacts_dir():
    """Create temporary artifacts directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_path = Path(temp_dir)
        yield artifacts_path


@pytest.fixture
def mock_execution_context(temp_artifacts_dir, sample_data):
    """Create mock execution context."""
    # Save sample data to artifacts directory
    train_file = temp_artifacts_dir / "train_preprocessed.parquet"
    val_file = temp_artifacts_dir / "val_preprocessed.parquet"
    
    # Split data
    train_data = sample_data.iloc[:80]
    val_data = sample_data.iloc[80:]
    
    train_data.to_parquet(train_file)
    val_data.to_parquet(val_file)
    
    context = Mock(spec=ExecutionContext)
    context.artifacts_path = str(temp_artifacts_dir)
    context.config = {
        'training': {
            'target_column': 'target',
            'model': {
                'framework': 'sklearn',
                'model_type': 'random_forest_classifier',
                'task_type': 'classification',
                'parameters': {'n_estimators': 10, 'random_state': 42}
            }
        }
    }
    
    return context


@pytest.fixture
def model_trainer():
    """Create model trainer instance."""
    trainer = ModelTrainer()
    return trainer


class TestHyperparameterConfig:
    """Test HyperparameterConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HyperparameterConfig()
        
        assert config.method == "optuna"
        assert config.n_trials == 50
        assert config.timeout is None
        assert config.sampler == "tpe"
        assert config.pruner == "median"
        assert config.direction == "maximize"
        assert config.metric == "accuracy"
        assert config.cv_folds == 5
        assert config.parameter_space == {}
    
    def test_custom_config(self):
        """Test custom configuration values."""
        param_space = {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10}
        }
        
        config = HyperparameterConfig(
            method="grid_search",
            n_trials=20,
            timeout=3600,
            sampler="random",
            pruner="successive_halving",
            direction="minimize",
            metric="mse",
            cv_folds=3,
            parameter_space=param_space
        )
        
        assert config.method == "grid_search"
        assert config.n_trials == 20
        assert config.timeout == 3600
        assert config.sampler == "random"
        assert config.pruner == "successive_halving"
        assert config.direction == "minimize"
        assert config.metric == "mse"
        assert config.cv_folds == 3
        assert config.parameter_space == param_space


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_optimization_result_creation(self):
        """Test creation of optimization result."""
        result = OptimizationResult(
            best_params={'n_estimators': 100, 'max_depth': 5},
            best_value=0.95,
            best_trial_number=15,
            n_trials=50,
            optimization_time=120.5,
            study_name="test_study",
            all_trials=[],
            pruned_trials=5,
            failed_trials=2
        )
        
        assert result.best_params == {'n_estimators': 100, 'max_depth': 5}
        assert result.best_value == 0.95
        assert result.best_trial_number == 15
        assert result.n_trials == 50
        assert result.optimization_time == 120.5
        assert result.study_name == "test_study"
        assert result.all_trials == []
        assert result.pruned_trials == 5
        assert result.failed_trials == 2


@pytest.mark.skipif(
    not pytest.importorskip("optuna", reason="Optuna not available"),
    reason="Optuna not available"
)
class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer class."""
    
    def test_optimizer_initialization(self, model_trainer):
        """Test optimizer initialization."""
        optimizer = HyperparameterOptimizer(model_trainer)
        
        assert optimizer.model_trainer == model_trainer
        assert optimizer.study is None
        assert optimizer.optimization_config is None
        assert optimizer.training_data is None
        assert optimizer.validation_data is None
    
    def test_optimizer_initialization_without_optuna(self, model_trainer):
        """Test optimizer initialization fails without Optuna."""
        with patch('mlpipeline.models.hyperparameter_optimization.OPTUNA_AVAILABLE', False):
            with pytest.raises(ModelError, match="Optuna is not installed"):
                HyperparameterOptimizer(model_trainer)
    
    def test_setup_optimization(self, model_trainer, mock_execution_context):
        """Test optimization setup."""
        # Setup model trainer first
        model_trainer.setup(mock_execution_context)
        
        optimizer = HyperparameterOptimizer(model_trainer)
        config = HyperparameterConfig(n_trials=10, cv_folds=3)
        
        optimizer.setup_optimization(config, mock_execution_context)
        
        assert optimizer.optimization_config == config
        assert optimizer.training_data is not None
        assert optimizer.validation_data is not None
        assert optimizer.study is not None
        assert optimizer.study.study_name.startswith("hyperopt_")
    
    def test_create_tpe_sampler(self, model_trainer):
        """Test TPE sampler creation."""
        optimizer = HyperparameterOptimizer(model_trainer)
        config = HyperparameterConfig(sampler="tpe", random_state=42)
        
        sampler = optimizer._create_sampler(config)
        
        # Check that it's a TPE sampler (exact type checking depends on Optuna version)
        assert hasattr(sampler, 'sample_relative')
    
    def test_create_random_sampler(self, model_trainer):
        """Test random sampler creation."""
        optimizer = HyperparameterOptimizer(model_trainer)
        config = HyperparameterConfig(sampler="random", random_state=42)
        
        sampler = optimizer._create_sampler(config)
        
        # Check that it's a random sampler
        assert hasattr(sampler, 'sample_relative')
    
    def test_create_unsupported_sampler(self, model_trainer):
        """Test unsupported sampler raises error."""
        optimizer = HyperparameterOptimizer(model_trainer)
        config = HyperparameterConfig(sampler="unsupported")
        
        with pytest.raises(ConfigurationError, match="Unsupported sampler"):
            optimizer._create_sampler(config)
    
    def test_create_median_pruner(self, model_trainer):
        """Test median pruner creation."""
        optimizer = HyperparameterOptimizer(model_trainer)
        config = HyperparameterConfig(pruner="median")
        
        pruner = optimizer._create_pruner(config)
        
        # Check that it's a median pruner
        assert hasattr(pruner, 'prune')
    
    def test_create_nop_pruner(self, model_trainer):
        """Test no-op pruner creation."""
        optimizer = HyperparameterOptimizer(model_trainer)
        config = HyperparameterConfig(pruner="none")
        
        pruner = optimizer._create_pruner(config)
        
        # Check that it's a no-op pruner
        assert hasattr(pruner, 'prune')
    
    def test_create_unsupported_pruner(self, model_trainer):
        """Test unsupported pruner raises error."""
        optimizer = HyperparameterOptimizer(model_trainer)
        config = HyperparameterConfig(pruner="unsupported")
        
        with pytest.raises(ConfigurationError, match="Unsupported pruner"):
            optimizer._create_pruner(config)
    
    def test_get_default_parameter_space_random_forest(self, model_trainer, mock_execution_context):
        """Test default parameter space for random forest."""
        model_trainer.setup(mock_execution_context)
        optimizer = HyperparameterOptimizer(model_trainer)
        
        # Mock trial object
        trial = Mock()
        trial.suggest_int.side_effect = lambda name, low, high: 50 if 'n_estimators' in name else 5
        trial.suggest_categorical.return_value = 'sqrt'
        
        params = optimizer._get_default_parameter_space(trial)
        
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'min_samples_split' in params
        assert 'min_samples_leaf' in params
        assert 'max_features' in params
    
    def test_suggest_parameters_with_custom_space(self, model_trainer):
        """Test parameter suggestion with custom parameter space."""
        optimizer = HyperparameterOptimizer(model_trainer)
        optimizer.optimization_config = HyperparameterConfig(
            parameter_space={
                'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd']}
            }
        )
        
        # Mock trial object
        trial = Mock()
        trial.suggest_int.return_value = 50
        trial.suggest_float.return_value = 0.1
        trial.suggest_categorical.return_value = 'adam'
        
        params = optimizer._suggest_parameters(trial)
        
        assert params['n_estimators'] == 50
        assert params['learning_rate'] == 0.1
        assert params['optimizer'] == 'adam'
        
        # Verify correct calls were made
        trial.suggest_int.assert_called_with('n_estimators', 10, 100, log=False)
        trial.suggest_float.assert_called_with('learning_rate', 0.01, 0.3, log=True)
        trial.suggest_categorical.assert_called_with('optimizer', ['adam', 'sgd'])
    
    def test_evaluate_fold_classification(self, model_trainer, mock_execution_context):
        """Test fold evaluation for classification."""
        model_trainer.setup(mock_execution_context)
        optimizer = HyperparameterOptimizer(model_trainer)
        optimizer.optimization_config = HyperparameterConfig(metric="accuracy")
        
        # Create mock model adapter
        mock_adapter = Mock()
        mock_adapter.config.task_type = "classification"
        mock_adapter.predict.return_value = np.array([0, 1, 0, 1])
        
        # Create test data
        X_val = pd.DataFrame({'feature_0': [1, 2, 3, 4], 'feature_1': [5, 6, 7, 8]})
        y_val = pd.Series([0, 1, 0, 1])
        
        score = optimizer._evaluate_fold(mock_adapter, X_val, y_val)
        
        assert score == 1.0  # Perfect accuracy
        mock_adapter.predict.assert_called_once_with(X_val)
    
    def test_evaluate_fold_regression(self, model_trainer):
        """Test fold evaluation for regression."""
        optimizer = HyperparameterOptimizer(model_trainer)
        optimizer.optimization_config = HyperparameterConfig(metric="mse")
        
        # Create mock model adapter
        mock_adapter = Mock()
        mock_adapter.config.task_type = "regression"
        mock_adapter.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Create test data
        X_val = pd.DataFrame({'feature_0': [1, 2, 3, 4], 'feature_1': [5, 6, 7, 8]})
        y_val = pd.Series([1.0, 2.0, 3.0, 4.0])
        
        score = optimizer._evaluate_fold(mock_adapter, X_val, y_val)
        
        assert score == 0.0  # Perfect MSE (negated, so 0)
        mock_adapter.predict.assert_called_once_with(X_val)
    
    def test_unsupported_metric_raises_error(self, model_trainer):
        """Test unsupported metric raises ValueError."""
        optimizer = HyperparameterOptimizer(model_trainer)
        optimizer.optimization_config = HyperparameterConfig(metric="unsupported_metric")
        
        mock_adapter = Mock()
        mock_adapter.config.task_type = "classification"
        
        X_val = pd.DataFrame({'feature_0': [1, 2]})
        y_val = pd.Series([0, 1])
        
        with pytest.raises(ValueError, match="Unsupported metric"):
            optimizer._evaluate_fold(mock_adapter, X_val, y_val)


@pytest.mark.skipif(
    not pytest.importorskip("optuna", reason="Optuna not available"),
    reason="Optuna not available"
)
class TestHyperparameterOptimizedTrainer:
    """Test HyperparameterOptimizedTrainer class."""
    
    def test_trainer_initialization(self):
        """Test optimized trainer initialization."""
        trainer = HyperparameterOptimizedTrainer()
        
        assert trainer.hyperparameter_optimizer is None
        assert trainer.optimization_result is None
        assert hasattr(trainer, 'adapter_registry')
    
    def test_validate_config_with_optimization_disabled(self):
        """Test config validation with optimization disabled."""
        trainer = HyperparameterOptimizedTrainer()
        
        config = {
            'model': {
                'framework': 'sklearn',
                'model_type': 'random_forest_classifier',
                'task_type': 'classification',
                'parameters': {}
            }
        }
        
        assert trainer.validate_config(config) is True
    
    def test_validate_config_with_optimization_enabled(self):
        """Test config validation with optimization enabled."""
        trainer = HyperparameterOptimizedTrainer()
        
        config = {
            'model': {
                'framework': 'sklearn',
                'model_type': 'random_forest_classifier',
                'task_type': 'classification',
                'parameters': {}
            },
            'hyperparameter_optimization': {
                'enabled': True,
                'method': 'optuna',
                'n_trials': 10
            }
        }
        
        assert trainer.validate_config(config) is True
    
    def test_validate_config_with_unsupported_method(self):
        """Test config validation with unsupported optimization method."""
        trainer = HyperparameterOptimizedTrainer()
        
        config = {
            'model': {
                'framework': 'sklearn',
                'model_type': 'random_forest_classifier',
                'task_type': 'classification',
                'parameters': {}
            },
            'hyperparameter_optimization': {
                'enabled': True,
                'method': 'unsupported_method'
            }
        }
        
        assert trainer.validate_config(config) is False
    
    def test_validate_config_without_optuna(self):
        """Test config validation fails when Optuna is not available."""
        with patch('mlpipeline.models.hyperparameter_optimization.OPTUNA_AVAILABLE', False):
            trainer = HyperparameterOptimizedTrainer()
            
            config = {
                'model': {
                    'framework': 'sklearn',
                    'model_type': 'random_forest_classifier',
                    'task_type': 'classification',
                    'parameters': {}
                },
                'hyperparameter_optimization': {
                    'enabled': True
                }
            }
            
            assert trainer.validate_config(config) is False
    
    def test_setup_without_optimization(self, mock_execution_context):
        """Test setup without hyperparameter optimization."""
        trainer = HyperparameterOptimizedTrainer()
        
        trainer.setup(mock_execution_context)
        
        assert trainer.model_adapter is not None
        assert trainer.hyperparameter_optimizer is None
    
    def test_setup_with_optimization(self, mock_execution_context):
        """Test setup with hyperparameter optimization."""
        # Add hyperparameter optimization config
        mock_execution_context.config['training']['hyperparameter_optimization'] = {
            'enabled': True,
            'method': 'optuna',
            'n_trials': 5,
            'metric': 'accuracy'
        }
        
        trainer = HyperparameterOptimizedTrainer()
        trainer.setup(mock_execution_context)
        
        assert trainer.model_adapter is not None
        assert trainer.hyperparameter_optimizer is not None
    
    def test_execute_without_optimization(self, mock_execution_context):
        """Test execution without hyperparameter optimization."""
        trainer = HyperparameterOptimizedTrainer()
        
        # Mock the parent execute method
        with patch.object(ModelTrainer, 'execute') as mock_execute:
            mock_execute.return_value = Mock(success=True)
            
            result = trainer.execute(mock_execution_context)
            
            mock_execute.assert_called_once_with(mock_execution_context)
            assert result.success is True
    
    @patch('mlpipeline.models.hyperparameter_optimization.HyperparameterOptimizer')
    def test_execute_with_optimization_success(self, mock_optimizer_class, mock_execution_context):
        """Test successful execution with hyperparameter optimization."""
        # Setup optimization config
        mock_execution_context.config['training']['hyperparameter_optimization'] = {
            'enabled': True,
            'method': 'optuna',
            'n_trials': 5
        }
        
        # Mock optimization result
        mock_optimization_result = Mock()
        mock_optimization_result.best_params = {'n_estimators': 100, 'max_depth': 5}
        mock_optimization_result.best_value = 0.95
        mock_optimization_result.n_trials = 5
        mock_optimization_result.pruned_trials = 1
        mock_optimization_result.failed_trials = 0
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = mock_optimization_result
        mock_optimizer.get_optimization_history.return_value = pd.DataFrame()
        mock_optimizer.save_study = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        trainer = HyperparameterOptimizedTrainer()
        
        # Mock parent execute method
        with patch.object(ModelTrainer, 'execute') as mock_execute:
            mock_result = Mock()
            mock_result.success = True
            mock_result.metadata = {}
            mock_result.artifacts = []
            mock_execute.return_value = mock_result
            
            result = trainer.execute(mock_execution_context)
            
            assert result.success is True
            assert 'hyperparameter_optimization' in result.metadata
            assert result.metadata['hyperparameter_optimization'] is True
            assert 'best_hyperparameters' in result.metadata
            assert result.metadata['best_hyperparameters'] == {'n_estimators': 100, 'max_depth': 5}
    
    def test_execute_with_optimization_failure(self, mock_execution_context):
        """Test execution failure with hyperparameter optimization."""
        # Setup optimization config
        mock_execution_context.config['training']['hyperparameter_optimization'] = {
            'enabled': True,
            'method': 'optuna',
            'n_trials': 5
        }
        
        trainer = HyperparameterOptimizedTrainer()
        
        # Mock optimizer to raise exception
        with patch('mlpipeline.models.hyperparameter_optimization.HyperparameterOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer.optimize.side_effect = Exception("Optimization failed")
            mock_optimizer_class.return_value = mock_optimizer
            
            result = trainer.execute(mock_execution_context)
            
            assert result.success is False
            assert "Optimization failed" in result.error_message


class TestIntegration:
    """Integration tests for hyperparameter optimization."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not available"),
        reason="Optuna not available"
    )
    def test_end_to_end_optimization(self, mock_execution_context, temp_artifacts_dir):
        """Test end-to-end hyperparameter optimization."""
        # Setup optimization config
        mock_execution_context.config['training']['hyperparameter_optimization'] = {
            'enabled': True,
            'method': 'optuna',
            'n_trials': 3,  # Small number for testing
            'metric': 'accuracy',
            'cv_folds': 2,  # Small number for testing
            'parameter_space': {
                'n_estimators': {'type': 'int', 'low': 5, 'high': 15},
                'max_depth': {'type': 'int', 'low': 2, 'high': 5}
            }
        }
        
        trainer = HyperparameterOptimizedTrainer()
        
        # This should run without errors
        result = trainer.execute(mock_execution_context)
        
        # Check that optimization completed
        assert result.success is True
        assert 'hyperparameter_optimization' in result.metadata
        assert result.metadata['hyperparameter_optimization'] is True
        assert 'best_hyperparameters' in result.metadata
        assert 'optimization_trials' in result.metadata
        
        # Check that optimization artifacts were created
        optimization_files = [
            'hyperparameter_optimization_results.json',
            'optimization_history.csv',
            'optuna_study.json'
        ]
        
        for filename in optimization_files:
            filepath = temp_artifacts_dir / filename
            assert filepath.exists(), f"Expected optimization artifact {filename} not found"
    
    def test_optimization_with_invalid_config(self, mock_execution_context):
        """Test optimization with invalid configuration."""
        # Setup invalid optimization config
        mock_execution_context.config['training']['hyperparameter_optimization'] = {
            'enabled': True,
            'method': 'invalid_method'
        }
        
        trainer = HyperparameterOptimizedTrainer()
        
        # This should fail validation
        assert trainer.validate_config(mock_execution_context.config['training']) is False


if __name__ == "__main__":
    pytest.main([__file__])