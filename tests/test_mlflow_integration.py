"""Tests for MLflow integration functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# Import the modules to test
from mlpipeline.models.mlflow_integration import (
    MLflowConfig, MLflowTracker, MLflowIntegratedTrainer,
    MLflowIntegratedEvaluator, MLflowIntegratedHyperparameterTrainer,
    MLflowRunInfo, MLFLOW_AVAILABLE
)
from mlpipeline.models.training import ModelConfig
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
from mlpipeline.core.errors import ModelError


class TestMLflowConfig:
    """Test MLflowConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MLflowConfig()
        
        assert config.tracking_uri is None
        assert config.experiment_name == "ml-pipeline-experiment"
        assert config.run_name is None
        assert config.log_params is True
        assert config.log_metrics is True
        assert config.log_artifacts is True
        assert config.log_model is True
        assert config.register_model is False
        assert config.tags == {}
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test-experiment",
            run_name="test-run",
            log_params=False,
            register_model=True,
            model_name="test-model",
            tags={"env": "test"}
        )
        
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "test-experiment"
        assert config.run_name == "test-run"
        assert config.log_params is False
        assert config.register_model is True
        assert config.model_name == "test-model"
        assert config.tags == {"env": "test"}


class TestMLflowRunInfo:
    """Test MLflowRunInfo dataclass."""
    
    def test_run_info_creation(self):
        """Test MLflowRunInfo creation and serialization."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        run_info = MLflowRunInfo(
            run_id="test-run-id",
            experiment_id="test-exp-id",
            run_name="test-run",
            status="FINISHED",
            start_time=start_time,
            end_time=end_time,
            artifact_uri="file:///tmp/artifacts",
            lifecycle_stage="active",
            user_id="test-user",
            tags={"env": "test"},
            params={"param1": "value1"},
            metrics={"accuracy": 0.85}
        )
        
        assert run_info.run_id == "test-run-id"
        assert run_info.experiment_id == "test-exp-id"
        assert run_info.status == "FINISHED"
        assert run_info.tags == {"env": "test"}
        assert run_info.params == {"param1": "value1"}
        assert run_info.metrics == {"accuracy": 0.85}
        
        # Test serialization
        run_dict = run_info.to_dict()
        assert run_dict["run_id"] == "test-run-id"
        assert run_dict["start_time"] == start_time.isoformat()
        assert run_dict["end_time"] == end_time.isoformat()


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestMLflowTracker:
    """Test MLflowTracker functionality."""
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.create_experiment')
    @patch('mlflow.set_experiment')
    def test_setup_mlflow(self, mock_set_exp, mock_create_exp, mock_get_exp, mock_set_uri):
        """Test MLflow setup process."""
        # Mock experiment doesn't exist
        mock_get_exp.return_value = None
        mock_create_exp.return_value = "test-exp-id"
        
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test-experiment"
        )
        
        tracker = MLflowTracker(config)
        
        # Verify setup calls
        mock_set_uri.assert_called_once_with("http://localhost:5000")
        mock_get_exp.assert_called_once_with("test-experiment")
        mock_create_exp.assert_called_once()
        mock_set_exp.assert_called_once_with("test-experiment")
        
        assert tracker.experiment_id == "test-exp-id"
    
    @patch('mlflow.start_run')
    def test_start_run(self, mock_start_run):
        """Test starting MLflow run."""
        mock_run = Mock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value = mock_run
        
        config = MLflowConfig(experiment_name="test-experiment")
        
        with patch.object(MLflowTracker, '_setup_mlflow'):
            tracker = MLflowTracker(config)
            tracker.experiment_id = "test-exp-id"
            
            run_id = tracker.start_run(run_name="test-run", tags={"env": "test"})
            
            assert run_id == "test-run-id"
            assert tracker.current_run == mock_run
            mock_start_run.assert_called_once()
    
    @patch('mlflow.end_run')
    def test_end_run(self, mock_end_run):
        """Test ending MLflow run."""
        config = MLflowConfig(experiment_name="test-experiment")
        
        with patch.object(MLflowTracker, '_setup_mlflow'):
            tracker = MLflowTracker(config)
            tracker.current_run = Mock()
            tracker.current_run.info.run_id = "test-run-id"
            
            tracker.end_run("FINISHED")
            
            mock_end_run.assert_called_once_with(status="FINISHED")
            assert tracker.current_run is None
    
    @patch('mlflow.log_params')
    def test_log_params(self, mock_log_params):
        """Test logging parameters."""
        config = MLflowConfig(experiment_name="test-experiment")
        
        with patch.object(MLflowTracker, '_setup_mlflow'):
            tracker = MLflowTracker(config)
            tracker.current_run = Mock()
            
            params = {
                "string_param": "value",
                "int_param": 42,
                "float_param": 3.14,
                "dict_param": {"nested": "value"},
                "list_param": [1, 2, 3]
            }
            
            tracker.log_params(params)
            
            # Verify parameters were converted to strings
            expected_params = {
                "string_param": "value",
                "int_param": "42",
                "float_param": "3.14",
                "dict_param": '{"nested": "value"}',
                "list_param": "[1, 2, 3]"
            }
            
            mock_log_params.assert_called_once_with(expected_params)
    
    @patch('mlflow.log_metrics')
    def test_log_metrics(self, mock_log_metrics):
        """Test logging metrics."""
        config = MLflowConfig(experiment_name="test-experiment")
        
        with patch.object(MLflowTracker, '_setup_mlflow'):
            tracker = MLflowTracker(config)
            tracker.current_run = Mock()
            
            metrics = {
                "accuracy": 0.85,
                "precision": 0.82,
                "invalid_metric": float('nan'),  # Should be filtered out
                "inf_metric": float('inf')  # Should be filtered out
            }
            
            tracker.log_metrics(metrics)
            
            # Verify only valid metrics were logged
            expected_metrics = {
                "accuracy": 0.85,
                "precision": 0.82
            }
            
            mock_log_metrics.assert_called_once_with(expected_metrics, step=None)
    
    @patch('mlflow.sklearn.log_model')
    def test_log_sklearn_model(self, mock_log_model):
        """Test logging sklearn model."""
        config = MLflowConfig(experiment_name="test-experiment")
        
        with patch.object(MLflowTracker, '_setup_mlflow'):
            tracker = MLflowTracker(config)
            tracker.current_run = Mock()
            
            mock_model = Mock()
            
            tracker.log_model(
                model=mock_model,
                model_name="test-model",
                framework="sklearn"
            )
            
            mock_log_model.assert_called_once()
            call_args = mock_log_model.call_args
            assert call_args[1]['sk_model'] == mock_model
            assert call_args[1]['artifact_path'] == "test-model"


class TestMLflowIntegratedTrainer:
    """Test MLflowIntegratedTrainer functionality."""
    
    def setup_method(self):
        """Setup test data and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save as training data
        df.to_parquet(self.temp_path / "train_preprocessed.parquet", index=False)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_validate_config_mlflow_disabled(self):
        """Test config validation when MLflow is disabled."""
        trainer = MLflowIntegratedTrainer()
        
        config = {
            'model': {
                'framework': 'sklearn',
                'model_type': 'random_forest_classifier',
                'task_type': 'classification'
            },
            'mlflow': {
                'enabled': False
            }
        }
        
        assert trainer.validate_config(config) is True
    
    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_validate_config_mlflow_enabled(self):
        """Test config validation when MLflow is enabled."""
        trainer = MLflowIntegratedTrainer()
        
        config = {
            'model': {
                'framework': 'sklearn',
                'model_type': 'random_forest_classifier',
                'task_type': 'classification'
            },
            'mlflow': {
                'enabled': True
            }
        }
        
        assert trainer.validate_config(config) is True
    
    @pytest.mark.skipif(MLFLOW_AVAILABLE, reason="Test for when MLflow is not available")
    def test_validate_config_mlflow_not_installed(self):
        """Test config validation when MLflow is not installed but enabled."""
        trainer = MLflowIntegratedTrainer()
        
        config = {
            'model': {
                'framework': 'sklearn',
                'model_type': 'random_forest_classifier',
                'task_type': 'classification'
            },
            'mlflow': {
                'enabled': True
            }
        }
        
        assert trainer.validate_config(config) is False
    
    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    @patch('mlpipeline.models.mlflow_integration.MLflowTracker')
    def test_setup_with_mlflow(self, mock_tracker_class):
        """Test setup with MLflow integration enabled."""
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        trainer = MLflowIntegratedTrainer()
        
        config = {
            'training': {
                'model': {
                    'framework': 'sklearn',
                    'model_type': 'random_forest_classifier',
                    'task_type': 'classification',
                    'parameters': {'n_estimators': 10}
                }
            },
            'mlflow': {
                'enabled': True,
                'experiment_name': 'test-experiment',
                'tracking_uri': 'http://localhost:5000'
            }
        }
        
        context = ExecutionContext(
            experiment_id="test-exp",
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config=config,
            artifacts_path=str(self.temp_path),
            logger=trainer.logger
        )
        
        trainer.setup(context)
        
        assert trainer.mlflow_tracker == mock_tracker
        mock_tracker_class.assert_called_once()
    
    def test_execute_without_mlflow(self):
        """Test execution without MLflow integration."""
        trainer = MLflowIntegratedTrainer()
        
        config = {
            'training': {
                'model': {
                    'framework': 'sklearn',
                    'model_type': 'random_forest_classifier',
                    'task_type': 'classification',
                    'parameters': {'n_estimators': 10, 'random_state': 42}
                },
                'target_column': 'target'
            }
        }
        
        context = ExecutionContext(
            experiment_id="test-exp",
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config=config,
            artifacts_path=str(self.temp_path),
            logger=trainer.logger
        )
        
        result = trainer.execute(context)
        
        assert result.success is True
        assert 'mlflow_run_id' not in result.metadata


class TestMLflowIntegratedEvaluator:
    """Test MLflowIntegratedEvaluator functionality."""
    
    def setup_method(self):
        """Setup test data and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock model and test data files
        import pickle
        from sklearn.ensemble import RandomForestClassifier
        
        # Create and save a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.randn(10, 5)
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        
        with open(self.temp_path / "test_model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # Create test data
        np.random.seed(42)
        X_test = np.random.randn(50, 5)
        y_test = np.random.randint(0, 2, 50)
        
        test_data = {
            'X_test': pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(5)]),
            'y_test': pd.Series(y_test),
            'task_type': 'classification'
        }
        
        with open(self.temp_path / "test_test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    @patch('mlpipeline.models.mlflow_integration.MLflowTracker')
    def test_setup_with_mlflow(self, mock_tracker_class):
        """Test setup with MLflow integration enabled."""
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        evaluator = MLflowIntegratedEvaluator()
        
        config = {
            'evaluation': {
                'metrics': ['accuracy', 'f1_score']
            },
            'mlflow': {
                'enabled': True,
                'experiment_name': 'test-evaluation'
            }
        }
        
        context = ExecutionContext(
            experiment_id="test-exp",
            stage_name="evaluation",
            component_type=ComponentType.MODEL_EVALUATION,
            config=config,
            artifacts_path=str(self.temp_path),
            logger=evaluator.logger
        )
        
        evaluator.setup(context)
        
        assert evaluator.mlflow_tracker == mock_tracker
        mock_tracker_class.assert_called_once()
    
    def test_execute_without_mlflow(self):
        """Test execution without MLflow integration."""
        evaluator = MLflowIntegratedEvaluator()
        
        config = {
            'evaluation': {
                'metrics': ['accuracy', 'f1_score']
            },
            'training': {
                'model': {
                    'task_type': 'classification'
                }
            }
        }
        
        context = ExecutionContext(
            experiment_id="test-exp",
            stage_name="evaluation",
            component_type=ComponentType.MODEL_EVALUATION,
            config=config,
            artifacts_path=str(self.temp_path),
            logger=evaluator.logger
        )
        
        result = evaluator.execute(context)
        
        assert result.success is True
        assert 'mlflow_run_id' not in result.metadata


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestMLflowIntegratedHyperparameterTrainer:
    """Test MLflowIntegratedHyperparameterTrainer functionality."""
    
    def setup_method(self):
        """Setup test data and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save as training data
        df.to_parquet(self.temp_path / "train_preprocessed.parquet", index=False)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    @patch('mlpipeline.models.mlflow_integration.MLflowTracker')
    def test_setup_with_mlflow(self, mock_tracker_class):
        """Test setup with MLflow integration enabled."""
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        trainer = MLflowIntegratedHyperparameterTrainer()
        
        config = {
            'training': {
                'model': {
                    'framework': 'sklearn',
                    'model_type': 'random_forest_classifier',
                    'task_type': 'classification',
                    'parameters': {'random_state': 42}
                },
                'hyperparameter_optimization': {
                    'enabled': True,
                    'n_trials': 5,
                    'metric': 'accuracy'
                }
            },
            'mlflow': {
                'enabled': True,
                'experiment_name': 'test-hyperopt'
            }
        }
        
        context = ExecutionContext(
            experiment_id="test-exp",
            stage_name="hyperopt",
            component_type=ComponentType.MODEL_TRAINING,
            config=config,
            artifacts_path=str(self.temp_path),
            logger=trainer.logger
        )
        
        trainer.setup(context)
        
        assert trainer.mlflow_tracker == mock_tracker
        mock_tracker_class.assert_called_once()


class TestMLflowIntegrationErrors:
    """Test error handling in MLflow integration."""
    
    @pytest.mark.skipif(MLFLOW_AVAILABLE, reason="Test for when MLflow is not available")
    def test_mlflow_tracker_without_mlflow(self):
        """Test MLflowTracker creation when MLflow is not installed."""
        config = MLflowConfig()
        
        with pytest.raises(ModelError, match="MLflow is not installed"):
            MLflowTracker(config)
    
    def test_invalid_mlflow_config(self):
        """Test handling of invalid MLflow configuration."""
        # Test with invalid model stage
        config = MLflowConfig(model_stage="InvalidStage")
        
        # This should not raise an error during config creation
        assert config.model_stage == "InvalidStage"
    
    def test_mlflow_config_with_none_values(self):
        """Test MLflowConfig with None values."""
        config = MLflowConfig(
            tracking_uri=None,
            registry_uri=None,
            run_name=None,
            model_name=None
        )
        
        assert config.tracking_uri is None
        assert config.registry_uri is None
        assert config.run_name is None
        assert config.model_name is None


if __name__ == "__main__":
    pytest.main([__file__])