"""Integration tests for MLflow and DVC components."""

import pytest
import tempfile
import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import os

from mlpipeline.models.mlflow_integration import MLflowTracker
from mlpipeline.data.versioning import DataVersionManager
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
from mlpipeline.models.training import ModelTrainer


class TestMLflowIntegration:
    """Test MLflow integration functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mlflow_integration = MLflowTracker()
        
        # Create sample training data
        np.random.seed(42)
        self.train_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        self.val_data = pd.DataFrame({
            'feature_1': np.random.randn(20),
            'feature_2': np.random.randn(20),
            'target': np.random.randint(0, 2, 20)
        })
        
        # Save training data
        self.train_data.to_parquet(Path(self.temp_dir) / "train_preprocessed.parquet", index=False)
        self.val_data.to_parquet(Path(self.temp_dir) / "val_preprocessed.parquet", index=False)
        
        # Create execution context
        self.context = ExecutionContext(
            experiment_id="mlflow_test",
            stage_name="mlflow_integration",
            component_type=ComponentType.MODEL_TRAINING,
            config={
                'mlflow': {
                    'tracking_uri': f"file://{self.temp_dir}/mlruns",
                    'experiment_name': "test_experiment",
                    'run_name': "test_run",
                    'tags': {"environment": "test", "version": "1.0"}
                },
                'training': {
                    'model': {
                        'framework': 'sklearn',
                        'model_type': 'logistic_regression',
                        'task_type': 'classification',
                        'parameters': {'random_state': 42}
                    },
                    'target_column': 'target'
                }
            },
            artifacts_path=self.temp_dir,
            logger=Mock(),
            metadata={}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('mlpipeline.models.mlflow_integration.mlflow')
    def test_mlflow_experiment_setup(self, mock_mlflow):
        """Test MLflow experiment setup."""
        # Mock MLflow methods
        mock_mlflow.set_tracking_uri = Mock()
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.start_run = Mock()
        mock_mlflow.end_run = Mock()
        mock_mlflow.log_param = Mock()
        mock_mlflow.log_metric = Mock()
        mock_mlflow.log_artifact = Mock()
        mock_mlflow.set_tag = Mock()
        
        # Mock active run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        
        result = self.mlflow_integration.execute(self.context)
        
        assert result.success is True
        
        # Verify MLflow setup calls
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.start_run.assert_called_once()
        
        # Verify tags were set
        expected_tags = {"environment": "test", "version": "1.0"}
        for key, value in expected_tags.items():
            mock_mlflow.set_tag.assert_any_call(key, value)
    
    @patch('mlpipeline.models.mlflow_integration.mlflow')
    def test_mlflow_parameter_logging(self, mock_mlflow):
        """Test MLflow parameter logging."""
        # Mock MLflow methods
        mock_mlflow.set_tracking_uri = Mock()
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.start_run = Mock()
        mock_mlflow.end_run = Mock()
        mock_mlflow.log_param = Mock()
        mock_mlflow.log_metric = Mock()
        mock_mlflow.log_artifact = Mock()
        mock_mlflow.set_tag = Mock()
        
        # Mock active run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        
        result = self.mlflow_integration.execute(self.context)
        
        assert result.success is True
        
        # Verify parameters were logged
        expected_params = {
            'framework': 'sklearn',
            'model_type': 'logistic_regression',
            'task_type': 'classification',
            'random_state': 42
        }
        
        for key, value in expected_params.items():
            mock_mlflow.log_param.assert_any_call(key, value)
    
    @patch('mlpipeline.models.mlflow_integration.mlflow')
    def test_mlflow_metric_logging(self, mock_mlflow):
        """Test MLflow metric logging."""
        # Mock MLflow methods
        mock_mlflow.set_tracking_uri = Mock()
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.start_run = Mock()
        mock_mlflow.end_run = Mock()
        mock_mlflow.log_param = Mock()
        mock_mlflow.log_metric = Mock()
        mock_mlflow.log_artifact = Mock()
        mock_mlflow.set_tag = Mock()
        
        # Mock active run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        
        # Add metrics to context
        self.context.metadata['metrics'] = {
            'train_accuracy': 0.85,
            'val_accuracy': 0.80,
            'train_loss': 0.15,
            'val_loss': 0.20
        }
        
        result = self.mlflow_integration.execute(self.context)
        
        assert result.success is True
        
        # Verify metrics were logged
        expected_metrics = {
            'train_accuracy': 0.85,
            'val_accuracy': 0.80,
            'train_loss': 0.15,
            'val_loss': 0.20
        }
        
        for key, value in expected_metrics.items():
            mock_mlflow.log_metric.assert_any_call(key, value)
    
    @patch('mlpipeline.models.mlflow_integration.mlflow')
    def test_mlflow_artifact_logging(self, mock_mlflow):
        """Test MLflow artifact logging."""
        # Mock MLflow methods
        mock_mlflow.set_tracking_uri = Mock()
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.start_run = Mock()
        mock_mlflow.end_run = Mock()
        mock_mlflow.log_param = Mock()
        mock_mlflow.log_metric = Mock()
        mock_mlflow.log_artifact = Mock()
        mock_mlflow.set_tag = Mock()
        
        # Mock active run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        
        # Create test artifacts
        model_path = Path(self.temp_dir) / "trained_model.joblib"
        model_path.write_text("mock model content")
        
        config_path = Path(self.temp_dir) / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump({"test": "config"}, f)
        
        self.context.metadata['artifacts'] = [str(model_path), str(config_path)]
        
        result = self.mlflow_integration.execute(self.context)
        
        assert result.success is True
        
        # Verify artifacts were logged
        mock_mlflow.log_artifact.assert_any_call(str(model_path))
        mock_mlflow.log_artifact.assert_any_call(str(config_path))
    
    @patch('mlpipeline.models.mlflow_integration.mlflow')
    def test_mlflow_model_registration(self, mock_mlflow):
        """Test MLflow model registration."""
        # Mock MLflow methods
        mock_mlflow.set_tracking_uri = Mock()
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.start_run = Mock()
        mock_mlflow.end_run = Mock()
        mock_mlflow.log_param = Mock()
        mock_mlflow.log_metric = Mock()
        mock_mlflow.log_artifact = Mock()
        mock_mlflow.set_tag = Mock()
        mock_mlflow.sklearn.log_model = Mock()
        
        # Mock active run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        
        # Add model registration config
        self.context.config['mlflow']['register_model'] = True
        self.context.config['mlflow']['model_name'] = "test_model"
        
        # Create mock model file
        model_path = Path(self.temp_dir) / "trained_model.joblib"
        model_path.write_text("mock model content")
        self.context.metadata['model_path'] = str(model_path)
        
        result = self.mlflow_integration.execute(self.context)
        
        assert result.success is True
        
        # Verify model was logged (would be called if model loading worked)
        # In real scenario, this would load and log the actual model
        assert result.metadata['mlflow_run_id'] == "test_run_id"
    
    def test_mlflow_integration_with_model_training(self):
        """Test MLflow integration with actual model training."""
        # Create a more complete training setup
        trainer = ModelTrainer()
        
        # Execute training first
        training_result = trainer.execute(self.context)
        assert training_result.success is True
        
        # Update context with training results
        self.context.metadata.update({
            'metrics': training_result.metrics,
            'artifacts': training_result.artifacts
        })
        
        # Mock MLflow for integration test
        with patch('mlpipeline.models.mlflow_integration.mlflow') as mock_mlflow:
            mock_mlflow.set_tracking_uri = Mock()
            mock_mlflow.set_experiment = Mock()
            mock_mlflow.start_run = Mock()
            mock_mlflow.end_run = Mock()
            mock_mlflow.log_param = Mock()
            mock_mlflow.log_metric = Mock()
            mock_mlflow.log_artifact = Mock()
            mock_mlflow.set_tag = Mock()
            
            # Mock active run
            mock_run = Mock()
            mock_run.info.run_id = "integration_run_id"
            mock_mlflow.active_run.return_value = mock_run
            
            # Execute MLflow integration
            mlflow_result = self.mlflow_integration.execute(self.context)
            
            assert mlflow_result.success is True
            assert mlflow_result.metadata['mlflow_run_id'] == "integration_run_id"
            
            # Verify training metrics were logged to MLflow
            mock_mlflow.log_metric.assert_called()
            mock_mlflow.log_param.assert_called()


class TestDVCIntegration:
    """Test DVC integration functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dvc_integration = DataVersionManager()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Save sample data
        self.data_path = Path(self.temp_dir) / "sample_data.csv"
        self.sample_data.to_csv(self.data_path, index=False)
        
        # Create execution context
        self.context = ExecutionContext(
            experiment_id="dvc_test",
            stage_name="data_versioning",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={
                'data_versioning': {
                    'enabled': True,
                    'remote_storage': f"{self.temp_dir}/dvc_remote",
                    'track_data': True,
                    'track_models': True
                },
                'data': {
                    'sources': [{'path': str(self.data_path)}]
                }
            },
            artifacts_path=self.temp_dir,
            logger=Mock(),
            metadata={}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('mlpipeline.data.versioning.subprocess.run')
    def test_dvc_initialization(self, mock_subprocess):
        """Test DVC repository initialization."""
        # Mock successful subprocess calls
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = self.dvc_integration.execute(self.context)
        
        assert result.success is True
        
        # Verify DVC init was called
        init_calls = [call for call in mock_subprocess.call_args_list 
                     if 'dvc init' in ' '.join(call[0][0])]
        assert len(init_calls) > 0
    
    @patch('mlpipeline.data.versioning.subprocess.run')
    def test_dvc_data_tracking(self, mock_subprocess):
        """Test DVC data file tracking."""
        # Mock successful subprocess calls
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = self.dvc_integration.execute(self.context)
        
        assert result.success is True
        
        # Verify DVC add was called for data files
        add_calls = [call for call in mock_subprocess.call_args_list 
                    if 'dvc add' in ' '.join(call[0][0])]
        assert len(add_calls) > 0
    
    @patch('mlpipeline.data.versioning.subprocess.run')
    def test_dvc_remote_setup(self, mock_subprocess):
        """Test DVC remote storage setup."""
        # Mock successful subprocess calls
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = self.dvc_integration.execute(self.context)
        
        assert result.success is True
        
        # Verify DVC remote was configured
        remote_calls = [call for call in mock_subprocess.call_args_list 
                       if 'dvc remote' in ' '.join(call[0][0])]
        assert len(remote_calls) > 0
    
    @patch('mlpipeline.data.versioning.subprocess.run')
    def test_dvc_pipeline_creation(self, mock_subprocess):
        """Test DVC pipeline creation."""
        # Mock successful subprocess calls
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Add pipeline stages to context
        self.context.config['dvc_pipeline'] = {
            'stages': [
                {
                    'name': 'preprocess',
                    'cmd': 'python preprocess.py',
                    'deps': ['data/raw.csv'],
                    'outs': ['data/processed.csv']
                }
            ]
        }
        
        result = self.dvc_integration.execute(self.context)
        
        assert result.success is True
        
        # Verify DVC pipeline was created
        pipeline_calls = [call for call in mock_subprocess.call_args_list 
                         if 'dvc stage add' in ' '.join(call[0][0]) or 'dvc run' in ' '.join(call[0][0])]
        assert len(pipeline_calls) > 0
    
    @patch('mlpipeline.data.versioning.subprocess.run')
    def test_dvc_error_handling(self, mock_subprocess):
        """Test DVC error handling."""
        # Mock failed subprocess call
        mock_subprocess.return_value = Mock(returncode=1, stdout="", stderr="DVC error")
        
        result = self.dvc_integration.execute(self.context)
        
        assert result.success is False
        assert "DVC error" in result.error_message or "failed" in result.error_message.lower()
    
    @patch('mlpipeline.data.versioning.subprocess.run')
    def test_dvc_version_tagging(self, mock_subprocess):
        """Test DVC version tagging."""
        # Mock successful subprocess calls
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Add version tag to context
        self.context.config['data_versioning']['version_tag'] = "v1.0.0"
        
        result = self.dvc_integration.execute(self.context)
        
        assert result.success is True
        
        # Verify git tag was created
        tag_calls = [call for call in mock_subprocess.call_args_list 
                    if 'git tag' in ' '.join(call[0][0])]
        assert len(tag_calls) > 0
    
    def test_dvc_integration_with_preprocessing(self):
        """Test DVC integration with data preprocessing."""
        from mlpipeline.data.preprocessing import DataPreprocessor
        
        # Create preprocessed data files
        train_data = self.sample_data.iloc[:3]
        val_data = self.sample_data.iloc[3:]
        
        train_path = Path(self.temp_dir) / "train_preprocessed.parquet"
        val_path = Path(self.temp_dir) / "val_preprocessed.parquet"
        
        train_data.to_parquet(train_path, index=False)
        val_data.to_parquet(val_path, index=False)
        
        # Update context with preprocessing artifacts
        self.context.metadata['artifacts'] = [str(train_path), str(val_path)]
        
        with patch('mlpipeline.data.versioning.subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
            
            result = self.dvc_integration.execute(self.context)
            
            assert result.success is True
            
            # Verify preprocessing artifacts were tracked
            add_calls = [call for call in mock_subprocess.call_args_list 
                        if 'dvc add' in ' '.join(call[0][0])]
            assert len(add_calls) > 0


class TestMLflowDVCIntegration:
    """Test combined MLflow and DVC integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Save sample data
        self.data_path = Path(self.temp_dir) / "sample_data.csv"
        self.sample_data.to_csv(self.data_path, index=False)
        
        # Create execution context with both MLflow and DVC config
        self.context = ExecutionContext(
            experiment_id="combined_test",
            stage_name="integration",
            component_type=ComponentType.MODEL_TRAINING,
            config={
                'mlflow': {
                    'tracking_uri': f"file://{self.temp_dir}/mlruns",
                    'experiment_name': "combined_experiment",
                    'run_name': "combined_run"
                },
                'data_versioning': {
                    'enabled': True,
                    'remote_storage': f"{self.temp_dir}/dvc_remote",
                    'track_data': True,
                    'track_models': True
                },
                'training': {
                    'model': {
                        'framework': 'sklearn',
                        'model_type': 'logistic_regression',
                        'task_type': 'classification',
                        'parameters': {'random_state': 42}
                    },
                    'target_column': 'target'
                }
            },
            artifacts_path=self.temp_dir,
            logger=Mock(),
            metadata={}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('mlpipeline.models.mlflow_integration.mlflow')
    @patch('mlpipeline.data.versioning.subprocess.run')
    def test_combined_mlflow_dvc_workflow(self, mock_subprocess, mock_mlflow):
        """Test combined MLflow and DVC workflow."""
        # Mock MLflow
        mock_mlflow.set_tracking_uri = Mock()
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.start_run = Mock()
        mock_mlflow.end_run = Mock()
        mock_mlflow.log_param = Mock()
        mock_mlflow.log_metric = Mock()
        mock_mlflow.log_artifact = Mock()
        mock_mlflow.set_tag = Mock()
        
        mock_run = Mock()
        mock_run.info.run_id = "combined_run_id"
        mock_mlflow.active_run.return_value = mock_run
        
        # Mock DVC
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Execute model training first
        from mlpipeline.models.training import ModelTrainer
        
        # Create training data
        train_data = self.sample_data.iloc[:80]
        val_data = self.sample_data.iloc[80:]
        
        train_path = Path(self.temp_dir) / "train_preprocessed.parquet"
        val_path = Path(self.temp_dir) / "val_preprocessed.parquet"
        
        train_data.to_parquet(train_path, index=False)
        val_data.to_parquet(val_path, index=False)
        
        trainer = ModelTrainer()
        training_result = trainer.execute(self.context)
        
        assert training_result.success is True
        
        # Execute DVC integration
        dvc_integration = DataVersionManager()
        self.context.metadata['artifacts'] = training_result.artifacts
        dvc_result = dvc_integration.execute(self.context)
        
        assert dvc_result.success is True
        
        # Execute MLflow integration
        mlflow_integration = MLflowTracker()
        self.context.metadata['metrics'] = training_result.metrics
        mlflow_result = mlflow_integration.execute(self.context)
        
        assert mlflow_result.success is True
        
        # Verify both systems were used
        assert mock_mlflow.start_run.called
        assert mock_subprocess.called
        
        # Verify artifacts were tracked in both systems
        mock_mlflow.log_artifact.assert_called()
        dvc_add_calls = [call for call in mock_subprocess.call_args_list 
                        if 'dvc add' in ' '.join(call[0][0])]
        assert len(dvc_add_calls) > 0
    
    @patch('mlpipeline.models.mlflow_integration.mlflow')
    @patch('mlpipeline.data.versioning.subprocess.run')
    def test_experiment_reproducibility_tracking(self, mock_subprocess, mock_mlflow):
        """Test experiment reproducibility with both MLflow and DVC."""
        # Mock MLflow
        mock_mlflow.set_tracking_uri = Mock()
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.start_run = Mock()
        mock_mlflow.end_run = Mock()
        mock_mlflow.log_param = Mock()
        mock_mlflow.log_metric = Mock()
        mock_mlflow.log_artifact = Mock()
        mock_mlflow.set_tag = Mock()
        
        mock_run = Mock()
        mock_run.info.run_id = "repro_run_id"
        mock_mlflow.active_run.return_value = mock_run
        
        # Mock DVC
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Add reproducibility metadata
        self.context.metadata.update({
            'git_commit': 'abc123',
            'python_version': '3.8.0',
            'dependencies': {'scikit-learn': '1.0.0', 'pandas': '1.3.0'}
        })
        
        # Execute integrations
        mlflow_integration = MLflowTracker()
        dvc_integration = DataVersionManager()
        
        mlflow_result = mlflow_integration.execute(self.context)
        dvc_result = dvc_integration.execute(self.context)
        
        assert mlflow_result.success is True
        assert dvc_result.success is True
        
        # Verify reproducibility information was logged
        mock_mlflow.set_tag.assert_any_call('git_commit', 'abc123')
        mock_mlflow.set_tag.assert_any_call('python_version', '3.8.0')
        
        # Verify DVC tracked the experiment
        git_calls = [call for call in mock_subprocess.call_args_list 
                    if 'git' in call[0][0][0]]
        assert len(git_calls) > 0