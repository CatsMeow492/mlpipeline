"""Tests for model evaluation and comparison system."""

import pytest
import tempfile
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from mlpipeline.models.evaluation import (
    EvaluationMetrics, ModelComparison, ModelEvaluator
)
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
from mlpipeline.core.errors import ModelError, ConfigurationError
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_regression


class TestEvaluationMetrics:
    """Test evaluation metrics dataclass."""
    
    def test_basic_metrics(self):
        """Test basic evaluation metrics."""
        metrics = EvaluationMetrics(
            task_type="classification",
            metrics={"accuracy": 0.85, "f1_score": 0.83}
        )
        
        assert metrics.task_type == "classification"
        assert metrics.metrics["accuracy"] == 0.85
        assert metrics.metrics["f1_score"] == 0.83
        assert metrics.confusion_matrix is None
        assert metrics.classification_report is None
    
    def test_complete_classification_metrics(self):
        """Test complete classification metrics."""
        cm = np.array([[10, 2], [3, 15]])
        roc_data = {'fpr': np.array([0, 0.1, 1]), 'tpr': np.array([0, 0.8, 1])}
        
        metrics = EvaluationMetrics(
            task_type="classification",
            metrics={"accuracy": 0.85, "roc_auc": 0.92},
            confusion_matrix=cm,
            classification_report={"0": {"precision": 0.77}},
            roc_curve_data=roc_data,
            feature_importance={"feature_0": 0.3, "feature_1": 0.7}
        )
        
        assert np.array_equal(metrics.confusion_matrix, cm)
        assert metrics.roc_curve_data['fpr'][1] == 0.1
        assert metrics.feature_importance["feature_1"] == 0.7
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        cm = np.array([[10, 2], [3, 15]])
        metrics = EvaluationMetrics(
            task_type="classification",
            metrics={"accuracy": 0.85},
            confusion_matrix=cm
        )
        
        result_dict = metrics.to_dict()
        
        assert result_dict["task_type"] == "classification"
        assert result_dict["metrics"]["accuracy"] == 0.85
        assert result_dict["confusion_matrix"] == cm.tolist()


class TestModelComparison:
    """Test model comparison dataclass."""
    
    def test_basic_comparison(self):
        """Test basic model comparison."""
        comparison = ModelComparison(
            model_names=["model_a", "model_b"],
            metrics_comparison={"accuracy": {"model_a": 0.85, "model_b": 0.82}},
            statistical_tests={"model_a_vs_model_b": {"p_value": 0.03}},
            best_model="model_a",
            ranking=["model_a", "model_b"]
        )
        
        assert comparison.model_names == ["model_a", "model_b"]
        assert comparison.best_model == "model_a"
        assert comparison.ranking[0] == "model_a"
        assert comparison.statistical_tests["model_a_vs_model_b"]["p_value"] == 0.03
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        comparison = ModelComparison(
            model_names=["model_a"],
            metrics_comparison={"accuracy": {"model_a": 0.85}},
            statistical_tests={},
            best_model="model_a",
            ranking=["model_a"]
        )
        
        result_dict = comparison.to_dict()
        
        assert result_dict["model_names"] == ["model_a"]
        assert result_dict["best_model"] == "model_a"


class TestModelEvaluator:
    """Test model evaluator component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.evaluator = ModelEvaluator()
        
        # Create sample classification data
        np.random.seed(42)
        X, y = make_classification(
            n_samples=200, n_features=4, n_classes=2, 
            n_redundant=0, random_state=42
        )
        
        # Split data
        split_idx = 150
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        # Train sample models
        self.model_lr = LogisticRegression(random_state=42)
        self.model_lr.fit(self.X_train, self.y_train)
        
        self.model_rf = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model_rf.fit(self.X_train, self.y_train)
        
        # Save models and test data
        with open(Path(self.temp_dir) / "logistic_model.pkl", 'wb') as f:
            pickle.dump(self.model_lr, f)
        
        with open(Path(self.temp_dir) / "random_forest_model.pkl", 'wb') as f:
            pickle.dump(self.model_rf, f)
        
        test_data = {
            'X_test': self.X_test,
            'y_test': self.y_test,
            'task_type': 'classification'
        }
        
        with open(Path(self.temp_dir) / "logistic_test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)
        
        with open(Path(self.temp_dir) / "random_forest_test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)
        
        # Create execution context
        self.context = ExecutionContext(
            experiment_id="test_exp",
            stage_name="evaluation",
            component_type=ComponentType.MODEL_EVALUATION,
            config={
                'evaluation': {
                    'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
                    'primary_metric': 'accuracy'
                }
            },
            artifacts_path=str(self.temp_dir),
            logger=Mock(),
            metadata={}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        assert self.evaluator.component_type == ComponentType.MODEL_EVALUATION
        assert len(self.evaluator.evaluation_results) == 0
        assert self.evaluator.comparison_results is None
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = {
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall']
            },
            'training': {
                'model': {
                    'task_type': 'classification'
                }
            }
        }
        
        assert self.evaluator.validate_config(config) is True
    
    def test_validate_config_invalid_metric(self):
        """Test configuration validation with invalid metric."""
        config = {
            'evaluation': {
                'metrics': ['accuracy', 'invalid_metric']
            },
            'training': {
                'model': {
                    'task_type': 'classification'
                }
            }
        }
        
        assert self.evaluator.validate_config(config) is False
    
    def test_validate_config_regression_metrics(self):
        """Test configuration validation with regression metrics."""
        config = {
            'evaluation': {
                'metrics': ['mse', 'mae', 'r2_score']
            },
            'training': {
                'model': {
                    'task_type': 'regression'
                }
            }
        }
        
        assert self.evaluator.validate_config(config) is True
    
    def test_setup_success(self):
        """Test successful setup."""
        self.evaluator.setup(self.context)
        # Setup should complete without errors
        assert True
    
    def test_setup_invalid_config(self):
        """Test setup with invalid configuration."""
        # Add invalid metric
        self.context.config['evaluation']['metrics'].append('invalid_metric')
        
        with pytest.raises(ConfigurationError):
            self.evaluator.setup(self.context)
    
    def test_load_models_and_data(self):
        """Test loading models and test data."""
        models_data = self.evaluator._load_models_and_data(self.context)
        
        assert len(models_data) == 2
        assert 'logistic' in models_data
        assert 'random_forest' in models_data
        
        # Check model data structure
        lr_data = models_data['logistic']
        assert 'model' in lr_data
        assert 'X_test' in lr_data
        assert 'y_test' in lr_data
        assert 'task_type' in lr_data
        assert lr_data['task_type'] == 'classification'
    
    def test_load_models_no_models(self):
        """Test loading when no models exist."""
        # Remove model files
        for file in Path(self.temp_dir).glob("*_model.pkl"):
            file.unlink()
        
        with pytest.raises(ModelError, match="No trained models found"):
            self.evaluator._load_models_and_data(self.context)
    
    def test_evaluate_single_model_classification(self):
        """Test evaluating a single classification model."""
        eval_config = {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        }
        
        result = self.evaluator._evaluate_single_model(
            self.model_lr, self.X_test, self.y_test, 'classification', eval_config
        )
        
        assert result.task_type == 'classification'
        assert 'accuracy' in result.metrics
        assert 'precision' in result.metrics
        assert 'recall' in result.metrics
        assert 'f1_score' in result.metrics
        assert 'roc_auc' in result.metrics
        
        # Check that all metrics are reasonable
        for metric_name, metric_value in result.metrics.items():
            assert 0 <= metric_value <= 1, f"Metric {metric_name} out of range: {metric_value}"
        
        # Check confusion matrix
        assert result.confusion_matrix is not None
        assert result.confusion_matrix.shape == (2, 2)
        
        # Check classification report
        assert result.classification_report is not None
        assert 'accuracy' in result.classification_report
        
        # Check ROC curve data
        assert result.roc_curve_data is not None
        assert 'fpr' in result.roc_curve_data
        assert 'tpr' in result.roc_curve_data
    
    def test_evaluate_single_model_regression(self):
        """Test evaluating a single regression model."""
        # Create regression data and model
        X_reg, y_reg = make_regression(n_samples=100, n_features=4, random_state=42)
        X_train_reg, X_test_reg = X_reg[:80], X_reg[80:]
        y_train_reg, y_test_reg = y_reg[:80], y_reg[80:]
        
        from sklearn.linear_model import LinearRegression
        model_reg = LinearRegression()
        model_reg.fit(X_train_reg, y_train_reg)
        
        eval_config = {
            'metrics': ['mse', 'mae', 'r2_score', 'rmse']
        }
        
        result = self.evaluator._evaluate_single_model(
            model_reg, X_test_reg, y_test_reg, 'regression', eval_config
        )
        
        assert result.task_type == 'regression'
        assert 'mse' in result.metrics
        assert 'mae' in result.metrics
        assert 'r2_score' in result.metrics
        assert 'rmse' in result.metrics
        
        # Check that MSE and RMSE are consistent
        assert abs(result.metrics['rmse'] - np.sqrt(result.metrics['mse'])) < 1e-6
        
        # R2 should be reasonable for this simple case
        assert result.metrics['r2_score'] > 0.5
    
    def test_compute_classification_metrics(self):
        """Test computing classification metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], 
                                [0.2, 0.8], [0.9, 0.1], [0.6, 0.4]])
        
        eval_config = {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        }
        
        metrics = self.evaluator._compute_classification_metrics(
            y_true, y_pred, y_pred_proba, eval_config
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Manual calculation for accuracy
        expected_accuracy = 4/6  # 4 correct out of 6
        assert abs(metrics['accuracy'] - expected_accuracy) < 1e-6
    
    def test_compute_regression_metrics(self):
        """Test computing regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        
        eval_config = {
            'metrics': ['mse', 'mae', 'r2_score', 'rmse', 'mape']
        }
        
        metrics = self.evaluator._compute_regression_metrics(
            y_true, y_pred, eval_config
        )
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        
        # Check RMSE calculation
        expected_rmse = np.sqrt(metrics['mse'])
        assert abs(metrics['rmse'] - expected_rmse) < 1e-6
    
    def test_get_feature_importance_tree_model(self):
        """Test feature importance extraction from tree-based model."""
        importance = self.evaluator._get_feature_importance(self.model_rf)
        
        assert importance is not None
        assert len(importance) == 4  # 4 features
        assert all(f'feature_{i}' in importance for i in range(4))
        assert all(isinstance(v, float) for v in importance.values())
        assert sum(importance.values()) > 0  # Should have some importance
    
    def test_get_feature_importance_linear_model(self):
        """Test feature importance extraction from linear model."""
        importance = self.evaluator._get_feature_importance(self.model_lr)
        
        assert importance is not None
        assert len(importance) == 4  # 4 features
        assert all(f'feature_{i}' in importance for i in range(4))
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_get_feature_importance_no_importance(self):
        """Test feature importance extraction from model without importance."""
        # Create a mock model without feature importance
        mock_model = Mock()
        del mock_model.feature_importances_
        del mock_model.coef_
        
        importance = self.evaluator._get_feature_importance(mock_model)
        assert importance is None
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create evaluation results for two models
        eval_results = {
            'model_a': EvaluationMetrics(
                task_type='classification',
                metrics={'accuracy': 0.85, 'f1_score': 0.83}
            ),
            'model_b': EvaluationMetrics(
                task_type='classification',
                metrics={'accuracy': 0.82, 'f1_score': 0.80}
            )
        }
        
        eval_config = {'primary_metric': 'accuracy'}
        
        comparison = self.evaluator._compare_models(eval_results, eval_config)
        
        assert comparison.model_names == ['model_a', 'model_b']
        assert comparison.best_model == 'model_a'  # Higher accuracy
        assert comparison.ranking[0] == 'model_a'
        
        # Check metrics comparison
        assert 'accuracy' in comparison.metrics_comparison
        assert comparison.metrics_comparison['accuracy']['model_a'] == 0.85
        assert comparison.metrics_comparison['accuracy']['model_b'] == 0.82
        
        # Check statistical tests
        assert 'model_a_vs_model_b' in comparison.statistical_tests
        test_result = comparison.statistical_tests['model_a_vs_model_b']
        assert 'p_value' in test_result
        assert 'difference' in test_result
        assert abs(test_result['difference'] - 0.03) < 1e-10  # 0.85 - 0.82
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_visualizations(self, mock_close, mock_savefig):
        """Test visualization generation."""
        eval_result = EvaluationMetrics(
            task_type='classification',
            metrics={'accuracy': 0.85, 'roc_auc': 0.92},
            confusion_matrix=np.array([[10, 2], [3, 15]]),
            roc_curve_data={'fpr': np.array([0, 0.1, 1]), 'tpr': np.array([0, 0.8, 1])},
            pr_curve_data={'precision': np.array([1, 0.8, 0.5]), 'recall': np.array([0, 0.5, 1])},
            feature_importance={'feature_0': 0.3, 'feature_1': 0.7}
        )
        
        self.evaluator._generate_visualizations(eval_result, 'test_model', self.temp_dir)
        
        # Should have called savefig multiple times (confusion matrix, ROC, PR curve, feature importance)
        assert mock_savefig.call_count >= 4
        assert mock_close.call_count >= 4
    
    def test_execute_success_single_model(self):
        """Test successful execution with single model."""
        # Remove one model to test single model case
        (Path(self.temp_dir) / "random_forest_model.pkl").unlink()
        (Path(self.temp_dir) / "random_forest_test_data.pkl").unlink()
        
        result = self.evaluator.execute(self.context)
        
        assert result.success is True
        assert len(result.artifacts) >= 2  # evaluation results + summary
        assert 'evaluation_time_seconds' in result.metrics
        assert 'models_evaluated' in result.metrics
        assert result.metrics['models_evaluated'] == 1
        
        # Check metadata
        assert result.metadata['models_evaluated'] == ['logistic']
        assert result.metadata['has_comparison'] is False
        assert result.metadata['visualizations_generated'] is True
    
    def test_execute_success_multiple_models(self):
        """Test successful execution with multiple models."""
        result = self.evaluator.execute(self.context)
        
        assert result.success is True
        assert len(result.artifacts) >= 3  # 2 model evaluations + comparison + summary
        assert 'evaluation_time_seconds' in result.metrics
        assert 'models_evaluated' in result.metrics
        assert result.metrics['models_evaluated'] == 2
        
        # Check metadata
        assert len(result.metadata['models_evaluated']) == 2
        assert result.metadata['has_comparison'] is True
        assert result.metadata['visualizations_generated'] is True
        
        # Check that comparison was performed
        assert self.evaluator.comparison_results is not None
        assert len(self.evaluator.comparison_results.model_names) == 2
    
    def test_execute_no_models(self):
        """Test execution with no models."""
        # Remove all model files
        for file in Path(self.temp_dir).glob("*_model.pkl"):
            file.unlink()
        
        result = self.evaluator.execute(self.context)
        
        assert result.success is False
        assert "No trained models found" in result.error_message
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results."""
        # Set up some evaluation results
        self.evaluator.evaluation_results = {
            'test_model': EvaluationMetrics(
                task_type='classification',
                metrics={'accuracy': 0.85}
            )
        }
        
        self.evaluator.comparison_results = ModelComparison(
            model_names=['test_model'],
            metrics_comparison={'accuracy': {'test_model': 0.85}},
            statistical_tests={},
            best_model='test_model',
            ranking=['test_model']
        )
        
        artifacts = self.evaluator._save_evaluation_results(self.context)
        
        assert len(artifacts) == 3  # model evaluation + comparison + summary
        
        # Check that files exist and contain expected data
        eval_file = Path(self.temp_dir) / "test_model_evaluation.json"
        assert eval_file.exists()
        
        with open(eval_file) as f:
            eval_data = json.load(f)
        assert eval_data['task_type'] == 'classification'
        assert eval_data['metrics']['accuracy'] == 0.85
        
        comparison_file = Path(self.temp_dir) / "model_comparison.json"
        assert comparison_file.exists()
        
        summary_file = Path(self.temp_dir) / "evaluation_summary.json"
        assert summary_file.exists()
        
        with open(summary_file) as f:
            summary_data = json.load(f)
        assert summary_data['best_model'] == 'test_model'
        assert 'test_model' in summary_data['models_evaluated']


class TestIntegrationEvaluation:
    """Integration tests for model evaluation system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create more realistic test scenario
        np.random.seed(42)
        X, y = make_classification(
            n_samples=1000, n_features=10, n_classes=3,
            n_informative=8, n_redundant=2, random_state=42
        )
        
        # Split data
        train_size = 600
        val_size = 200
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # Train multiple models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            # Save model
            with open(Path(self.temp_dir) / f"{name}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            # Save test data
            test_data = {
                'X_test': X_test,
                'y_test': y_test,
                'task_type': 'classification'
            }
            with open(Path(self.temp_dir) / f"{name}_test_data.pkl", 'wb') as f:
                pickle.dump(test_data, f)
        
        self.context = ExecutionContext(
            experiment_id="integration_test",
            stage_name="evaluation",
            component_type=ComponentType.MODEL_EVALUATION,
            config={
                'evaluation': {
                    'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
                    'primary_metric': 'f1_score'
                }
            },
            artifacts_path=str(self.temp_dir),
            logger=Mock(),
            metadata={}
        )
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        evaluator = ModelEvaluator()
        
        # Execute evaluation
        result = evaluator.execute(self.context)
        
        # Verify success
        assert result.success is True
        
        # Verify artifacts were created
        artifacts_path = Path(self.temp_dir)
        
        # Check individual model evaluation files
        lr_eval_file = artifacts_path / "logistic_regression_evaluation.json"
        rf_eval_file = artifacts_path / "random_forest_evaluation.json"
        assert lr_eval_file.exists()
        assert rf_eval_file.exists()
        
        # Check comparison file
        comparison_file = artifacts_path / "model_comparison.json"
        assert comparison_file.exists()
        
        # Check summary file
        summary_file = artifacts_path / "evaluation_summary.json"
        assert summary_file.exists()
        
        # Verify visualization files
        viz_files = list(artifacts_path.glob("*_confusion_matrix.png"))
        assert len(viz_files) >= 2  # One for each model
        
        # Verify content of evaluation results
        with open(lr_eval_file) as f:
            lr_data = json.load(f)
        
        assert lr_data['task_type'] == 'classification'
        assert 'accuracy' in lr_data['metrics']
        assert 'f1_score' in lr_data['metrics']
        assert lr_data['confusion_matrix'] is not None
        
        # Verify comparison results
        with open(comparison_file) as f:
            comparison_data = json.load(f)
        
        assert len(comparison_data['model_names']) == 2
        assert comparison_data['best_model'] in ['logistic_regression', 'random_forest']
        assert 'f1_score' in comparison_data['metrics_comparison']
        
        # Verify statistical tests were performed
        assert len(comparison_data['statistical_tests']) > 0
        test_key = list(comparison_data['statistical_tests'].keys())[0]
        assert 'p_value' in comparison_data['statistical_tests'][test_key]
        assert 'difference' in comparison_data['statistical_tests'][test_key]