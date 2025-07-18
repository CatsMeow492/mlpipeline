"""Integration tests for drift detection with synthetic data."""

import pytest
import tempfile
import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
from datetime import datetime, timedelta

from mlpipeline.monitoring.drift_detection import DriftDetector
from mlpipeline.monitoring.alerts import AlertManager
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
from mlpipeline.models.inference import ModelInferenceEngine


class TestDriftDetectionIntegration:
    """Test drift detection integration with synthetic data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.drift_detector = DriftDetector()
        
        # Create baseline data (no drift)
        np.random.seed(42)
        self.baseline_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000),
            'feature_3': np.random.exponential(1, 1000),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),
            'target': np.random.randint(0, 2, 1000)
        })
        
        # Save baseline data
        self.baseline_path = Path(self.temp_dir) / "baseline_data.parquet"
        self.baseline_data.to_parquet(self.baseline_path, index=False)
        
        # Create execution context
        self.context = ExecutionContext(
            experiment_id="drift_test",
            stage_name="drift_detection",
            component_type=ComponentType.DRIFT_DETECTION,
            config={
                'drift_detection': {
                    'enabled': True,
                    'baseline_data': str(self.baseline_path),
                    'methods': ['evidently', 'kl_divergence', 'psi'],
                    'thresholds': {
                        'data_drift': 0.1,
                        'prediction_drift': 0.05,
                        'feature_drift': 0.15
                    },
                    'numerical_features': ['feature_1', 'feature_2', 'feature_3'],
                    'categorical_features': ['categorical_feature'],
                    'target_column': 'target'
                },
                'alerts': {
                    'enabled': True,
                    'channels': ['email', 'slack'],
                    'email': {
                        'recipients': ['test@example.com'],
                        'smtp_server': 'localhost'
                    },
                    'slack': {
                        'webhook_url': 'https://hooks.slack.com/test'
                    }
                }
            },
            artifacts_path=self.temp_dir,
            logger=Mock(),
            metadata={}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_drifted_data(self, drift_type='feature', drift_magnitude=2.0):
        """Create synthetic drifted data."""
        np.random.seed(123)  # Different seed for drift
        
        if drift_type == 'feature':
            # Feature drift: shift mean of feature_1
            drifted_data = pd.DataFrame({
                'feature_1': np.random.normal(drift_magnitude, 1, 500),  # Shifted mean
                'feature_2': np.random.normal(5, 2, 500),  # No change
                'feature_3': np.random.exponential(1, 500),  # No change
                'categorical_feature': np.random.choice(['A', 'B', 'C'], 500, p=[0.5, 0.3, 0.2]),
                'target': np.random.randint(0, 2, 500)
            })
        elif drift_type == 'covariate':
            # Covariate drift: change distribution of categorical feature
            drifted_data = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, 500),
                'feature_2': np.random.normal(5, 2, 500),
                'feature_3': np.random.exponential(1, 500),
                'categorical_feature': np.random.choice(['A', 'B', 'C'], 500, p=[0.2, 0.3, 0.5]),  # Changed distribution
                'target': np.random.randint(0, 2, 500)
            })
        elif drift_type == 'concept':
            # Concept drift: change relationship between features and target
            features = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, 500),
                'feature_2': np.random.normal(5, 2, 500),
                'feature_3': np.random.exponential(1, 500),
                'categorical_feature': np.random.choice(['A', 'B', 'C'], 500, p=[0.5, 0.3, 0.2])
            })
            # Reverse the relationship for concept drift
            target = (features['feature_1'] < 0).astype(int)  # Opposite of original relationship
            drifted_data = features.copy()
            drifted_data['target'] = target
        else:
            # No drift
            drifted_data = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, 500),
                'feature_2': np.random.normal(5, 2, 500),
                'feature_3': np.random.exponential(1, 500),
                'categorical_feature': np.random.choice(['A', 'B', 'C'], 500, p=[0.5, 0.3, 0.2]),
                'target': np.random.randint(0, 2, 500)
            })
        
        return drifted_data
    
    def test_no_drift_detection(self):
        """Test drift detection with no drift present."""
        # Create current data with no drift
        current_data = self.create_drifted_data(drift_type='none')
        current_path = Path(self.temp_dir) / "current_data.parquet"
        current_data.to_parquet(current_path, index=False)
        
        # Update context with current data
        self.context.config['drift_detection']['current_data'] = str(current_path)
        
        result = self.drift_detector.execute(self.context)
        
        assert result.success is True
        assert result.metadata['drift_detected'] is False
        assert result.metrics['overall_drift_score'] < self.context.config['drift_detection']['thresholds']['data_drift']
        
        # Verify drift report was created
        drift_report_path = Path(self.temp_dir) / "drift_report.json"
        assert drift_report_path.exists()
        
        with open(drift_report_path, 'r') as f:
            report = json.load(f)
            assert report['drift_detected'] is False
            assert 'feature_drift_scores' in report
    
    def test_feature_drift_detection(self):
        """Test detection of feature drift."""
        # Create data with feature drift
        drifted_data = self.create_drifted_data(drift_type='feature', drift_magnitude=3.0)
        current_path = Path(self.temp_dir) / "drifted_data.parquet"
        drifted_data.to_parquet(current_path, index=False)
        
        # Update context with drifted data
        self.context.config['drift_detection']['current_data'] = str(current_path)
        
        result = self.drift_detector.execute(self.context)
        
        assert result.success is True
        assert result.metadata['drift_detected'] is True
        assert result.metrics['overall_drift_score'] > self.context.config['drift_detection']['thresholds']['data_drift']
        
        # Verify specific feature drift was detected
        assert 'feature_1_drift_score' in result.metrics
        assert result.metrics['feature_1_drift_score'] > self.context.config['drift_detection']['thresholds']['feature_drift']
        
        # Verify drift report contains details
        drift_report_path = Path(self.temp_dir) / "drift_report.json"
        assert drift_report_path.exists()
        
        with open(drift_report_path, 'r') as f:
            report = json.load(f)
            assert report['drift_detected'] is True
            assert 'feature_1' in report['feature_drift_scores']
            assert report['feature_drift_scores']['feature_1'] > 0.15
    
    def test_covariate_drift_detection(self):
        """Test detection of covariate drift."""
        # Create data with covariate drift
        drifted_data = self.create_drifted_data(drift_type='covariate')
        current_path = Path(self.temp_dir) / "covariate_drifted_data.parquet"
        drifted_data.to_parquet(current_path, index=False)
        
        # Update context with drifted data
        self.context.config['drift_detection']['current_data'] = str(current_path)
        
        result = self.drift_detector.execute(self.context)
        
        assert result.success is True
        assert result.metadata['drift_detected'] is True
        
        # Verify categorical feature drift was detected
        assert 'categorical_feature_drift_score' in result.metrics
        assert result.metrics['categorical_feature_drift_score'] > self.context.config['drift_detection']['thresholds']['feature_drift']
    
    def test_concept_drift_detection(self):
        """Test detection of concept drift."""
        # Create data with concept drift
        drifted_data = self.create_drifted_data(drift_type='concept')
        current_path = Path(self.temp_dir) / "concept_drifted_data.parquet"
        drifted_data.to_parquet(current_path, index=False)
        
        # Update context with drifted data
        self.context.config['drift_detection']['current_data'] = str(current_path)
        
        result = self.drift_detector.execute(self.context)
        
        assert result.success is True
        
        # Concept drift might be detected through target distribution changes
        if 'target_drift_score' in result.metrics:
            # If target drift is measured, it should be significant
            assert result.metrics['target_drift_score'] > 0.1
    
    def test_multiple_drift_methods(self):
        """Test using multiple drift detection methods."""
        # Create data with moderate drift
        drifted_data = self.create_drifted_data(drift_type='feature', drift_magnitude=1.5)
        current_path = Path(self.temp_dir) / "multi_method_data.parquet"
        drifted_data.to_parquet(current_path, index=False)
        
        # Update context with drifted data
        self.context.config['drift_detection']['current_data'] = str(current_path)
        
        result = self.drift_detector.execute(self.context)
        
        assert result.success is True
        
        # Verify multiple methods were used
        assert 'kl_divergence_score' in result.metrics or 'evidently_score' in result.metrics
        assert 'psi_score' in result.metrics or 'overall_drift_score' in result.metrics
        
        # Verify drift report contains method-specific results
        drift_report_path = Path(self.temp_dir) / "drift_report.json"
        assert drift_report_path.exists()
        
        with open(drift_report_path, 'r') as f:
            report = json.load(f)
            assert 'methods_used' in report
            assert len(report['methods_used']) > 1
    
    def test_drift_detection_with_time_series(self):
        """Test drift detection with time series data."""
        # Create time series data with gradual drift
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Gradual drift over time
        drift_factor = np.linspace(0, 2, len(dates))
        
        time_series_data = pd.DataFrame({
            'date': dates,
            'feature_1': np.random.normal(0, 1, len(dates)) + drift_factor,
            'feature_2': np.random.normal(5, 2, len(dates)),
            'feature_3': np.random.exponential(1, len(dates)),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], len(dates), p=[0.5, 0.3, 0.2]),
            'target': np.random.randint(0, 2, len(dates))
        })
        
        # Split into baseline (first 6 months) and current (last 3 months)
        baseline_ts = time_series_data[time_series_data['date'] < '2023-07-01'].drop('date', axis=1)
        current_ts = time_series_data[time_series_data['date'] >= '2023-10-01'].drop('date', axis=1)
        
        # Save time series data
        baseline_ts_path = Path(self.temp_dir) / "baseline_ts.parquet"
        current_ts_path = Path(self.temp_dir) / "current_ts.parquet"
        
        baseline_ts.to_parquet(baseline_ts_path, index=False)
        current_ts.to_parquet(current_ts_path, index=False)
        
        # Update context for time series
        self.context.config['drift_detection']['baseline_data'] = str(baseline_ts_path)
        self.context.config['drift_detection']['current_data'] = str(current_ts_path)
        
        result = self.drift_detector.execute(self.context)
        
        assert result.success is True
        assert result.metadata['drift_detected'] is True
        
        # Should detect drift in feature_1 due to gradual shift
        assert result.metrics['feature_1_drift_score'] > self.context.config['drift_detection']['thresholds']['feature_drift']
    
    def test_drift_detection_with_missing_values(self):
        """Test drift detection with missing values."""
        # Create data with missing values
        drifted_data = self.create_drifted_data(drift_type='feature', drift_magnitude=2.0)
        
        # Introduce missing values
        missing_indices = np.random.choice(drifted_data.index, size=50, replace=False)
        drifted_data.loc[missing_indices, 'feature_1'] = np.nan
        drifted_data.loc[missing_indices[:25], 'categorical_feature'] = np.nan
        
        current_path = Path(self.temp_dir) / "missing_values_data.parquet"
        drifted_data.to_parquet(current_path, index=False)
        
        # Update context
        self.context.config['drift_detection']['current_data'] = str(current_path)
        self.context.config['drift_detection']['handle_missing'] = 'drop'
        
        result = self.drift_detector.execute(self.context)
        
        assert result.success is True
        # Should still detect drift despite missing values
        assert result.metadata['drift_detected'] is True
    
    def test_drift_detection_performance_metrics(self):
        """Test drift detection performance and timing."""
        # Create larger dataset for performance testing
        np.random.seed(42)
        large_baseline = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 10000),
            'feature_2': np.random.normal(5, 2, 10000),
            'feature_3': np.random.exponential(1, 10000),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 10000, p=[0.5, 0.3, 0.2]),
            'target': np.random.randint(0, 2, 10000)
        })
        
        large_current = self.create_drifted_data(drift_type='feature', drift_magnitude=1.0)
        # Expand current data
        large_current = pd.concat([large_current] * 10, ignore_index=True)
        
        # Save large datasets
        large_baseline_path = Path(self.temp_dir) / "large_baseline.parquet"
        large_current_path = Path(self.temp_dir) / "large_current.parquet"
        
        large_baseline.to_parquet(large_baseline_path, index=False)
        large_current.to_parquet(large_current_path, index=False)
        
        # Update context
        self.context.config['drift_detection']['baseline_data'] = str(large_baseline_path)
        self.context.config['drift_detection']['current_data'] = str(large_current_path)
        
        import time
        start_time = time.time()
        result = self.drift_detector.execute(self.context)
        execution_time = time.time() - start_time
        
        assert result.success is True
        assert execution_time < 30.0  # Should complete within 30 seconds
        assert result.execution_time is not None
        assert result.execution_time > 0
        
        # Verify performance metrics
        assert 'baseline_samples' in result.metadata
        assert 'current_samples' in result.metadata
        assert result.metadata['baseline_samples'] == 10000
        assert result.metadata['current_samples'] == 5000


class TestDriftDetectionWithAlerting:
    """Test drift detection integration with alerting system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager()
        
        # Create baseline data
        np.random.seed(42)
        self.baseline_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000),
            'target': np.random.randint(0, 2, 1000)
        })
        
        # Save baseline data
        self.baseline_path = Path(self.temp_dir) / "baseline_data.parquet"
        self.baseline_data.to_parquet(self.baseline_path, index=False)
        
        # Create execution context
        self.context = ExecutionContext(
            experiment_id="drift_alert_test",
            stage_name="drift_monitoring",
            component_type=ComponentType.DRIFT_DETECTION,
            config={
                'drift_detection': {
                    'enabled': True,
                    'baseline_data': str(self.baseline_path),
                    'methods': ['kl_divergence'],
                    'thresholds': {
                        'data_drift': 0.1,
                        'feature_drift': 0.15
                    },
                    'numerical_features': ['feature_1', 'feature_2'],
                    'target_column': 'target'
                },
                'alerts': {
                    'enabled': True,
                    'drift_threshold': 0.1,
                    'channels': ['email'],
                    'email': {
                        'recipients': ['admin@example.com'],
                        'smtp_server': 'localhost',
                        'smtp_port': 587
                    }
                }
            },
            artifacts_path=self.temp_dir,
            logger=Mock(),
            metadata={}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('mlpipeline.monitoring.alerts.smtplib.SMTP')
    def test_drift_alert_triggered(self, mock_smtp):
        """Test that alerts are triggered when drift is detected."""
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Create drifted data
        np.random.seed(123)
        drifted_data = pd.DataFrame({
            'feature_1': np.random.normal(3, 1, 500),  # Significant drift
            'feature_2': np.random.normal(5, 2, 500),
            'target': np.random.randint(0, 2, 500)
        })
        
        current_path = Path(self.temp_dir) / "drifted_data.parquet"
        drifted_data.to_parquet(current_path, index=False)
        
        # Update context
        self.context.config['drift_detection']['current_data'] = str(current_path)
        
        # Execute drift detection
        drift_result = self.drift_detector.execute(self.context)
        
        assert drift_result.success is True
        assert drift_result.metadata['drift_detected'] is True
        
        # Execute alerting
        alert_context = self.context
        alert_context.metadata.update(drift_result.metadata)
        alert_context.metadata['drift_metrics'] = drift_result.metrics
        
        alert_result = self.alert_manager.execute(alert_context)
        
        assert alert_result.success is True
        assert alert_result.metadata['alerts_sent'] > 0
        
        # Verify email was sent
        mock_server.send_message.assert_called()
    
    @patch('mlpipeline.monitoring.alerts.smtplib.SMTP')
    def test_no_alert_when_no_drift(self, mock_smtp):
        """Test that no alerts are sent when no drift is detected."""
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Create data with no drift
        np.random.seed(42)  # Same seed as baseline
        no_drift_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 500),
            'feature_2': np.random.normal(5, 2, 500),
            'target': np.random.randint(0, 2, 500)
        })
        
        current_path = Path(self.temp_dir) / "no_drift_data.parquet"
        no_drift_data.to_parquet(current_path, index=False)
        
        # Update context
        self.context.config['drift_detection']['current_data'] = str(current_path)
        
        # Execute drift detection
        drift_result = self.drift_detector.execute(self.context)
        
        assert drift_result.success is True
        assert drift_result.metadata['drift_detected'] is False
        
        # Execute alerting
        alert_context = self.context
        alert_context.metadata.update(drift_result.metadata)
        alert_context.metadata['drift_metrics'] = drift_result.metrics
        
        alert_result = self.alert_manager.execute(alert_context)
        
        assert alert_result.success is True
        assert alert_result.metadata.get('alerts_sent', 0) == 0
        
        # Verify no email was sent
        mock_server.send_message.assert_not_called()
    
    @patch('requests.post')
    def test_slack_alert_integration(self, mock_post):
        """Test Slack alert integration."""
        # Mock successful Slack webhook
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"ok": True}
        
        # Update context for Slack alerts
        self.context.config['alerts']['channels'] = ['slack']
        self.context.config['alerts']['slack'] = {
            'webhook_url': 'https://hooks.slack.com/test'
        }
        
        # Create drifted data
        np.random.seed(123)
        drifted_data = pd.DataFrame({
            'feature_1': np.random.normal(3, 1, 500),
            'feature_2': np.random.normal(5, 2, 500),
            'target': np.random.randint(0, 2, 500)
        })
        
        current_path = Path(self.temp_dir) / "slack_drift_data.parquet"
        drifted_data.to_parquet(current_path, index=False)
        
        # Update context
        self.context.config['drift_detection']['current_data'] = str(current_path)
        
        # Execute drift detection
        drift_result = self.drift_detector.execute(self.context)
        
        assert drift_result.success is True
        assert drift_result.metadata['drift_detected'] is True
        
        # Execute alerting
        alert_context = self.context
        alert_context.metadata.update(drift_result.metadata)
        alert_context.metadata['drift_metrics'] = drift_result.metrics
        
        alert_result = self.alert_manager.execute(alert_context)
        
        assert alert_result.success is True
        assert alert_result.metadata['alerts_sent'] > 0
        
        # Verify Slack webhook was called
        mock_post.assert_called()
        call_args = mock_post.call_args
        assert 'https://hooks.slack.com/test' in call_args[0]
    
    def test_alert_suppression(self):
        """Test alert suppression to prevent spam."""
        # Create alert history file to simulate recent alerts
        alert_history_path = Path(self.temp_dir) / "alert_history.json"
        recent_alert = {
            'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat(),
            'experiment_id': 'drift_alert_test',
            'alert_type': 'drift_detection',
            'severity': 'high'
        }
        
        with open(alert_history_path, 'w') as f:
            json.dump([recent_alert], f)
        
        # Update context with alert history
        self.context.config['alerts']['history_file'] = str(alert_history_path)
        self.context.config['alerts']['suppression_window'] = 60  # 60 minutes
        
        # Create drifted data
        np.random.seed(123)
        drifted_data = pd.DataFrame({
            'feature_1': np.random.normal(3, 1, 500),
            'feature_2': np.random.normal(5, 2, 500),
            'target': np.random.randint(0, 2, 500)
        })
        
        current_path = Path(self.temp_dir) / "suppressed_drift_data.parquet"
        drifted_data.to_parquet(current_path, index=False)
        
        # Update context
        self.context.config['drift_detection']['current_data'] = str(current_path)
        
        # Execute drift detection
        drift_result = self.drift_detector.execute(self.context)
        
        assert drift_result.success is True
        assert drift_result.metadata['drift_detected'] is True
        
        # Execute alerting
        alert_context = self.context
        alert_context.metadata.update(drift_result.metadata)
        alert_context.metadata['drift_metrics'] = drift_result.metrics
        
        with patch('mlpipeline.monitoring.alerts.smtplib.SMTP'):
            alert_result = self.alert_manager.execute(alert_context)
        
        assert alert_result.success is True
        # Alert should be suppressed due to recent alert
        assert alert_result.metadata.get('alerts_suppressed', 0) > 0