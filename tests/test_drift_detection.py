"""
Tests for drift detection functionality using Evidently AI.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from mlpipeline.monitoring.drift_detection import DriftDetector, DriftMonitor
from mlpipeline.core.errors import PipelineError


class TestDriftDetector:
    """Test cases for DriftDetector class."""
    
    @pytest.fixture
    def sample_baseline_data(self):
        """Create sample baseline data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'feature3': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.choice([0, 1], 1000)
        })
    
    @pytest.fixture
    def sample_current_data_no_drift(self):
        """Create sample current data with no drift."""
        np.random.seed(43)
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 500),
            'feature2': np.random.normal(5, 2, 500),
            'feature3': np.random.choice(['A', 'B', 'C'], 500),
            'target': np.random.choice([0, 1], 500)
        })
    
    @pytest.fixture
    def sample_current_data_with_drift(self):
        """Create sample current data with drift."""
        np.random.seed(44)
        return pd.DataFrame({
            'feature1': np.random.normal(2, 1, 500),  # Mean shifted
            'feature2': np.random.normal(5, 4, 500),  # Variance increased
            'feature3': np.random.choice(['A', 'D', 'E'], 500),  # New categories
            'target': np.random.choice([0, 1], 500)
        })
    
    @pytest.fixture
    def drift_detector(self, sample_baseline_data):
        """Create DriftDetector instance with baseline data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            detector = DriftDetector(
                baseline_data=sample_baseline_data,
                output_dir=temp_dir
            )
            yield detector
    
    def test_drift_detector_initialization(self, sample_baseline_data):
        """Test DriftDetector initialization."""
        detector = DriftDetector(baseline_data=sample_baseline_data)
        
        assert detector.baseline_data is not None
        assert detector.baseline_data.shape == sample_baseline_data.shape
        assert detector.drift_thresholds['data_drift'] == 0.1
        assert detector.drift_thresholds['prediction_drift'] == 0.05
        assert detector.drift_thresholds['feature_drift'] == 0.1
    
    def test_drift_detector_custom_thresholds(self, sample_baseline_data):
        """Test DriftDetector with custom thresholds."""
        custom_thresholds = {
            'data_drift': 0.2,
            'prediction_drift': 0.1,
            'feature_drift': 0.15
        }
        
        detector = DriftDetector(
            baseline_data=sample_baseline_data,
            drift_thresholds=custom_thresholds
        )
        
        assert detector.drift_thresholds == custom_thresholds
    
    def test_set_baseline(self, sample_baseline_data):
        """Test setting baseline data."""
        detector = DriftDetector()
        detector.set_baseline(sample_baseline_data)
        
        assert detector.baseline_data is not None
        assert detector.baseline_data.shape == sample_baseline_data.shape
    
    def test_load_baseline_data_csv(self, sample_baseline_data):
        """Test loading baseline data from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_baseline_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            detector = DriftDetector(baseline_path=csv_path)
            assert detector.baseline_data is not None
            assert detector.baseline_data.shape[0] == sample_baseline_data.shape[0]
        finally:
            Path(csv_path).unlink()
    
    def test_load_baseline_data_invalid_format(self):
        """Test loading baseline data with invalid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid data")
            txt_path = f.name
        
        try:
            with pytest.raises(PipelineError) as exc_info:
                DriftDetector(baseline_path=txt_path)
            assert "Unsupported file format" in str(exc_info.value)
        finally:
            Path(txt_path).unlink()
    
    @patch('mlpipeline.monitoring.drift_detection.Report')
    def test_detect_data_drift_no_drift(self, mock_report, drift_detector, sample_current_data_no_drift):
        """Test data drift detection with no drift."""
        # Mock Evidently report
        mock_report_instance = Mock()
        mock_report.return_value = mock_report_instance
        mock_report_instance.as_dict.return_value = {
            'metrics': [{
                'metric': 'DatasetDriftMetric',
                'result': {
                    'drift_share': 0.05,
                    'number_of_drifted_columns': 1
                }
            }]
        }
        
        result = drift_detector.detect_data_drift(sample_current_data_no_drift, save_report=False)
        
        assert result['drift_detected'] is False
        assert result['drift_score'] == 0.05
        assert result['drift_type'] == 'data'
        assert result['drifted_columns'] == 1
        assert 'timestamp' in result
    
    @patch('mlpipeline.monitoring.drift_detection.Report')
    def test_detect_data_drift_with_drift(self, mock_report, drift_detector, sample_current_data_with_drift):
        """Test data drift detection with drift."""
        # Mock Evidently report
        mock_report_instance = Mock()
        mock_report.return_value = mock_report_instance
        mock_report_instance.as_dict.return_value = {
            'metrics': [{
                'metric': 'DatasetDriftMetric',
                'result': {
                    'drift_share': 0.15,
                    'number_of_drifted_columns': 3
                }
            }]
        }
        
        result = drift_detector.detect_data_drift(sample_current_data_with_drift, save_report=False)
        
        assert result['drift_detected'] is True
        assert result['drift_score'] == 0.15
        assert result['drift_type'] == 'data'
        assert result['drifted_columns'] == 3
    
    def test_detect_data_drift_no_baseline(self):
        """Test data drift detection without baseline data."""
        detector = DriftDetector()
        
        with pytest.raises(PipelineError) as exc_info:
            detector.detect_data_drift(pd.DataFrame({'col': [1, 2, 3]}))
        assert "No baseline data available" in str(exc_info.value)
    
    @patch('mlpipeline.monitoring.drift_detection.Report')
    def test_detect_prediction_drift(self, mock_report):
        """Test prediction drift detection."""
        # Mock Evidently report
        mock_report_instance = Mock()
        mock_report.return_value = mock_report_instance
        mock_report_instance.as_dict.return_value = {
            'metrics': [{
                'metric': 'DatasetDriftMetric',
                'result': {
                    'drift_share': 0.08,
                    'number_of_drifted_columns': 1
                }
            }]
        }
        
        detector = DriftDetector()
        baseline_predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        current_predictions = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        
        result = detector.detect_prediction_drift(
            baseline_predictions, 
            current_predictions, 
            save_report=False
        )
        
        assert result['drift_detected'] is True  # 0.08 > 0.05 threshold
        assert result['drift_score'] == 0.08
        assert result['drift_type'] == 'prediction'
    
    @patch('mlpipeline.monitoring.drift_detection.Report')
    def test_detect_feature_drift(self, mock_report, drift_detector, sample_current_data_no_drift):
        """Test feature-level drift detection."""
        # Mock Evidently report
        mock_report_instance = Mock()
        mock_report.return_value = mock_report_instance
        mock_report_instance.as_dict.return_value = {
            'metrics': [
                {
                    'metric': 'ColumnDriftMetric',
                    'result': {
                        'column_name': 'feature1',
                        'drift_score': 0.05,
                        'drift_detected': False,
                        'stattest_name': 'ks',
                        'threshold': 0.1,
                        'p_value': 0.2
                    }
                },
                {
                    'metric': 'ColumnDriftMetric',
                    'result': {
                        'column_name': 'feature2',
                        'drift_score': 0.15,
                        'drift_detected': True,
                        'stattest_name': 'ks',
                        'threshold': 0.1,
                        'p_value': 0.01
                    }
                }
            ]
        }
        
        result = drift_detector.detect_feature_drift(
            sample_current_data_no_drift, 
            feature_columns=['feature1', 'feature2'],
            save_report=False
        )
        
        assert result['drift_detected'] is False  # 1/2 = 0.5 > 0.1 threshold, but let's check logic
        assert result['drift_type'] == 'feature'
        assert result['total_features'] == 2
        assert result['drifted_features'] == 1
        assert 'feature1' in result['feature_results']
        assert 'feature2' in result['feature_results']
        assert result['feature_results']['feature1']['drift_detected'] is False
        assert result['feature_results']['feature2']['drift_detected'] is True
    
    @patch('mlpipeline.monitoring.drift_detection.TestSuite')
    def test_run_drift_test_suite(self, mock_test_suite, drift_detector, sample_current_data_no_drift):
        """Test drift test suite execution."""
        # Mock Evidently test suite
        mock_test_suite_instance = Mock()
        mock_test_suite.return_value = mock_test_suite_instance
        mock_test_suite_instance.as_dict.return_value = {
            'tests': [
                {
                    'name': 'TestColumnDrift',
                    'status': 'SUCCESS',
                    'description': 'Test column drift',
                    'parameters': {}
                },
                {
                    'name': 'TestShareOfMissingValues',
                    'status': 'FAIL',
                    'description': 'Test missing values',
                    'parameters': {}
                }
            ]
        }
        
        result = drift_detector.run_drift_test_suite(sample_current_data_no_drift, save_report=False)
        
        assert result['total_tests'] == 2
        assert result['tests_passed'] == 1
        assert result['tests_failed'] == 1
        assert result['success_rate'] == 0.5
        assert len(result['test_details']) == 2
    
    def test_get_baseline_statistics(self, drift_detector):
        """Test getting baseline statistics."""
        stats = drift_detector.get_baseline_statistics()
        
        assert 'shape' in stats
        assert 'columns' in stats
        assert 'dtypes' in stats
        assert 'missing_values' in stats
        assert 'numeric_stats' in stats
        assert 'categorical_stats' in stats
        
        # Check numeric stats
        assert 'feature1' in stats['numeric_stats']
        assert 'feature2' in stats['numeric_stats']
        assert 'mean' in stats['numeric_stats']['feature1']
        assert 'std' in stats['numeric_stats']['feature1']
        
        # Check categorical stats
        assert 'feature3' in stats['categorical_stats']
        assert 'unique_values' in stats['categorical_stats']['feature3']
    
    def test_get_baseline_statistics_no_baseline(self):
        """Test getting baseline statistics without baseline data."""
        detector = DriftDetector()
        stats = detector.get_baseline_statistics()
        
        assert stats == {}


class TestDriftMonitor:
    """Test cases for DriftMonitor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100)
        })
    
    @pytest.fixture
    def mock_drift_detector(self):
        """Create mock drift detector."""
        detector = Mock(spec=DriftDetector)
        detector.detect_data_drift.return_value = {
            'drift_detected': False,
            'drift_score': 0.05,
            'drift_type': 'data'
        }
        detector.detect_feature_drift.return_value = {
            'drift_detected': False,
            'drift_score': 0.03,
            'drift_type': 'feature'
        }
        detector.run_drift_test_suite.return_value = {
            'total_tests': 3,
            'tests_passed': 3,
            'tests_failed': 0,
            'success_rate': 1.0
        }
        return detector
    
    def test_drift_monitor_initialization(self):
        """Test DriftMonitor initialization."""
        monitor = DriftMonitor()
        
        assert monitor.detectors == {}
        assert monitor.monitoring_config == {}
        assert monitor.monitoring_history == []
    
    def test_add_detector(self, mock_drift_detector):
        """Test adding drift detector to monitor."""
        monitor = DriftMonitor()
        monitor.add_detector('test_detector', mock_drift_detector)
        
        assert 'test_detector' in monitor.detectors
        assert monitor.detectors['test_detector'] == mock_drift_detector
    
    def test_monitor_drift_single_detector(self, mock_drift_detector, sample_data):
        """Test drift monitoring with single detector."""
        monitor = DriftMonitor()
        monitor.add_detector('test_detector', mock_drift_detector)
        
        results = monitor.monitor_drift(sample_data)
        
        assert 'timestamp' in results
        assert 'detector_results' in results
        assert 'overall_drift_detected' in results
        assert 'summary' in results
        assert 'test_detector' in results['detector_results']
        
        # Check that detector methods were called
        mock_drift_detector.detect_data_drift.assert_called_once()
        mock_drift_detector.detect_feature_drift.assert_called_once()
        mock_drift_detector.run_drift_test_suite.assert_called_once()
    
    def test_monitor_drift_no_detectors(self, sample_data):
        """Test drift monitoring with no detectors."""
        monitor = DriftMonitor()
        
        results = monitor.monitor_drift(sample_data)
        
        assert results['detector_results'] == {}
        assert results['overall_drift_detected'] is False
    
    def test_monitor_drift_specific_detector(self, mock_drift_detector, sample_data):
        """Test drift monitoring with specific detector."""
        monitor = DriftMonitor()
        monitor.add_detector('detector1', mock_drift_detector)
        monitor.add_detector('detector2', Mock(spec=DriftDetector))
        
        results = monitor.monitor_drift(sample_data, detector_name='detector1')
        
        assert 'detector1' in results['detector_results']
        assert 'detector2' not in results['detector_results']
    
    def test_monitor_drift_with_drift_detected(self, sample_data):
        """Test drift monitoring when drift is detected."""
        # Create detector that reports drift
        detector_with_drift = Mock(spec=DriftDetector)
        detector_with_drift.detect_data_drift.return_value = {
            'drift_detected': True,
            'drift_score': 0.15,
            'drift_type': 'data'
        }
        detector_with_drift.detect_feature_drift.return_value = {
            'drift_detected': False,
            'drift_score': 0.03,
            'drift_type': 'feature'
        }
        detector_with_drift.run_drift_test_suite.return_value = {
            'total_tests': 3,
            'tests_passed': 2,
            'tests_failed': 1,
            'success_rate': 0.67
        }
        
        monitor = DriftMonitor()
        monitor.add_detector('drift_detector', detector_with_drift)
        
        results = monitor.monitor_drift(sample_data)
        
        assert results['overall_drift_detected'] is True
        assert 'data_drift' in results['summary']['drift_types_detected']
        assert results['summary']['detectors_with_drift'] == 1
        assert len(results['summary']['recommendations']) > 0
    
    def test_monitor_drift_detector_error(self, sample_data):
        """Test drift monitoring when detector raises error."""
        # Create detector that raises error
        error_detector = Mock(spec=DriftDetector)
        error_detector.detect_data_drift.side_effect = Exception("Test error")
        
        monitor = DriftMonitor()
        monitor.add_detector('error_detector', error_detector)
        
        results = monitor.monitor_drift(sample_data)
        
        assert 'error_detector' in results['detector_results']
        assert 'error' in results['detector_results']['error_detector']
        assert results['detector_results']['error_detector']['error'] == "Test error"
    
    def test_get_monitoring_history(self, mock_drift_detector, sample_data):
        """Test getting monitoring history."""
        monitor = DriftMonitor()
        monitor.add_detector('test_detector', mock_drift_detector)
        
        # Run monitoring multiple times
        monitor.monitor_drift(sample_data)
        monitor.monitor_drift(sample_data)
        monitor.monitor_drift(sample_data)
        
        # Get full history
        history = monitor.get_monitoring_history()
        assert len(history) == 3
        
        # Get limited history
        limited_history = monitor.get_monitoring_history(limit=2)
        assert len(limited_history) == 2
    
    def test_generate_monitoring_summary_no_drift(self):
        """Test monitoring summary generation with no drift."""
        monitor = DriftMonitor()
        detector_results = {
            'detector1': {
                'data_drift': {'drift_detected': False},
                'feature_drift': {'drift_detected': False}
            }
        }
        
        summary = monitor._generate_monitoring_summary(detector_results)
        
        assert summary['total_detectors'] == 1
        assert summary['detectors_with_drift'] == 0
        assert summary['drift_types_detected'] == []
        assert len(summary['recommendations']) == 0
    
    def test_generate_monitoring_summary_with_drift(self):
        """Test monitoring summary generation with drift."""
        monitor = DriftMonitor()
        detector_results = {
            'detector1': {
                'data_drift': {'drift_detected': True},
                'feature_drift': {'drift_detected': False}
            },
            'detector2': {
                'data_drift': {'drift_detected': False},
                'feature_drift': {'drift_detected': True}
            }
        }
        
        summary = monitor._generate_monitoring_summary(detector_results)
        
        assert summary['total_detectors'] == 2
        assert summary['detectors_with_drift'] == 2
        assert 'data_drift' in summary['drift_types_detected']
        assert 'feature_drift' in summary['drift_types_detected']
        assert len(summary['recommendations']) > 0


if __name__ == '__main__':
    pytest.main([__file__])