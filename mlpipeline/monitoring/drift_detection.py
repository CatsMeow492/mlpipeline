"""
Drift detection module using Evidently AI for data and prediction drift monitoring.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from evidently import ColumnType, DataDefinition
from evidently import Report
try:
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import (
        ColumnDriftMetric,
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
    )
    EVIDENTLY_LEGACY = True
except ImportError:
    # Newer evidently API
    EVIDENTLY_LEGACY = False

# TestSuite functionality is not available in newer evidently versions
# We'll use Report-based approach instead

from ..core.interfaces import PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
from ..core.errors import PipelineError


logger = logging.getLogger(__name__)


class DriftDetector(PipelineComponent):
    """
    Drift detection component using Evidently AI for comprehensive drift analysis.
    
    Supports multiple drift detection methods:
    - KL divergence
    - Population Stability Index (PSI)
    - Wasserstein distance
    - Jensen-Shannon divergence
    """
    
    def __init__(
        self,
        baseline_data: Optional[pd.DataFrame] = None,
        baseline_path: Optional[str] = None,
        drift_thresholds: Optional[Dict[str, float]] = None,
        column_mapping: Optional[Dict[str, Any]] = None,
        output_dir: str = "drift_reports",
    ):
        """
        Initialize drift detector.
        
        Args:
            baseline_data: Reference dataset for drift comparison
            baseline_path: Path to baseline data file
            drift_thresholds: Custom thresholds for drift detection
            column_mapping: Evidently column mapping configuration
            output_dir: Directory to save drift reports
        """
        from ..core.interfaces import ComponentType
        super().__init__(ComponentType.DRIFT_DETECTION)
        self.baseline_data = baseline_data
        self.baseline_path = baseline_path
        self.drift_thresholds = drift_thresholds or {
            "data_drift": 0.1,
            "prediction_drift": 0.05,
            "feature_drift": 0.1,
        }
        self.column_mapping = column_mapping
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load baseline data if path provided
        if baseline_path and not baseline_data:
            self._load_baseline_data()
    

    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate drift detection configuration."""
        required_fields = ['baseline_data', 'drift_thresholds']
        for field in required_fields:
            if field not in config:
                return False
        return True
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute drift detection on current data."""
        try:
            current_data = context.metadata.get('current_data')
            if current_data is None:
                return ExecutionResult(
                    success=False,
                    error_message="No current data provided for drift detection"
                )
            
            # Perform drift detection
            drift_results = self.detect_data_drift(current_data)
            
            return ExecutionResult(
                success=True,
                metrics=drift_results,
                metadata={'drift_detected': drift_results.get('drift_detected', False)}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=f"Drift detection failed: {str(e)}"
            )
    
    def _load_baseline_data(self) -> None:
        """Load baseline data from file."""
        try:
            baseline_path = Path(self.baseline_path)
            if baseline_path.suffix == '.csv':
                self.baseline_data = pd.read_csv(baseline_path)
            elif baseline_path.suffix == '.parquet':
                self.baseline_data = pd.read_parquet(baseline_path)
            elif baseline_path.suffix == '.json':
                self.baseline_data = pd.read_json(baseline_path)
            else:
                raise ValueError(f"Unsupported file format: {baseline_path.suffix}")
            
            logger.info(f"Loaded baseline data with shape {self.baseline_data.shape}")
            
        except Exception as e:
            raise PipelineError(
                f"Failed to load baseline data from {self.baseline_path}",
                "BASELINE_LOAD_ERROR",
                {"path": self.baseline_path, "error": str(e)}
            )
    
    def set_baseline(self, data: pd.DataFrame) -> None:
        """Set baseline data for drift comparison."""
        self.baseline_data = data.copy()
        logger.info(f"Set baseline data with shape {data.shape}")
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True,
        report_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect data drift between baseline and current data.
        
        Args:
            current_data: Current dataset to compare against baseline
            save_report: Whether to save detailed drift report
            report_name: Custom name for the report file
            
        Returns:
            Dictionary containing drift detection results
        """
        if self.baseline_data is None:
            raise PipelineError(
                "No baseline data available for drift detection",
                "NO_BASELINE_ERROR",
                {}
            )
        
        try:
            # Create data drift report
            data_drift_report = Report(metrics=[
                DataDriftPreset(),
                DatasetDriftMetric(),
                DatasetMissingValuesMetric(),
            ])
            
            # Run the report
            data_drift_report.run(
                reference_data=self.baseline_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract results
            report_dict = data_drift_report.as_dict()
            results = self._extract_drift_results(report_dict, "data")
            
            # Save report if requested
            if save_report:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = report_name or f"data_drift_report_{timestamp}"
                self._save_report(data_drift_report, report_name)
            
            logger.info(f"Data drift detection completed. Drift detected: {results['drift_detected']}")
            return results
            
        except Exception as e:
            raise PipelineError(
                f"Data drift detection failed: {str(e)}",
                "DRIFT_DETECTION_ERROR",
                {"error": str(e)}
            )
    
    def detect_prediction_drift(
        self,
        baseline_predictions: Union[pd.Series, np.ndarray],
        current_predictions: Union[pd.Series, np.ndarray],
        save_report: bool = True,
        report_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect prediction drift between baseline and current predictions.
        
        Args:
            baseline_predictions: Reference predictions
            current_predictions: Current predictions to compare
            save_report: Whether to save detailed drift report
            report_name: Custom name for the report file
            
        Returns:
            Dictionary containing prediction drift results
        """
        try:
            # Convert to DataFrames for Evidently
            baseline_df = pd.DataFrame({'prediction': baseline_predictions})
            current_df = pd.DataFrame({'prediction': current_predictions})
            
            # Create target drift report
            target_drift_report = Report(metrics=[
                TargetDriftPreset(),
                ColumnDriftMetric(column_name='prediction'),
            ])
            
            # Set column mapping for predictions
            column_mapping = {'target': 'prediction'}
            
            # Run the report
            target_drift_report.run(
                reference_data=baseline_df,
                current_data=current_df,
                column_mapping=column_mapping
            )
            
            # Extract results
            report_dict = target_drift_report.as_dict()
            results = self._extract_drift_results(report_dict, "prediction")
            
            # Save report if requested
            if save_report:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = report_name or f"prediction_drift_report_{timestamp}"
                self._save_report(target_drift_report, report_name)
            
            logger.info(f"Prediction drift detection completed. Drift detected: {results['drift_detected']}")
            return results
            
        except Exception as e:
            raise PipelineError(
                f"Prediction drift detection failed: {str(e)}",
                "PREDICTION_DRIFT_ERROR",
                {"error": str(e)}
            )
    
    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        save_report: bool = True,
        report_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect drift for individual features.
        
        Args:
            current_data: Current dataset to analyze
            feature_columns: Specific columns to analyze (all if None)
            save_report: Whether to save detailed drift report
            report_name: Custom name for the report file
            
        Returns:
            Dictionary containing feature-level drift results
        """
        if self.baseline_data is None:
            raise PipelineError(
                "No baseline data available for feature drift detection",
                "NO_BASELINE_ERROR",
                {}
            )
        
        try:
            # Determine columns to analyze
            if feature_columns is None:
                feature_columns = list(current_data.columns)
            
            # Create feature-specific drift metrics
            feature_metrics = [ColumnDriftMetric(column_name=col) for col in feature_columns]
            feature_drift_report = Report(metrics=feature_metrics)
            
            # Run the report
            feature_drift_report.run(
                reference_data=self.baseline_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract feature-level results
            report_dict = feature_drift_report.as_dict()
            results = self._extract_feature_drift_results(report_dict, feature_columns)
            
            # Save report if requested
            if save_report:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = report_name or f"feature_drift_report_{timestamp}"
                self._save_report(feature_drift_report, report_name)
            
            logger.info(f"Feature drift detection completed for {len(feature_columns)} features")
            return results
            
        except Exception as e:
            raise PipelineError(
                f"Feature drift detection failed: {str(e)}",
                "FEATURE_DRIFT_ERROR",
                {"error": str(e)}
            )
    
    def run_drift_test_suite(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True,
        report_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive drift test suite with pass/fail results.
        
        Note: TestSuite functionality is not available in newer evidently versions.
        This method returns a simplified test result based on drift detection.
        
        Args:
            current_data: Current dataset to test
            save_report: Whether to save test results
            report_name: Custom name for the test report
            
        Returns:
            Dictionary containing test results
        """
        if self.baseline_data is None:
            raise PipelineError(
                "No baseline data available for drift testing",
                "NO_BASELINE_ERROR",
                {}
            )
        
        try:
            # Since TestSuite is not available, we'll use drift detection results
            # to simulate test results
            data_drift_results = self.detect_data_drift(current_data, save_report=False)
            feature_drift_results = self.detect_feature_drift(current_data, save_report=False)
            
            # Create simplified test results
            tests = []
            
            # Data drift test
            tests.append({
                'name': 'DataDriftTest',
                'status': 'FAIL' if data_drift_results['drift_detected'] else 'SUCCESS',
                'description': f"Data drift detection with threshold {data_drift_results.get('threshold', 0.1)}",
                'parameters': {'drift_score': data_drift_results.get('drift_score', 0.0)},
            })
            
            # Feature drift test
            tests.append({
                'name': 'FeatureDriftTest',
                'status': 'FAIL' if feature_drift_results['drift_detected'] else 'SUCCESS',
                'description': f"Feature drift detection with threshold {feature_drift_results.get('threshold', 0.1)}",
                'parameters': {'drift_score': feature_drift_results.get('drift_score', 0.0)},
            })
            
            # Calculate summary
            total_tests = len(tests)
            passed_tests = sum(1 for test in tests if test['status'] == 'SUCCESS')
            
            results = {
                'total_tests': total_tests,
                'tests_passed': passed_tests,
                'tests_failed': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                'timestamp': datetime.now().isoformat(),
                'test_details': tests
            }
            
            # Save simplified report if requested
            if save_report:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = report_name or f"drift_test_suite_{timestamp}"
                self._save_simplified_test_report(results, report_name)
            
            logger.info(f"Drift test suite completed. Tests passed: {results['tests_passed']}/{results['total_tests']}")
            return results
            
        except Exception as e:
            raise PipelineError(
                f"Drift test suite failed: {str(e)}",
                "DRIFT_TEST_ERROR",
                {"error": str(e)}
            ) 
   
    def _extract_drift_results(self, report_dict: Dict[str, Any], drift_type: str) -> Dict[str, Any]:
        """Extract drift results from Evidently report."""
        try:
            metrics = report_dict.get('metrics', [])
            
            # Find dataset drift metric
            dataset_drift_metric = None
            for metric in metrics:
                if metric.get('metric') == 'DatasetDriftMetric':
                    dataset_drift_metric = metric
                    break
            
            if not dataset_drift_metric:
                return {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'drift_type': drift_type,
                    'timestamp': datetime.now().isoformat(),
                    'details': 'No drift metric found in report'
                }
            
            result = dataset_drift_metric.get('result', {})
            drift_share = result.get('drift_share', 0.0)
            number_of_drifted_columns = result.get('number_of_drifted_columns', 0)
            
            # Determine if drift detected based on threshold
            threshold = self.drift_thresholds.get(f'{drift_type}_drift', 0.1)
            drift_detected = drift_share > threshold
            
            return {
                'drift_detected': drift_detected,
                'drift_score': drift_share,
                'drift_type': drift_type,
                'drifted_columns': number_of_drifted_columns,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat(),
                'details': result
            }
            
        except Exception as e:
            logger.error(f"Error extracting drift results: {e}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'drift_type': drift_type,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _extract_feature_drift_results(
        self, 
        report_dict: Dict[str, Any], 
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Extract feature-level drift results."""
        try:
            metrics = report_dict.get('metrics', [])
            feature_results = {}
            
            for metric in metrics:
                if metric.get('metric') == 'ColumnDriftMetric':
                    column_name = metric.get('result', {}).get('column_name')
                    if column_name in feature_columns:
                        result = metric.get('result', {})
                        drift_score = result.get('drift_score', 0.0)
                        drift_detected = result.get('drift_detected', False)
                        
                        feature_results[column_name] = {
                            'drift_detected': drift_detected,
                            'drift_score': drift_score,
                            'stattest_name': result.get('stattest_name', 'unknown'),
                            'threshold': result.get('threshold', 0.1),
                            'p_value': result.get('p_value'),
                        }
            
            # Calculate overall feature drift
            total_features = len(feature_columns)
            drifted_features = sum(1 for r in feature_results.values() if r['drift_detected'])
            overall_drift_score = drifted_features / total_features if total_features > 0 else 0.0
            
            threshold = self.drift_thresholds.get('feature_drift', 0.1)
            overall_drift_detected = overall_drift_score > threshold
            
            return {
                'drift_detected': overall_drift_detected,
                'drift_score': overall_drift_score,
                'drift_type': 'feature',
                'total_features': total_features,
                'drifted_features': drifted_features,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat(),
                'feature_results': feature_results
            }
            
        except Exception as e:
            logger.error(f"Error extracting feature drift results: {e}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'drift_type': 'feature',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _extract_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test suite results."""
        try:
            tests = test_results.get('tests', [])
            total_tests = len(tests)
            passed_tests = sum(1 for test in tests if test.get('status') == 'SUCCESS')
            failed_tests = total_tests - passed_tests
            
            test_details = []
            for test in tests:
                test_details.append({
                    'name': test.get('name', 'Unknown'),
                    'status': test.get('status', 'UNKNOWN'),
                    'description': test.get('description', ''),
                    'parameters': test.get('parameters', {}),
                })
            
            return {
                'total_tests': total_tests,
                'tests_passed': passed_tests,
                'tests_failed': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                'timestamp': datetime.now().isoformat(),
                'test_details': test_details
            }
            
        except Exception as e:
            logger.error(f"Error extracting test results: {e}")
            return {
                'total_tests': 0,
                'tests_passed': 0,
                'tests_failed': 0,
                'success_rate': 0.0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _save_report(self, report: Report, report_name: str) -> None:
        """Save Evidently report to file."""
        try:
            # Save as HTML
            html_path = self.output_dir / f"{report_name}.html"
            report.save_html(str(html_path))
            
            # Save as JSON
            json_path = self.output_dir / f"{report_name}.json"
            report.save_json(str(json_path))
            
            logger.info(f"Saved drift report to {html_path} and {json_path}")
            
        except Exception as e:
            logger.error(f"Failed to save drift report: {e}")
    
    def _save_simplified_test_report(self, results: Dict[str, Any], report_name: str) -> None:
        """Save simplified test report to file."""
        try:
            # Save as JSON
            json_path = self.output_dir / f"{report_name}.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved simplified test report to {json_path}")
            
        except Exception as e:
            logger.error(f"Failed to save simplified test report: {e}")
    
    def get_baseline_statistics(self) -> Dict[str, Any]:
        """Get baseline data statistics for reference."""
        if self.baseline_data is None:
            return {}
        
        try:
            stats = {
                'shape': self.baseline_data.shape,
                'columns': list(self.baseline_data.columns),
                'dtypes': self.baseline_data.dtypes.to_dict(),
                'missing_values': self.baseline_data.isnull().sum().to_dict(),
                'numeric_stats': {},
                'categorical_stats': {}
            }
            
            # Numeric column statistics
            numeric_cols = self.baseline_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                stats['numeric_stats'][col] = {
                    'mean': float(self.baseline_data[col].mean()),
                    'std': float(self.baseline_data[col].std()),
                    'min': float(self.baseline_data[col].min()),
                    'max': float(self.baseline_data[col].max()),
                    'median': float(self.baseline_data[col].median()),
                }
            
            # Categorical column statistics
            categorical_cols = self.baseline_data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                value_counts = self.baseline_data[col].value_counts()
                stats['categorical_stats'][col] = {
                    'unique_values': int(self.baseline_data[col].nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing baseline statistics: {e}")
            return {'error': str(e)}


class DriftMonitor:
    """
    High-level drift monitoring orchestrator that manages multiple drift detectors
    and provides unified drift monitoring capabilities.
    """
    
    def __init__(
        self,
        detectors: Optional[Dict[str, DriftDetector]] = None,
        monitoring_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize drift monitor.
        
        Args:
            detectors: Dictionary of named drift detectors
            monitoring_config: Configuration for monitoring behavior
        """
        self.detectors = detectors or {}
        self.monitoring_config = monitoring_config or {}
        self.monitoring_history = []
    
    def add_detector(self, name: str, detector: DriftDetector) -> None:
        """Add a drift detector to the monitor."""
        self.detectors[name] = detector
        logger.info(f"Added drift detector: {name}")
    
    def monitor_drift(
        self,
        current_data: pd.DataFrame,
        detector_name: Optional[str] = None,
        include_predictions: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive drift monitoring.
        
        Args:
            current_data: Current dataset to monitor
            detector_name: Specific detector to use (all if None)
            include_predictions: Predictions to monitor for drift
            
        Returns:
            Comprehensive drift monitoring results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'detector_results': {},
            'overall_drift_detected': False,
            'summary': {}
        }
        
        # Determine which detectors to use
        detectors_to_use = (
            {detector_name: self.detectors[detector_name]} 
            if detector_name and detector_name in self.detectors
            else self.detectors
        )
        
        if not detectors_to_use:
            logger.warning("No drift detectors available for monitoring")
            return results
        
        # Run drift detection for each detector
        for name, detector in detectors_to_use.items():
            try:
                detector_results = {}
                
                # Data drift detection
                data_drift = detector.detect_data_drift(current_data, save_report=True)
                detector_results['data_drift'] = data_drift
                
                # Feature drift detection
                feature_drift = detector.detect_feature_drift(current_data, save_report=True)
                detector_results['feature_drift'] = feature_drift
                
                # Prediction drift detection if predictions provided
                if include_predictions is not None:
                    # Need baseline predictions - this would typically come from stored baseline
                    # For now, we'll skip prediction drift if no baseline predictions available
                    logger.info("Prediction drift detection requires baseline predictions")
                
                # Test suite
                test_results = detector.run_drift_test_suite(current_data, save_report=True)
                detector_results['test_suite'] = test_results
                
                results['detector_results'][name] = detector_results
                
                # Update overall drift status
                if (data_drift.get('drift_detected', False) or 
                    feature_drift.get('drift_detected', False)):
                    results['overall_drift_detected'] = True
                
            except Exception as e:
                logger.error(f"Error running drift detection for {name}: {e}")
                results['detector_results'][name] = {'error': str(e)}
        
        # Generate summary
        results['summary'] = self._generate_monitoring_summary(results['detector_results'])
        
        # Store in monitoring history
        self.monitoring_history.append(results)
        
        return results
    
    def _generate_monitoring_summary(self, detector_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of monitoring results."""
        summary = {
            'total_detectors': len(detector_results),
            'detectors_with_drift': 0,
            'drift_types_detected': [],
            'recommendations': []
        }
        
        for detector_name, results in detector_results.items():
            if 'error' in results:
                continue
                
            detector_has_drift = False
            
            # Check data drift
            if results.get('data_drift', {}).get('drift_detected', False):
                detector_has_drift = True
                if 'data_drift' not in summary['drift_types_detected']:
                    summary['drift_types_detected'].append('data_drift')
            
            # Check feature drift
            if results.get('feature_drift', {}).get('drift_detected', False):
                detector_has_drift = True
                if 'feature_drift' not in summary['drift_types_detected']:
                    summary['drift_types_detected'].append('feature_drift')
            
            if detector_has_drift:
                summary['detectors_with_drift'] += 1
        
        # Generate recommendations
        if summary['detectors_with_drift'] > 0:
            summary['recommendations'].extend([
                "Investigate data quality and preprocessing steps",
                "Consider retraining models with recent data",
                "Review feature engineering and selection",
                "Monitor model performance metrics closely"
            ])
        
        return summary
    
    def get_monitoring_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get monitoring history."""
        if limit:
            return self.monitoring_history[-limit:]
        return self.monitoring_history