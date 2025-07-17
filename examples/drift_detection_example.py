"""
Example demonstrating drift detection capabilities using Evidently AI.

This example shows how to:
1. Set up drift detection with baseline data
2. Detect different types of drift (data, feature, prediction)
3. Run comprehensive drift monitoring
4. Generate drift reports and visualizations
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

from mlpipeline.monitoring.drift_detection import DriftDetector, DriftMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample datasets for drift detection demonstration."""
    np.random.seed(42)
    
    # Baseline data (training data)
    baseline_data = pd.DataFrame({
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'credit_score': np.random.normal(650, 100, 1000),
        'employment_type': np.random.choice(['full_time', 'part_time', 'self_employed'], 1000, p=[0.7, 0.2, 0.1]),
        'region': np.random.choice(['north', 'south', 'east', 'west'], 1000, p=[0.3, 0.3, 0.2, 0.2]),
        'default_risk': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })
    
    # Current data with no drift (similar distribution)
    np.random.seed(43)
    current_data_no_drift = pd.DataFrame({
        'age': np.random.normal(35, 10, 500),
        'income': np.random.normal(50000, 15000, 500),
        'credit_score': np.random.normal(650, 100, 500),
        'employment_type': np.random.choice(['full_time', 'part_time', 'self_employed'], 500, p=[0.7, 0.2, 0.1]),
        'region': np.random.choice(['north', 'south', 'east', 'west'], 500, p=[0.3, 0.3, 0.2, 0.2]),
        'default_risk': np.random.choice([0, 1], 500, p=[0.8, 0.2])
    })
    
    # Current data with drift (shifted distributions)
    np.random.seed(44)
    current_data_with_drift = pd.DataFrame({
        'age': np.random.normal(40, 12, 500),  # Age shifted higher with more variance
        'income': np.random.normal(55000, 20000, 500),  # Income increased with more variance
        'credit_score': np.random.normal(620, 120, 500),  # Credit score decreased with more variance
        'employment_type': np.random.choice(['full_time', 'part_time', 'self_employed', 'unemployed'], 500, p=[0.5, 0.2, 0.1, 0.2]),  # New category
        'region': np.random.choice(['north', 'south', 'east', 'west'], 500, p=[0.1, 0.1, 0.4, 0.4]),  # Distribution shifted
        'default_risk': np.random.choice([0, 1], 500, p=[0.6, 0.4])  # Higher default rate
    })
    
    return baseline_data, current_data_no_drift, current_data_with_drift


def demonstrate_basic_drift_detection():
    """Demonstrate basic drift detection functionality."""
    logger.info("=== Basic Drift Detection Demo ===")
    
    # Create sample data
    baseline_data, current_no_drift, current_with_drift = create_sample_data()
    
    # Initialize drift detector
    detector = DriftDetector(
        baseline_data=baseline_data,
        drift_thresholds={
            'data_drift': 0.1,
            'prediction_drift': 0.05,
            'feature_drift': 0.1
        },
        output_dir="drift_reports"
    )
    
    logger.info(f"Baseline data shape: {baseline_data.shape}")
    logger.info(f"Baseline statistics: {detector.get_baseline_statistics()}")
    
    # Test with data that has no drift
    logger.info("\n--- Testing data with NO drift ---")
    try:
        no_drift_results = detector.detect_data_drift(
            current_no_drift, 
            save_report=True,
            report_name="no_drift_example"
        )
        logger.info(f"No drift results: {no_drift_results}")
    except Exception as e:
        logger.error(f"Error in no-drift detection: {e}")
    
    # Test with data that has drift
    logger.info("\n--- Testing data WITH drift ---")
    try:
        drift_results = detector.detect_data_drift(
            current_with_drift, 
            save_report=True,
            report_name="with_drift_example"
        )
        logger.info(f"Drift results: {drift_results}")
    except Exception as e:
        logger.error(f"Error in drift detection: {e}")


def demonstrate_feature_drift_detection():
    """Demonstrate feature-level drift detection."""
    logger.info("\n=== Feature-Level Drift Detection Demo ===")
    
    # Create sample data
    baseline_data, _, current_with_drift = create_sample_data()
    
    # Initialize drift detector
    detector = DriftDetector(baseline_data=baseline_data, output_dir="drift_reports")
    
    # Detect feature-level drift
    try:
        feature_drift_results = detector.detect_feature_drift(
            current_with_drift,
            feature_columns=['age', 'income', 'credit_score', 'employment_type'],
            save_report=True,
            report_name="feature_drift_example"
        )
        
        logger.info(f"Feature drift results: {feature_drift_results}")
        
        # Print individual feature results
        for feature, result in feature_drift_results.get('feature_results', {}).items():
            logger.info(f"Feature '{feature}': drift_detected={result['drift_detected']}, "
                       f"drift_score={result['drift_score']:.4f}")
    
    except Exception as e:
        logger.error(f"Error in feature drift detection: {e}")


def demonstrate_prediction_drift_detection():
    """Demonstrate prediction drift detection."""
    logger.info("\n=== Prediction Drift Detection Demo ===")
    
    # Create sample predictions
    np.random.seed(42)
    baseline_predictions = np.random.beta(2, 5, 1000)  # Baseline predictions
    
    np.random.seed(45)
    current_predictions_no_drift = np.random.beta(2, 5, 500)  # Similar distribution
    current_predictions_with_drift = np.random.beta(5, 2, 500)  # Different distribution
    
    # Initialize drift detector
    detector = DriftDetector(output_dir="drift_reports")
    
    # Test prediction drift with no drift
    logger.info("\n--- Testing predictions with NO drift ---")
    try:
        no_pred_drift_results = detector.detect_prediction_drift(
            baseline_predictions,
            current_predictions_no_drift,
            save_report=True,
            report_name="prediction_no_drift_example"
        )
        logger.info(f"No prediction drift results: {no_pred_drift_results}")
    except Exception as e:
        logger.error(f"Error in no prediction drift detection: {e}")
    
    # Test prediction drift with drift
    logger.info("\n--- Testing predictions WITH drift ---")
    try:
        pred_drift_results = detector.detect_prediction_drift(
            baseline_predictions,
            current_predictions_with_drift,
            save_report=True,
            report_name="prediction_with_drift_example"
        )
        logger.info(f"Prediction drift results: {pred_drift_results}")
    except Exception as e:
        logger.error(f"Error in prediction drift detection: {e}")


def demonstrate_drift_test_suite():
    """Demonstrate drift test suite functionality."""
    logger.info("\n=== Drift Test Suite Demo ===")
    
    # Create sample data
    baseline_data, current_no_drift, current_with_drift = create_sample_data()
    
    # Initialize drift detector
    detector = DriftDetector(baseline_data=baseline_data, output_dir="drift_reports")
    
    # Run test suite on data with no drift
    logger.info("\n--- Running test suite on data with NO drift ---")
    try:
        no_drift_test_results = detector.run_drift_test_suite(
            current_no_drift,
            save_report=True,
            report_name="test_suite_no_drift"
        )
        logger.info(f"Test suite results (no drift): {no_drift_test_results}")
    except Exception as e:
        logger.error(f"Error in test suite (no drift): {e}")
    
    # Run test suite on data with drift
    logger.info("\n--- Running test suite on data WITH drift ---")
    try:
        drift_test_results = detector.run_drift_test_suite(
            current_with_drift,
            save_report=True,
            report_name="test_suite_with_drift"
        )
        logger.info(f"Test suite results (with drift): {drift_test_results}")
    except Exception as e:
        logger.error(f"Error in test suite (with drift): {e}")


def demonstrate_drift_monitor():
    """Demonstrate comprehensive drift monitoring."""
    logger.info("\n=== Comprehensive Drift Monitoring Demo ===")
    
    # Create sample data
    baseline_data, current_no_drift, current_with_drift = create_sample_data()
    
    # Create multiple drift detectors
    detector1 = DriftDetector(
        baseline_data=baseline_data,
        drift_thresholds={'data_drift': 0.1, 'feature_drift': 0.1},
        output_dir="drift_reports/detector1"
    )
    
    detector2 = DriftDetector(
        baseline_data=baseline_data,
        drift_thresholds={'data_drift': 0.05, 'feature_drift': 0.05},  # More sensitive
        output_dir="drift_reports/detector2"
    )
    
    # Initialize drift monitor
    monitor = DriftMonitor()
    monitor.add_detector('standard_detector', detector1)
    monitor.add_detector('sensitive_detector', detector2)
    
    # Monitor data with no drift
    logger.info("\n--- Monitoring data with NO drift ---")
    try:
        no_drift_monitoring = monitor.monitor_drift(current_no_drift)
        logger.info(f"Monitoring results (no drift): {no_drift_monitoring['summary']}")
    except Exception as e:
        logger.error(f"Error in monitoring (no drift): {e}")
    
    # Monitor data with drift
    logger.info("\n--- Monitoring data WITH drift ---")
    try:
        drift_monitoring = monitor.monitor_drift(current_with_drift)
        logger.info(f"Monitoring results (with drift): {drift_monitoring['summary']}")
        
        # Print recommendations
        recommendations = drift_monitoring['summary'].get('recommendations', [])
        if recommendations:
            logger.info("Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
    except Exception as e:
        logger.error(f"Error in monitoring (with drift): {e}")
    
    # Get monitoring history
    history = monitor.get_monitoring_history()
    logger.info(f"\nMonitoring history: {len(history)} entries")


def demonstrate_baseline_statistics():
    """Demonstrate baseline statistics functionality."""
    logger.info("\n=== Baseline Statistics Demo ===")
    
    # Create sample data
    baseline_data, _, _ = create_sample_data()
    
    # Initialize drift detector
    detector = DriftDetector(baseline_data=baseline_data)
    
    # Get baseline statistics
    stats = detector.get_baseline_statistics()
    
    logger.info(f"Baseline data shape: {stats['shape']}")
    logger.info(f"Columns: {stats['columns']}")
    
    # Print numeric statistics
    logger.info("\nNumeric column statistics:")
    for col, col_stats in stats['numeric_stats'].items():
        logger.info(f"  {col}: mean={col_stats['mean']:.2f}, std={col_stats['std']:.2f}, "
                   f"min={col_stats['min']:.2f}, max={col_stats['max']:.2f}")
    
    # Print categorical statistics
    logger.info("\nCategorical column statistics:")
    for col, col_stats in stats['categorical_stats'].items():
        logger.info(f"  {col}: unique_values={col_stats['unique_values']}, "
                   f"most_frequent='{col_stats['most_frequent']}'")


def main():
    """Run all drift detection demonstrations."""
    logger.info("Starting Drift Detection Examples")
    
    # Create output directory
    Path("drift_reports").mkdir(exist_ok=True)
    
    try:
        # Run demonstrations
        demonstrate_baseline_statistics()
        demonstrate_basic_drift_detection()
        demonstrate_feature_drift_detection()
        demonstrate_prediction_drift_detection()
        demonstrate_drift_test_suite()
        demonstrate_drift_monitor()
        
        logger.info("\n=== All demonstrations completed successfully! ===")
        logger.info("Check the 'drift_reports' directory for generated HTML and JSON reports.")
        
    except Exception as e:
        logger.error(f"Error in demonstrations: {e}")
        raise


if __name__ == "__main__":
    main()