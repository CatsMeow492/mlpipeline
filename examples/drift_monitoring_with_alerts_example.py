"""
Comprehensive example combining drift detection with alerting system.

This example demonstrates:
1. Setting up drift detection with baseline data
2. Configuring alert rules for different drift scenarios
3. Integrating drift monitoring with alert management
4. Automated drift monitoring with alert generation
5. Creating comprehensive drift and alert reports
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from mlpipeline.monitoring.drift_detection import DriftDetector, DriftMonitor
from mlpipeline.monitoring.alerts import (
    AlertRule,
    AlertManager,
    EmailAlertChannel,
    SlackAlertChannel,
    WebhookAlertChannel,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_datasets():
    """Create sample datasets for drift monitoring demonstration."""
    np.random.seed(42)
    
    # Baseline data (training data)
    baseline_data = pd.DataFrame({
        'age': np.random.normal(35, 10, 2000),
        'income': np.random.normal(50000, 15000, 2000),
        'credit_score': np.random.normal(650, 100, 2000),
        'employment_years': np.random.normal(8, 5, 2000),
        'debt_ratio': np.random.beta(2, 5, 2000),
        'employment_type': np.random.choice(['full_time', 'part_time', 'self_employed'], 2000, p=[0.7, 0.2, 0.1]),
        'region': np.random.choice(['north', 'south', 'east', 'west'], 2000, p=[0.3, 0.3, 0.2, 0.2]),
        'default_risk': np.random.choice([0, 1], 2000, p=[0.8, 0.2])
    })
    
    # Current data with no drift (week 1)
    np.random.seed(43)
    current_data_week1 = pd.DataFrame({
        'age': np.random.normal(35, 10, 500),
        'income': np.random.normal(50000, 15000, 500),
        'credit_score': np.random.normal(650, 100, 500),
        'employment_years': np.random.normal(8, 5, 500),
        'debt_ratio': np.random.beta(2, 5, 500),
        'employment_type': np.random.choice(['full_time', 'part_time', 'self_employed'], 500, p=[0.7, 0.2, 0.1]),
        'region': np.random.choice(['north', 'south', 'east', 'west'], 500, p=[0.3, 0.3, 0.2, 0.2]),
        'default_risk': np.random.choice([0, 1], 500, p=[0.8, 0.2])
    })
    
    # Current data with moderate drift (week 2)
    np.random.seed(44)
    current_data_week2 = pd.DataFrame({
        'age': np.random.normal(37, 11, 500),  # Slight age increase
        'income': np.random.normal(52000, 16000, 500),  # Income increase
        'credit_score': np.random.normal(645, 105, 500),  # Slight score decrease
        'employment_years': np.random.normal(8, 5, 500),
        'debt_ratio': np.random.beta(2.2, 4.8, 500),  # Slightly higher debt
        'employment_type': np.random.choice(['full_time', 'part_time', 'self_employed'], 500, p=[0.65, 0.25, 0.1]),
        'region': np.random.choice(['north', 'south', 'east', 'west'], 500, p=[0.25, 0.35, 0.2, 0.2]),
        'default_risk': np.random.choice([0, 1], 500, p=[0.78, 0.22])
    })
    
    # Current data with significant drift (week 3)
    np.random.seed(45)
    current_data_week3 = pd.DataFrame({
        'age': np.random.normal(42, 15, 500),  # Significant age increase
        'income': np.random.normal(45000, 20000, 500),  # Income decrease with higher variance
        'credit_score': np.random.normal(620, 130, 500),  # Significant score decrease
        'employment_years': np.random.normal(6, 6, 500),  # Lower employment years
        'debt_ratio': np.random.beta(3, 4, 500),  # Much higher debt ratios
        'employment_type': np.random.choice(['full_time', 'part_time', 'self_employed', 'unemployed'], 500, p=[0.5, 0.2, 0.1, 0.2]),  # New category
        'region': np.random.choice(['north', 'south', 'east', 'west'], 500, p=[0.1, 0.1, 0.4, 0.4]),  # Distribution shift
        'default_risk': np.random.choice([0, 1], 500, p=[0.6, 0.4])  # Much higher default rate
    })
    
    return baseline_data, current_data_week1, current_data_week2, current_data_week3


def setup_drift_detection_system(baseline_data):
    """Set up comprehensive drift detection system."""
    logger.info("=== Setting up Drift Detection System ===")
    
    # Create drift detectors with different sensitivity levels
    standard_detector = DriftDetector(
        baseline_data=baseline_data,
        drift_thresholds={
            'data_drift': 0.1,
            'prediction_drift': 0.05,
            'feature_drift': 0.1
        },
        output_dir="drift_reports/standard"
    )
    
    sensitive_detector = DriftDetector(
        baseline_data=baseline_data,
        drift_thresholds={
            'data_drift': 0.05,
            'prediction_drift': 0.03,
            'feature_drift': 0.05
        },
        output_dir="drift_reports/sensitive"
    )
    
    # Create drift monitor
    drift_monitor = DriftMonitor()
    drift_monitor.add_detector('standard', standard_detector)
    drift_monitor.add_detector('sensitive', sensitive_detector)
    
    logger.info(f"Created drift monitor with {len(drift_monitor.detectors)} detectors")
    return drift_monitor


def setup_alert_system():
    """Set up comprehensive alert system for drift monitoring."""
    logger.info("=== Setting up Alert System ===")
    
    # Create alert rules for different drift scenarios
    alert_rules = [
        AlertRule(
            name="moderate_data_drift",
            condition="data_drift_score > 0.05",
            severity="medium",
            channels=["email"],
            threshold_value=0.05,
            comparison_operator=">",
            metric_name="data_drift_score",
            cooldown_minutes=60
        ),
        AlertRule(
            name="high_data_drift",
            condition="data_drift_score > 0.1",
            severity="high",
            channels=["email", "slack"],
            threshold_value=0.1,
            comparison_operator=">",
            metric_name="data_drift_score",
            cooldown_minutes=30
        ),
        AlertRule(
            name="critical_data_drift",
            condition="data_drift_score > 0.2",
            severity="critical",
            channels=["email", "slack", "webhook"],
            threshold_value=0.2,
            comparison_operator=">",
            metric_name="data_drift_score",
            cooldown_minutes=15
        ),
        AlertRule(
            name="feature_drift_alert",
            condition="feature_drift_score > 0.1",
            severity="medium",
            channels=["email"],
            threshold_value=0.1,
            comparison_operator=">",
            metric_name="feature_drift_score",
            cooldown_minutes=45
        ),
        AlertRule(
            name="multiple_drifted_features",
            condition="drifted_features > 3",
            severity="high",
            channels=["email", "slack"],
            threshold_value=3,
            comparison_operator=">",
            metric_name="drifted_features",
            cooldown_minutes=60
        ),
        AlertRule(
            name="test_suite_failure",
            condition="test_success_rate < 0.8",
            severity="medium",
            channels=["email"],
            threshold_value=0.8,
            comparison_operator="<",
            metric_name="test_success_rate",
            cooldown_minutes=90
        )
    ]
    
    # Create alert channels (using mock configurations for demo)
    email_channel = EmailAlertChannel('email', {
        'enabled': True,
        'smtp_server': 'smtp.example.com',
        'from_email': 'drift-alerts@company.com',
        'to_emails': ['ml-team@company.com', 'data-team@company.com']
    })
    
    slack_channel = SlackAlertChannel('slack', {
        'enabled': True,
        'webhook_url': 'https://hooks.slack.com/services/MOCK/WEBHOOK/URL',
        'channel': '#ml-drift-alerts',
        'username': 'Drift Monitor Bot'
    })
    
    webhook_channel = WebhookAlertChannel('webhook', {
        'enabled': True,
        'webhook_url': 'https://api.company.com/ml-alerts',
        'headers': {'Authorization': 'Bearer mock-token'}
    })
    
    # Create alert manager
    alert_manager = AlertManager(alert_history_file="drift_alert_history.json")
    
    # Add rules and channels
    for rule in alert_rules:
        alert_manager.add_rule(rule)
    
    alert_manager.add_channel(email_channel)
    alert_manager.add_channel(slack_channel)
    alert_manager.add_channel(webhook_channel)
    
    logger.info(f"Created alert manager with {len(alert_rules)} rules and {len(alert_manager.channels)} channels")
    return alert_manager


def run_drift_monitoring_with_alerts(drift_monitor, alert_manager, current_data, week_name):
    """Run drift monitoring and generate alerts based on results."""
    logger.info(f"\n=== Monitoring Drift for {week_name} ===")
    
    try:
        # Run drift monitoring
        monitoring_results = drift_monitor.monitor_drift(current_data)
        
        # Extract metrics for alert evaluation
        metrics = {}
        
        for detector_name, detector_results in monitoring_results['detector_results'].items():
            if 'error' in detector_results:
                logger.warning(f"Error in detector {detector_name}: {detector_results['error']}")
                continue
            
            # Extract data drift metrics
            data_drift = detector_results.get('data_drift', {})
            if data_drift:
                metrics[f'{detector_name}_data_drift_score'] = data_drift.get('drift_score', 0.0)
                metrics['data_drift_score'] = max(metrics.get('data_drift_score', 0.0), data_drift.get('drift_score', 0.0))
            
            # Extract feature drift metrics
            feature_drift = detector_results.get('feature_drift', {})
            if feature_drift:
                metrics[f'{detector_name}_feature_drift_score'] = feature_drift.get('drift_score', 0.0)
                metrics['feature_drift_score'] = max(metrics.get('feature_drift_score', 0.0), feature_drift.get('drift_score', 0.0))
                metrics['drifted_features'] = max(metrics.get('drifted_features', 0), feature_drift.get('drifted_features', 0))
            
            # Extract test suite metrics
            test_suite = detector_results.get('test_suite', {})
            if test_suite:
                metrics['test_success_rate'] = min(metrics.get('test_success_rate', 1.0), test_suite.get('success_rate', 1.0))
        
        # Log monitoring summary
        summary = monitoring_results.get('summary', {})
        logger.info(f"Overall drift detected: {monitoring_results.get('overall_drift_detected', False)}")
        logger.info(f"Detectors with drift: {summary.get('detectors_with_drift', 0)}/{summary.get('total_detectors', 0)}")
        logger.info(f"Drift types detected: {summary.get('drift_types_detected', [])}")
        
        # Evaluate metrics against alert rules
        logger.info(f"Evaluating {len(metrics)} metrics against alert rules...")
        alerts = alert_manager.evaluate_metrics(metrics, f"drift_monitor_{week_name}")
        
        if alerts:
            logger.info(f"Generated {len(alerts)} alerts:")
            for alert in alerts:
                logger.info(f"  - {alert.severity.upper()}: {alert.title}")
                logger.info(f"    Message: {alert.message}")
        else:
            logger.info("No alerts generated")
        
        return monitoring_results, alerts, metrics
        
    except Exception as e:
        logger.error(f"Error in drift monitoring for {week_name}: {e}")
        return None, [], {}


def generate_comprehensive_report(drift_monitor, alert_manager, all_results):
    """Generate comprehensive drift and alert report."""
    logger.info("\n=== Generating Comprehensive Report ===")
    
    # Collect all monitoring results
    report_data = {
        "report_timestamp": datetime.now().isoformat(),
        "monitoring_period": "3 weeks",
        "summary": {
            "total_monitoring_runs": len(all_results),
            "total_alerts_generated": sum(len(result['alerts']) for result in all_results),
            "weeks_with_drift": sum(1 for result in all_results if result['monitoring_results'] and result['monitoring_results'].get('overall_drift_detected', False)),
        },
        "weekly_results": [],
        "alert_statistics": alert_manager.get_alert_statistics(),
        "drift_trends": {},
        "recommendations": []
    }
    
    # Process weekly results
    for week_name, result in all_results.items():
        if result['monitoring_results']:
            weekly_data = {
                "week": week_name,
                "drift_detected": result['monitoring_results'].get('overall_drift_detected', False),
                "detectors_with_drift": result['monitoring_results']['summary'].get('detectors_with_drift', 0),
                "drift_types": result['monitoring_results']['summary'].get('drift_types_detected', []),
                "alerts_generated": len(result['alerts']),
                "alert_severities": {},
                "key_metrics": result['metrics']
            }
            
            # Count alert severities
            for alert in result['alerts']:
                severity = alert.severity
                weekly_data['alert_severities'][severity] = weekly_data['alert_severities'].get(severity, 0) + 1
            
            report_data['weekly_results'].append(weekly_data)
    
    # Analyze drift trends
    drift_scores = []
    feature_drift_scores = []
    drifted_features_counts = []
    
    for result in all_results.values():
        metrics = result['metrics']
        drift_scores.append(metrics.get('data_drift_score', 0.0))
        feature_drift_scores.append(metrics.get('feature_drift_score', 0.0))
        drifted_features_counts.append(metrics.get('drifted_features', 0))
    
    if drift_scores:
        report_data['drift_trends'] = {
            "data_drift_trend": {
                "values": drift_scores,
                "average": sum(drift_scores) / len(drift_scores),
                "max": max(drift_scores),
                "increasing": len(drift_scores) > 1 and drift_scores[-1] > drift_scores[0]
            },
            "feature_drift_trend": {
                "values": feature_drift_scores,
                "average": sum(feature_drift_scores) / len(feature_drift_scores),
                "max": max(feature_drift_scores),
                "increasing": len(feature_drift_scores) > 1 and feature_drift_scores[-1] > feature_drift_scores[0]
            },
            "drifted_features_trend": {
                "values": drifted_features_counts,
                "average": sum(drifted_features_counts) / len(drifted_features_counts),
                "max": max(drifted_features_counts),
                "increasing": len(drifted_features_counts) > 1 and drifted_features_counts[-1] > drifted_features_counts[0]
            }
        }
    
    # Generate recommendations
    recommendations = []
    
    if report_data['summary']['weeks_with_drift'] > 0:
        recommendations.append("Drift detected in multiple monitoring periods - investigate data quality and preprocessing")
    
    if report_data['drift_trends'].get('data_drift_trend', {}).get('increasing', False):
        recommendations.append("Data drift is increasing over time - consider model retraining")
    
    if report_data['drift_trends'].get('drifted_features_trend', {}).get('max', 0) > 3:
        recommendations.append("Multiple features showing drift - review feature engineering pipeline")
    
    if report_data['alert_statistics']['resolution_rate'] < 0.5:
        recommendations.append("Low alert resolution rate - review alert handling processes")
    
    if not recommendations:
        recommendations.append("Monitoring system is functioning well - continue regular monitoring")
    
    report_data['recommendations'] = recommendations
    
    # Save comprehensive report
    report_file = "comprehensive_drift_alert_report.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"Comprehensive report saved to {report_file}")
    
    # Print summary
    logger.info("\n--- Report Summary ---")
    logger.info(f"Monitoring runs: {report_data['summary']['total_monitoring_runs']}")
    logger.info(f"Total alerts: {report_data['summary']['total_alerts_generated']}")
    logger.info(f"Weeks with drift: {report_data['summary']['weeks_with_drift']}")
    logger.info(f"Alert resolution rate: {report_data['alert_statistics']['resolution_rate']:.2%}")
    
    logger.info("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")
    
    return report_data


def demonstrate_alert_lifecycle_management(alert_manager):
    """Demonstrate alert lifecycle management."""
    logger.info("\n=== Alert Lifecycle Management Demo ===")
    
    # Get active alerts
    active_alerts = alert_manager.get_active_alerts()
    logger.info(f"Current active alerts: {len(active_alerts)}")
    
    if active_alerts:
        # Demonstrate resolving alerts
        for i, alert in enumerate(active_alerts[:2]):  # Resolve first 2 alerts
            resolution_message = f"Resolved by retraining model and updating preprocessing pipeline"
            alert_manager.resolve_alert(alert.id, resolution_message)
            logger.info(f"Resolved alert: {alert.title}")
        
        # Demonstrate suppressing alerts
        if len(active_alerts) > 2:
            suppress_until = datetime.now() + timedelta(hours=4)
            alert_manager.suppress_alert(active_alerts[2].id, suppress_until)
            logger.info(f"Suppressed alert until {suppress_until}: {active_alerts[2].title}")
        
        # Demonstrate escalating alerts
        if len(active_alerts) > 3:
            alert_manager.escalate_alert(active_alerts[3].id, ["webhook"])
            logger.info(f"Escalated alert: {active_alerts[3].title}")
    
    # Show updated status
    updated_active = alert_manager.get_active_alerts()
    logger.info(f"Active alerts after management: {len(updated_active)}")
    
    # Show alert history with different filters
    recent_history = alert_manager.get_alert_history(limit=10)
    logger.info(f"Recent alert history: {len(recent_history)} alerts")
    
    high_severity = alert_manager.get_alert_history(severity="high")
    critical_severity = alert_manager.get_alert_history(severity="critical")
    logger.info(f"High severity alerts: {len(high_severity)}")
    logger.info(f"Critical severity alerts: {len(critical_severity)}")


def main():
    """Run comprehensive drift monitoring with alerts demonstration."""
    logger.info("Starting Comprehensive Drift Monitoring with Alerts")
    
    # Create output directories
    Path("drift_reports").mkdir(exist_ok=True)
    Path("drift_reports/standard").mkdir(exist_ok=True)
    Path("drift_reports/sensitive").mkdir(exist_ok=True)
    
    try:
        # Create sample datasets
        baseline_data, week1_data, week2_data, week3_data = create_sample_datasets()
        logger.info(f"Created datasets - Baseline: {baseline_data.shape}, Week1: {week1_data.shape}, Week2: {week2_data.shape}, Week3: {week3_data.shape}")
        
        # Set up systems
        drift_monitor = setup_drift_detection_system(baseline_data)
        alert_manager = setup_alert_system()
        
        # Run monitoring for each week
        all_results = {}
        
        # Week 1 - No drift expected
        week1_monitoring, week1_alerts, week1_metrics = run_drift_monitoring_with_alerts(
            drift_monitor, alert_manager, week1_data, "Week1"
        )
        all_results["Week1"] = {
            "monitoring_results": week1_monitoring,
            "alerts": week1_alerts,
            "metrics": week1_metrics
        }
        
        # Week 2 - Moderate drift expected
        week2_monitoring, week2_alerts, week2_metrics = run_drift_monitoring_with_alerts(
            drift_monitor, alert_manager, week2_data, "Week2"
        )
        all_results["Week2"] = {
            "monitoring_results": week2_monitoring,
            "alerts": week2_alerts,
            "metrics": week2_metrics
        }
        
        # Week 3 - Significant drift expected
        week3_monitoring, week3_alerts, week3_metrics = run_drift_monitoring_with_alerts(
            drift_monitor, alert_manager, week3_data, "Week3"
        )
        all_results["Week3"] = {
            "monitoring_results": week3_monitoring,
            "alerts": week3_alerts,
            "metrics": week3_metrics
        }
        
        # Demonstrate alert lifecycle management
        demonstrate_alert_lifecycle_management(alert_manager)
        
        # Generate comprehensive report
        report_data = generate_comprehensive_report(drift_monitor, alert_manager, all_results)
        
        logger.info("\n=== All demonstrations completed successfully! ===")
        logger.info("Generated files:")
        logger.info("  - drift_alert_history.json (alert history)")
        logger.info("  - comprehensive_drift_alert_report.json (full report)")
        logger.info("  - drift_reports/ (HTML and JSON drift reports)")
        
    except Exception as e:
        logger.error(f"Error in demonstrations: {e}")
        raise


if __name__ == "__main__":
    main()