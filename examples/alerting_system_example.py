"""
Example demonstrating the alerting and reporting system.

This example shows how to:
1. Set up alert rules and channels
2. Configure different types of alert channels (email, Slack, webhook)
3. Generate alerts based on metrics
4. Manage alert lifecycle (resolve, suppress, escalate)
5. Generate alert reports and statistics
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

from mlpipeline.monitoring.alerts import (
    Alert,
    AlertRule,
    AlertManager,
    EmailAlertChannel,
    SlackAlertChannel,
    WebhookAlertChannel,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_alert_rules():
    """Create sample alert rules for demonstration."""
    rules = [
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
            cooldown_minutes=15,
            escalation_minutes=60
        ),
        AlertRule(
            name="model_performance_degradation",
            condition="accuracy < 0.8",
            severity="medium",
            channels=["email"],
            threshold_value=0.8,
            comparison_operator="<",
            metric_name="accuracy",
            cooldown_minutes=60
        ),
        AlertRule(
            name="prediction_drift",
            condition="prediction_drift_score > 0.05",
            severity="medium",
            channels=["slack"],
            threshold_value=0.05,
            comparison_operator=">",
            metric_name="prediction_drift_score",
            cooldown_minutes=45
        ),
        AlertRule(
            name="missing_data_threshold",
            condition="missing_data_percentage > 10",
            severity="low",
            channels=["email"],
            threshold_value=10.0,
            comparison_operator=">",
            metric_name="missing_data_percentage",
            cooldown_minutes=120
        )
    ]
    
    return rules


def create_sample_alert_channels():
    """Create sample alert channels for demonstration."""
    channels = {}
    
    # Email channel configuration
    email_config = {
        'enabled': True,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'alerts@example.com',
        'password': 'your_app_password',  # Use app password for Gmail
        'from_email': 'ml-pipeline-alerts@example.com',
        'to_emails': ['admin@example.com', 'data-team@example.com'],
        'use_tls': True
    }
    channels['email'] = EmailAlertChannel('email', email_config)
    
    # Slack channel configuration
    slack_config = {
        'enabled': True,
        'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
        'channel': '#ml-alerts',
        'username': 'ML Pipeline Bot'
    }
    channels['slack'] = SlackAlertChannel('slack', slack_config)
    
    # Webhook channel configuration
    webhook_config = {
        'enabled': True,
        'webhook_url': 'https://api.example.com/alerts',
        'headers': {
            'Authorization': 'Bearer your_api_token',
            'Content-Type': 'application/json'
        },
        'method': 'POST',
        'timeout': 30
    }
    channels['webhook'] = WebhookAlertChannel('webhook', webhook_config)
    
    return channels


def demonstrate_basic_alerting():
    """Demonstrate basic alerting functionality."""
    logger.info("=== Basic Alerting Demo ===")
    
    # Create alert manager
    manager = AlertManager(alert_history_file="alert_history.json")
    
    # Add rules and channels
    rules = create_sample_alert_rules()
    for rule in rules:
        manager.add_rule(rule)
    
    channels = create_sample_alert_channels()
    for channel in channels.values():
        manager.add_channel(channel)
    
    logger.info(f"Configured {len(manager.rules)} alert rules")
    logger.info(f"Configured {len(manager.channels)} alert channels")
    
    # Test channel connections (will fail with demo config, but shows the concept)
    logger.info("\n--- Testing Alert Channels ---")
    channel_results = manager.test_channels()
    for channel_name, result in channel_results.items():
        status = "✓ Connected" if result else "✗ Failed"
        logger.info(f"{channel_name}: {status}")
    
    # Simulate metrics that trigger alerts
    logger.info("\n--- Simulating Metric Evaluation ---")
    
    # Metrics that should trigger alerts
    high_drift_metrics = {
        "data_drift_score": 0.15,  # Triggers high_data_drift rule
        "accuracy": 0.75,          # Triggers model_performance_degradation rule
        "prediction_drift_score": 0.03,  # Below threshold
        "missing_data_percentage": 5.0    # Below threshold
    }
    
    alerts = manager.evaluate_metrics(high_drift_metrics, "production_model")
    logger.info(f"Generated {len(alerts)} alerts from high drift metrics")
    
    for alert in alerts:
        logger.info(f"  - {alert.title} ({alert.severity}): {alert.message}")
    
    # Metrics that trigger critical alerts
    critical_metrics = {
        "data_drift_score": 0.25,  # Triggers critical_data_drift rule
        "accuracy": 0.70,          # Triggers model_performance_degradation rule
        "prediction_drift_score": 0.08,  # Triggers prediction_drift rule
        "missing_data_percentage": 15.0   # Triggers missing_data_threshold rule
    }
    
    critical_alerts = manager.evaluate_metrics(critical_metrics, "production_model")
    logger.info(f"Generated {len(critical_alerts)} alerts from critical metrics")
    
    for alert in critical_alerts:
        logger.info(f"  - {alert.title} ({alert.severity}): {alert.message}")
    
    return manager


def demonstrate_alert_management(manager):
    """Demonstrate alert management features."""
    logger.info("\n=== Alert Management Demo ===")
    
    # Get active alerts
    active_alerts = manager.get_active_alerts()
    logger.info(f"Active alerts: {len(active_alerts)}")
    
    if active_alerts:
        # Resolve first alert
        first_alert = active_alerts[0]
        logger.info(f"Resolving alert: {first_alert.title}")
        manager.resolve_alert(first_alert.id, "Issue has been fixed by retraining model")
        
        # Suppress second alert if exists
        if len(active_alerts) > 1:
            second_alert = active_alerts[1]
            suppress_until = datetime.now() + timedelta(hours=2)
            logger.info(f"Suppressing alert until {suppress_until}: {second_alert.title}")
            manager.suppress_alert(second_alert.id, suppress_until)
        
        # Escalate third alert if exists
        if len(active_alerts) > 2:
            third_alert = active_alerts[2]
            logger.info(f"Escalating alert: {third_alert.title}")
            manager.escalate_alert(third_alert.id, ["webhook"])
    
    # Show updated active alerts
    updated_active = manager.get_active_alerts()
    logger.info(f"Active alerts after management: {len(updated_active)}")


def demonstrate_alert_reporting(manager):
    """Demonstrate alert reporting and statistics."""
    logger.info("\n=== Alert Reporting Demo ===")
    
    # Get alert statistics
    stats = manager.get_alert_statistics()
    logger.info("Alert Statistics:")
    logger.info(f"  Total alerts: {stats['total_alerts']}")
    logger.info(f"  Active alerts: {stats['active_alerts']}")
    logger.info(f"  Recent alerts (24h): {stats['recent_alerts_24h']}")
    logger.info(f"  Resolution rate: {stats['resolution_rate']:.2%}")
    
    logger.info("\nSeverity Distribution:")
    for severity, count in stats['severity_distribution'].items():
        logger.info(f"  {severity}: {count}")
    
    logger.info("\nAlert Type Distribution:")
    for alert_type, count in stats['type_distribution'].items():
        logger.info(f"  {alert_type}: {count}")
    
    # Get alert history
    history = manager.get_alert_history(limit=10)
    logger.info(f"\nRecent Alert History ({len(history)} alerts):")
    for alert in history:
        status = "RESOLVED" if alert.resolved else "ACTIVE"
        if alert.suppressed:
            status += " (SUPPRESSED)"
        if alert.escalated:
            status += " (ESCALATED)"
        
        logger.info(f"  [{alert.timestamp.strftime('%H:%M:%S')}] {alert.title} - {status}")
    
    # Get alerts by severity
    high_severity_alerts = manager.get_alert_history(severity="high")
    critical_alerts = manager.get_alert_history(severity="critical")
    
    logger.info(f"\nHigh severity alerts: {len(high_severity_alerts)}")
    logger.info(f"Critical alerts: {len(critical_alerts)}")


def demonstrate_alert_filtering(manager):
    """Demonstrate alert filtering and querying."""
    logger.info("\n=== Alert Filtering Demo ===")
    
    # Filter by time range
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent_alerts = manager.get_alert_history(start_time=one_hour_ago)
    logger.info(f"Alerts in last hour: {len(recent_alerts)}")
    
    # Filter by alert type
    drift_alerts = manager.get_alert_history(alert_type="high_data_drift")
    logger.info(f"Data drift alerts: {len(drift_alerts)}")
    
    performance_alerts = manager.get_alert_history(alert_type="model_performance_degradation")
    logger.info(f"Performance degradation alerts: {len(performance_alerts)}")
    
    # Get active alerts by severity
    critical_active = manager.get_active_alerts(severity="critical")
    high_active = manager.get_active_alerts(severity="high")
    medium_active = manager.get_active_alerts(severity="medium")
    
    logger.info(f"Active critical alerts: {len(critical_active)}")
    logger.info(f"Active high severity alerts: {len(high_active)}")
    logger.info(f"Active medium severity alerts: {len(medium_active)}")


def demonstrate_cooldown_behavior(manager):
    """Demonstrate alert cooldown behavior."""
    logger.info("\n=== Alert Cooldown Demo ===")
    
    # Create a rule with short cooldown for demonstration
    short_cooldown_rule = AlertRule(
        name="demo_cooldown_rule",
        condition="test_metric > 5",
        severity="low",
        channels=["email"],
        threshold_value=5.0,
        comparison_operator=">",
        metric_name="test_metric",
        cooldown_minutes=0.1  # 6 seconds for demo
    )
    
    manager.add_rule(short_cooldown_rule)
    
    # Trigger alert multiple times
    test_metrics = {"test_metric": 10.0}
    
    logger.info("Triggering alert first time...")
    alerts1 = manager.evaluate_metrics(test_metrics, "cooldown_test")
    logger.info(f"Alerts generated: {len(alerts1)}")
    
    logger.info("Triggering alert immediately (should be suppressed)...")
    alerts2 = manager.evaluate_metrics(test_metrics, "cooldown_test")
    logger.info(f"Alerts generated: {len(alerts2)}")
    
    logger.info("Waiting for cooldown period...")
    time.sleep(7)  # Wait longer than cooldown period
    
    logger.info("Triggering alert after cooldown...")
    alerts3 = manager.evaluate_metrics(test_metrics, "cooldown_test")
    logger.info(f"Alerts generated: {len(alerts3)}")


def create_alert_dashboard_data(manager):
    """Create data for alert dashboard visualization."""
    logger.info("\n=== Creating Dashboard Data ===")
    
    # Prepare dashboard data
    dashboard_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": manager.get_alert_statistics(),
        "active_alerts": [
            {
                "id": alert.id,
                "title": alert.title,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "alert_type": alert.alert_type
            }
            for alert in manager.get_active_alerts()
        ],
        "recent_history": [
            {
                "id": alert.id,
                "title": alert.title,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "alert_type": alert.alert_type,
                "resolved": alert.resolved,
                "escalated": alert.escalated,
                "suppressed": alert.suppressed
            }
            for alert in manager.get_alert_history(limit=20)
        ]
    }
    
    # Save dashboard data
    dashboard_file = "alert_dashboard_data.json"
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    logger.info(f"Dashboard data saved to {dashboard_file}")
    return dashboard_data


def main():
    """Run all alerting system demonstrations."""
    logger.info("Starting Alerting System Examples")
    
    try:
        # Run demonstrations
        manager = demonstrate_basic_alerting()
        demonstrate_alert_management(manager)
        demonstrate_alert_reporting(manager)
        demonstrate_alert_filtering(manager)
        demonstrate_cooldown_behavior(manager)
        create_alert_dashboard_data(manager)
        
        logger.info("\n=== All alerting demonstrations completed successfully! ===")
        logger.info("Check 'alert_history.json' for persistent alert history.")
        logger.info("Check 'alert_dashboard_data.json' for dashboard data.")
        
    except Exception as e:
        logger.error(f"Error in demonstrations: {e}")
        raise


if __name__ == "__main__":
    main()