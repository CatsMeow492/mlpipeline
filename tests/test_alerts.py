"""
Tests for alert management system.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mlpipeline.monitoring.alerts import (
    Alert,
    AlertRule,
    AlertManager,
    EmailAlertChannel,
    SlackAlertChannel,
    WebhookAlertChannel,
)


class TestAlert:
    """Test cases for Alert dataclass."""
    
    def test_alert_creation(self):
        """Test Alert creation with required fields."""
        alert = Alert(
            id="test_alert_1",
            alert_type="drift_detection",
            severity="high",
            title="Data Drift Detected",
            message="Significant drift detected in feature1",
            timestamp=datetime.now(),
            source="drift_detector"
        )
        
        assert alert.id == "test_alert_1"
        assert alert.alert_type == "drift_detection"
        assert alert.severity == "high"
        assert alert.title == "Data Drift Detected"
        assert alert.resolved is False
        assert alert.escalated is False
        assert alert.suppressed is False
    
    def test_alert_with_metadata(self):
        """Test Alert creation with metadata."""
        metadata = {"drift_score": 0.15, "feature": "age"}
        alert = Alert(
            id="test_alert_2",
            alert_type="drift_detection",
            severity="medium",
            title="Feature Drift",
            message="Drift in age feature",
            timestamp=datetime.now(),
            source="feature_monitor",
            metadata=metadata
        )
        
        assert alert.metadata == metadata
        assert alert.metadata["drift_score"] == 0.15


class TestAlertRule:
    """Test cases for AlertRule dataclass."""
    
    def test_alert_rule_creation(self):
        """Test AlertRule creation."""
        rule = AlertRule(
            name="high_drift_alert",
            condition="drift_score > 0.1",
            severity="high",
            channels=["email", "slack"],
            threshold_value=0.1,
            comparison_operator=">",
            metric_name="drift_score"
        )
        
        assert rule.name == "high_drift_alert"
        assert rule.severity == "high"
        assert rule.channels == ["email", "slack"]
        assert rule.threshold_value == 0.1
        assert rule.comparison_operator == ">"
        assert rule.metric_name == "drift_score"
        assert rule.enabled is True
        assert rule.cooldown_minutes == 60


class TestEmailAlertChannel:
    """Test cases for EmailAlertChannel."""
    
    def test_email_channel_initialization(self):
        """Test EmailAlertChannel initialization."""
        config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'test@example.com',
            'password': 'password',
            'from_email': 'alerts@example.com',
            'to_emails': ['admin@example.com'],
            'use_tls': True
        }
        
        channel = EmailAlertChannel('email', config)
        
        assert channel.name == 'email'
        assert channel.smtp_server == 'smtp.gmail.com'
        assert channel.smtp_port == 587
        assert channel.username == 'test@example.com'
        assert channel.to_emails == ['admin@example.com']
        assert channel.use_tls is True
    
    def test_create_email_body(self):
        """Test email body creation."""
        config = {
            'from_email': 'alerts@example.com',
            'to_emails': ['admin@example.com']
        }
        channel = EmailAlertChannel('email', config)
        
        alert = Alert(
            id="test_alert",
            alert_type="drift_detection",
            severity="high",
            title="Test Alert",
            message="This is a test alert",
            timestamp=datetime.now(),
            source="test_source",
            metadata={"test_key": "test_value"}
        )
        
        body = channel._create_email_body(alert)
        
        assert "Test Alert" in body
        assert "high" in body
        assert "This is a test alert" in body
        assert "test_source" in body
        assert "test_key" in body
    
    @patch('smtplib.SMTP')
    def test_send_alert_success(self, mock_smtp):
        """Test successful email alert sending."""
        config = {
            'smtp_server': 'smtp.example.com',
            'from_email': 'alerts@example.com',
            'to_emails': ['admin@example.com']
        }
        channel = EmailAlertChannel('email', config)
        
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        alert = Alert(
            id="test_alert",
            alert_type="test",
            severity="medium",
            title="Test Alert",
            message="Test message",
            timestamp=datetime.now(),
            source="test"
        )
        
        result = channel.send_alert(alert)
        
        assert result is True
        mock_server.send_message.assert_called_once()
    
    @patch('smtplib.SMTP')
    def test_send_alert_failure(self, mock_smtp):
        """Test email alert sending failure."""
        config = {
            'smtp_server': 'smtp.example.com',
            'from_email': 'alerts@example.com',
            'to_emails': ['admin@example.com']
        }
        channel = EmailAlertChannel('email', config)
        
        # Mock SMTP server to raise exception
        mock_smtp.side_effect = Exception("SMTP Error")
        
        alert = Alert(
            id="test_alert",
            alert_type="test",
            severity="medium",
            title="Test Alert",
            message="Test message",
            timestamp=datetime.now(),
            source="test"
        )
        
        result = channel.send_alert(alert)
        
        assert result is False


class TestSlackAlertChannel:
    """Test cases for SlackAlertChannel."""
    
    def test_slack_channel_initialization(self):
        """Test SlackAlertChannel initialization."""
        config = {
            'webhook_url': 'https://hooks.slack.com/test',
            'channel': '#alerts',
            'username': 'AlertBot'
        }
        
        channel = SlackAlertChannel('slack', config)
        
        assert channel.name == 'slack'
        assert channel.webhook_url == 'https://hooks.slack.com/test'
        assert channel.channel == '#alerts'
        assert channel.username == 'AlertBot'
    
    def test_create_slack_payload(self):
        """Test Slack payload creation."""
        config = {
            'webhook_url': 'https://hooks.slack.com/test',
            'channel': '#alerts',
            'username': 'AlertBot'
        }
        channel = SlackAlertChannel('slack', config)
        
        alert = Alert(
            id="test_alert",
            alert_type="drift_detection",
            severity="high",
            title="Test Alert",
            message="This is a test alert",
            timestamp=datetime.now(),
            source="test_source"
        )
        
        payload = channel._create_slack_payload(alert)
        
        assert payload['username'] == 'AlertBot'
        assert payload['channel'] == '#alerts'
        assert len(payload['attachments']) == 1
        
        attachment = payload['attachments'][0]
        assert attachment['title'] == 'Test Alert'
        assert attachment['text'] == 'This is a test alert'
        assert attachment['color'] == 'danger'  # high severity
    
    @patch('urllib.request.urlopen')
    def test_send_alert_success(self, mock_urlopen):
        """Test successful Slack alert sending."""
        config = {
            'webhook_url': 'https://hooks.slack.com/test'
        }
        channel = SlackAlertChannel('slack', config)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        alert = Alert(
            id="test_alert",
            alert_type="test",
            severity="medium",
            title="Test Alert",
            message="Test message",
            timestamp=datetime.now(),
            source="test"
        )
        
        result = channel.send_alert(alert)
        
        assert result is True
        mock_urlopen.assert_called_once()


class TestWebhookAlertChannel:
    """Test cases for WebhookAlertChannel."""
    
    def test_webhook_channel_initialization(self):
        """Test WebhookAlertChannel initialization."""
        config = {
            'webhook_url': 'https://api.example.com/alerts',
            'headers': {'Authorization': 'Bearer token'},
            'method': 'POST',
            'timeout': 30
        }
        
        channel = WebhookAlertChannel('webhook', config)
        
        assert channel.name == 'webhook'
        assert channel.webhook_url == 'https://api.example.com/alerts'
        assert channel.headers == {'Authorization': 'Bearer token'}
        assert channel.method == 'POST'
        assert channel.timeout == 30
    
    def test_create_webhook_payload(self):
        """Test webhook payload creation."""
        config = {'webhook_url': 'https://api.example.com/alerts'}
        channel = WebhookAlertChannel('webhook', config)
        
        alert = Alert(
            id="test_alert",
            alert_type="drift_detection",
            severity="high",
            title="Test Alert",
            message="This is a test alert",
            timestamp=datetime.now(),
            source="test_source",
            metadata={"test_key": "test_value"}
        )
        
        payload = channel._create_webhook_payload(alert)
        
        assert payload['alert_id'] == 'test_alert'
        assert payload['alert_type'] == 'drift_detection'
        assert payload['severity'] == 'high'
        assert payload['title'] == 'Test Alert'
        assert payload['message'] == 'This is a test alert'
        assert payload['source'] == 'test_source'
        assert payload['metadata'] == {"test_key": "test_value"}
        assert payload['resolved'] is False


class TestAlertManager:
    """Test cases for AlertManager."""
    
    @pytest.fixture
    def sample_rule(self):
        """Create a sample alert rule."""
        return AlertRule(
            name="test_rule",
            condition="drift_score > 0.1",
            severity="high",
            channels=["test_channel"],
            threshold_value=0.1,
            comparison_operator=">",
            metric_name="drift_score"
        )
    
    @pytest.fixture
    def mock_channel(self):
        """Create a mock alert channel."""
        channel = Mock()
        channel.name = "test_channel"
        channel.send_alert.return_value = True
        return channel
    
    def test_alert_manager_initialization(self):
        """Test AlertManager initialization."""
        manager = AlertManager()
        
        assert manager.rules == []
        assert manager.channels == {}
        assert manager.alert_history == []
        assert manager.active_alerts == {}
        assert manager.last_alert_times == {}
    
    def test_add_rule(self, sample_rule):
        """Test adding alert rule."""
        manager = AlertManager()
        manager.add_rule(sample_rule)
        
        assert len(manager.rules) == 1
        assert manager.rules[0] == sample_rule
    
    def test_add_channel(self, mock_channel):
        """Test adding alert channel."""
        manager = AlertManager()
        manager.add_channel(mock_channel)
        
        assert "test_channel" in manager.channels
        assert manager.channels["test_channel"] == mock_channel
    
    def test_check_condition(self):
        """Test condition checking logic."""
        manager = AlertManager()
        
        assert manager._check_condition(0.15, ">", 0.1) is True
        assert manager._check_condition(0.05, ">", 0.1) is False
        assert manager._check_condition(0.1, ">=", 0.1) is True
        assert manager._check_condition(0.05, "<", 0.1) is True
        assert manager._check_condition(0.1, "==", 0.1) is True
    
    def test_evaluate_metrics_no_alert(self, sample_rule, mock_channel):
        """Test metric evaluation that doesn't trigger alert."""
        manager = AlertManager()
        manager.add_rule(sample_rule)
        manager.add_channel(mock_channel)
        
        # Metric value below threshold
        metrics = {"drift_score": 0.05}
        alerts = manager.evaluate_metrics(metrics, "test_source")
        
        assert len(alerts) == 0
        mock_channel.send_alert.assert_not_called()
    
    def test_evaluate_metrics_with_alert(self, sample_rule, mock_channel):
        """Test metric evaluation that triggers alert."""
        manager = AlertManager()
        manager.add_rule(sample_rule)
        manager.add_channel(mock_channel)
        
        # Metric value above threshold
        metrics = {"drift_score": 0.15}
        alerts = manager.evaluate_metrics(metrics, "test_source")
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.alert_type == "test_rule"
        assert alert.severity == "high"
        assert alert.source == "test_source"
        assert alert.metadata["metric_value"] == 0.15
        
        mock_channel.send_alert.assert_called_once_with(alert)
    
    def test_evaluate_metrics_cooldown(self, sample_rule, mock_channel):
        """Test alert cooldown functionality."""
        # Set short cooldown for testing
        sample_rule.cooldown_minutes = 0.01  # ~0.6 seconds
        
        manager = AlertManager()
        manager.add_rule(sample_rule)
        manager.add_channel(mock_channel)
        
        metrics = {"drift_score": 0.15}
        
        # First alert should be generated
        alerts1 = manager.evaluate_metrics(metrics, "test_source")
        assert len(alerts1) == 1
        
        # Second alert immediately should be suppressed
        alerts2 = manager.evaluate_metrics(metrics, "test_source")
        assert len(alerts2) == 0
        
        # Only one call to send_alert
        assert mock_channel.send_alert.call_count == 1
    
    def test_resolve_alert(self, sample_rule, mock_channel):
        """Test alert resolution."""
        manager = AlertManager()
        manager.add_rule(sample_rule)
        manager.add_channel(mock_channel)
        
        # Generate alert
        metrics = {"drift_score": 0.15}
        alerts = manager.evaluate_metrics(metrics, "test_source")
        alert = alerts[0]
        
        # Resolve alert
        result = manager.resolve_alert(alert.id, "Issue fixed")
        
        assert result is True
        assert alert.resolved is True
        assert alert.resolved_at is not None
        assert alert.metadata["resolution_message"] == "Issue fixed"
        assert alert.id not in manager.active_alerts
    
    def test_suppress_alert(self, sample_rule, mock_channel):
        """Test alert suppression."""
        manager = AlertManager()
        manager.add_rule(sample_rule)
        manager.add_channel(mock_channel)
        
        # Generate alert
        metrics = {"drift_score": 0.15}
        alerts = manager.evaluate_metrics(metrics, "test_source")
        alert = alerts[0]
        
        # Suppress alert
        suppress_until = datetime.now() + timedelta(hours=1)
        result = manager.suppress_alert(alert.id, suppress_until)
        
        assert result is True
        assert alert.suppressed is True
        assert alert.suppressed_until == suppress_until
    
    def test_escalate_alert(self, sample_rule, mock_channel):
        """Test alert escalation."""
        manager = AlertManager()
        manager.add_rule(sample_rule)
        manager.add_channel(mock_channel)
        
        # Add escalation channel
        escalation_channel = Mock()
        escalation_channel.name = "escalation_channel"
        escalation_channel.send_alert.return_value = True
        manager.add_channel(escalation_channel)
        
        # Generate alert
        metrics = {"drift_score": 0.15}
        alerts = manager.evaluate_metrics(metrics, "test_source")
        alert = alerts[0]
        
        # Escalate alert
        result = manager.escalate_alert(alert.id, ["escalation_channel"])
        
        assert result is True
        assert alert.escalated is True
        assert alert.escalated_at is not None
        escalation_channel.send_alert.assert_called_once_with(alert)
    
    def test_get_active_alerts(self, sample_rule, mock_channel):
        """Test getting active alerts."""
        manager = AlertManager()
        manager.add_rule(sample_rule)
        manager.add_channel(mock_channel)
        
        # Generate alerts
        metrics = {"drift_score": 0.15}
        manager.evaluate_metrics(metrics, "source1")
        manager.evaluate_metrics(metrics, "source2")
        
        active_alerts = manager.get_active_alerts()
        assert len(active_alerts) == 2
        
        # Test severity filter
        high_alerts = manager.get_active_alerts(severity="high")
        assert len(high_alerts) == 2
        
        medium_alerts = manager.get_active_alerts(severity="medium")
        assert len(medium_alerts) == 0
    
    def test_get_alert_history(self, sample_rule, mock_channel):
        """Test getting alert history."""
        manager = AlertManager()
        manager.add_rule(sample_rule)
        manager.add_channel(mock_channel)
        
        # Generate alerts
        metrics = {"drift_score": 0.15}
        manager.evaluate_metrics(metrics, "source1")
        manager.evaluate_metrics(metrics, "source2")
        
        history = manager.get_alert_history()
        assert len(history) == 2
        
        # Test limit
        limited_history = manager.get_alert_history(limit=1)
        assert len(limited_history) == 1
        
        # Test severity filter
        high_history = manager.get_alert_history(severity="high")
        assert len(high_history) == 2
        
        medium_history = manager.get_alert_history(severity="medium")
        assert len(medium_history) == 0
    
    def test_get_alert_statistics(self, sample_rule, mock_channel):
        """Test alert statistics."""
        manager = AlertManager()
        manager.add_rule(sample_rule)
        manager.add_channel(mock_channel)
        
        # Generate alerts
        metrics = {"drift_score": 0.15}
        alerts = manager.evaluate_metrics(metrics, "source1")
        manager.evaluate_metrics(metrics, "source2")
        
        # Resolve one alert
        manager.resolve_alert(alerts[0].id)
        
        stats = manager.get_alert_statistics()
        
        assert stats['total_alerts'] == 2
        assert stats['active_alerts'] == 1
        assert stats['severity_distribution']['high'] == 2
        assert stats['type_distribution']['test_rule'] == 2
        assert stats['resolution_rate'] == 0.5
    
    def test_test_channels(self, mock_channel):
        """Test channel testing functionality."""
        mock_channel.test_connection.return_value = True
        
        manager = AlertManager()
        manager.add_channel(mock_channel)
        
        results = manager.test_channels()
        
        assert results["test_channel"] is True
        mock_channel.test_connection.assert_called_once()
    
    def test_alert_history_persistence(self, sample_rule, mock_channel):
        """Test alert history persistence."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            history_file = f.name
        
        try:
            # Create manager with history file
            manager = AlertManager(alert_history_file=history_file)
            manager.add_rule(sample_rule)
            manager.add_channel(mock_channel)
            
            # Generate alert
            metrics = {"drift_score": 0.15}
            manager.evaluate_metrics(metrics, "test_source")
            
            # Create new manager and load history
            manager2 = AlertManager(alert_history_file=history_file)
            
            assert len(manager2.alert_history) == 1
            loaded_alert = manager2.alert_history[0]
            assert loaded_alert.alert_type == "test_rule"
            assert loaded_alert.severity == "high"
            
        finally:
            Path(history_file).unlink()


if __name__ == '__main__':
    pytest.main([__file__])