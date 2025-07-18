"""
Alert management system for drift detection and monitoring.

This module provides configurable alerting capabilities with support for:
- Multiple alert channels (email, Slack, webhooks)
- Configurable thresholds and conditions
- Alert suppression and escalation logic
- Alert history and reporting
"""

import json
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from ..core.interfaces import PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
from ..core.errors import PipelineError


logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Represents an alert with metadata and content."""
    
    id: str
    alert_type: str
    severity: str
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalated_at: Optional[datetime] = None
    suppressed: bool = False
    suppressed_until: Optional[datetime] = None


@dataclass
class AlertRule:
    """Defines conditions and actions for alert generation."""
    
    name: str
    condition: str  # e.g., "drift_score > 0.1"
    severity: str  # "low", "medium", "high", "critical"
    channels: List[str]
    threshold_value: float
    comparison_operator: str  # ">", "<", ">=", "<=", "=="
    metric_name: str
    enabled: bool = True
    cooldown_minutes: int = 60
    escalation_minutes: Optional[int] = None
    suppression_conditions: Optional[Dict[str, Any]] = None


class AlertChannel(ABC):
    """Abstract base class for alert channels."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
    
    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the channel is properly configured and reachable."""
        pass


class EmailAlertChannel(AlertChannel):
    """Email alert channel using SMTP."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not self.enabled or not self.to_emails:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent successfully to {self.to_emails}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test SMTP connection."""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
            return True
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        severity_colors = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
        
        color = severity_colors.get(alert.severity, '#6c757d')
        
        return f"""
        <html>
        <body>
            <h2 style="color: {color};">{alert.title}</h2>
            <p><strong>Severity:</strong> <span style="color: {color};">{alert.severity.upper()}</span></p>
            <p><strong>Source:</strong> {alert.source}</p>
            <p><strong>Timestamp:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Alert Type:</strong> {alert.alert_type}</p>
            
            <h3>Message</h3>
            <p>{alert.message}</p>
            
            <h3>Metadata</h3>
            <pre>{json.dumps(alert.metadata, indent=2)}</pre>
        </body>
        </html>
        """


class SlackAlertChannel(AlertChannel):
    """Slack alert channel using webhooks."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel')
        self.username = config.get('username', 'ML Pipeline Alert')
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.enabled or not self.webhook_url:
            return False
        
        try:
            # Create Slack message
            payload = self._create_slack_payload(alert)
            
            # Send to Slack
            req = Request(
                self.webhook_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urlopen(req) as response:
                if response.status == 200:
                    logger.info(f"Slack alert sent successfully to {self.channel}")
                    return True
                else:
                    logger.error(f"Slack alert failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test Slack webhook."""
        if not self.webhook_url:
            return False
        
        try:
            test_payload = {
                "text": "ML Pipeline Alert System - Connection Test",
                "username": self.username
            }
            
            req = Request(
                self.webhook_url,
                data=json.dumps(test_payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urlopen(req) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Slack connection test failed: {e}")
            return False
    
    def _create_slack_payload(self, alert: Alert) -> Dict[str, Any]:
        """Create Slack message payload."""
        severity_colors = {
            'low': 'good',
            'medium': 'warning',
            'high': 'danger',
            'critical': 'danger'
        }
        
        color = severity_colors.get(alert.severity, 'good')
        
        attachment = {
            "color": color,
            "title": alert.title,
            "text": alert.message,
            "fields": [
                {"title": "Severity", "value": alert.severity.upper(), "short": True},
                {"title": "Source", "value": alert.source, "short": True},
                {"title": "Alert Type", "value": alert.alert_type, "short": True},
                {"title": "Timestamp", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
            ],
            "footer": "ML Pipeline Alert System",
            "ts": int(alert.timestamp.timestamp())
        }
        
        payload = {
            "username": self.username,
            "attachments": [attachment]
        }
        
        if self.channel:
            payload["channel"] = self.channel
        
        return payload


class WebhookAlertChannel(AlertChannel):
    """Generic webhook alert channel."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {})
        self.method = config.get('method', 'POST')
        self.timeout = config.get('timeout', 30)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        if not self.enabled or not self.webhook_url:
            return False
        
        try:
            # Create webhook payload
            payload = self._create_webhook_payload(alert)
            
            # Send webhook
            req = Request(
                self.webhook_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={**self.headers, 'Content-Type': 'application/json'}
            )
            
            with urlopen(req, timeout=self.timeout) as response:
                if 200 <= response.status < 300:
                    logger.info(f"Webhook alert sent successfully to {self.webhook_url}")
                    return True
                else:
                    logger.error(f"Webhook alert failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test webhook endpoint."""
        if not self.webhook_url:
            return False
        
        try:
            test_payload = {"test": True, "message": "Connection test"}
            
            req = Request(
                self.webhook_url,
                data=json.dumps(test_payload).encode('utf-8'),
                headers={**self.headers, 'Content-Type': 'application/json'}
            )
            
            with urlopen(req, timeout=self.timeout) as response:
                return 200 <= response.status < 300
                
        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False
    
    def _create_webhook_payload(self, alert: Alert) -> Dict[str, Any]:
        """Create webhook payload."""
        return {
            "alert_id": alert.id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "title": alert.title,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "source": alert.source,
            "metadata": alert.metadata,
            "resolved": alert.resolved,
            "escalated": alert.escalated,
            "suppressed": alert.suppressed
        }


class AlertManager(PipelineComponent):
    """
    Manages alert generation, routing, and lifecycle.
    
    Features:
    - Multiple alert channels (email, Slack, webhooks)
    - Configurable alert rules and thresholds
    - Alert suppression and escalation
    - Alert history and reporting
    """
    
    def __init__(
        self,
        rules: Optional[List[AlertRule]] = None,
        channels: Optional[Dict[str, AlertChannel]] = None,
        alert_history_file: Optional[str] = None,
        max_history_size: int = 10000,
    ):
        """
        Initialize AlertManager.
        
        Args:
            rules: List of alert rules
            channels: Dictionary of alert channels
            alert_history_file: File to persist alert history
            max_history_size: Maximum number of alerts to keep in history
        """
        from ..core.interfaces import ComponentType
        super().__init__(ComponentType.DRIFT_DETECTION)
        self.rules = rules or []
        self.channels = channels or {}
        self.alert_history_file = alert_history_file
        self.max_history_size = max_history_size
        self.alert_history: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Load alert history if file exists
        if alert_history_file:
            self._load_alert_history()
    

    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate alert manager configuration."""
        # Basic validation - can be extended
        return True
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute alert evaluation on provided metrics."""
        try:
            metrics = context.metadata.get('metrics', {})
            source = context.metadata.get('source', 'pipeline')
            
            # Evaluate metrics and generate alerts
            alerts = self.evaluate_metrics(metrics, source)
            
            return ExecutionResult(
                success=True,
                metrics={'alerts_generated': len(alerts)},
                metadata={'alerts': [alert.to_dict() for alert in alerts]}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=f"Alert evaluation failed: {str(e)}"
            )
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_channel(self, channel: AlertChannel) -> None:
        """Add an alert channel."""
        self.channels[channel.name] = channel
        logger.info(f"Added alert channel: {channel.name}")
    
    def evaluate_metrics(self, metrics: Dict[str, Any], source: str = "unknown") -> List[Alert]:
        """
        Evaluate metrics against alert rules and generate alerts.
        
        Args:
            metrics: Dictionary of metric values
            source: Source of the metrics
            
        Returns:
            List of generated alerts
        """
        generated_alerts = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                alert = self._evaluate_rule(rule, metrics, source)
                if alert:
                    generated_alerts.append(alert)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        return generated_alerts
    
    def _evaluate_rule(self, rule: AlertRule, metrics: Dict[str, Any], source: str) -> Optional[Alert]:
        """Evaluate a single rule against metrics."""
        metric_value = metrics.get(rule.metric_name)
        if metric_value is None:
            return None
        
        # Check if condition is met
        condition_met = self._check_condition(
            metric_value, 
            rule.comparison_operator, 
            rule.threshold_value
        )
        
        if not condition_met:
            return None
        
        # Check cooldown period
        rule_key = f"{rule.name}_{source}"
        last_alert_time = self.last_alert_times.get(rule_key)
        if last_alert_time:
            cooldown_period = timedelta(minutes=rule.cooldown_minutes)
            if datetime.now() - last_alert_time < cooldown_period:
                logger.debug(f"Rule {rule.name} in cooldown period")
                return None
        
        # Generate alert
        alert_id = f"{rule.name}_{source}_{int(time.time())}"
        alert = Alert(
            id=alert_id,
            alert_type=rule.name,
            severity=rule.severity,
            title=f"{rule.name} Alert",
            message=f"Metric '{rule.metric_name}' value {metric_value} {rule.comparison_operator} {rule.threshold_value}",
            timestamp=datetime.now(),
            source=source,
            metadata={
                'rule_name': rule.name,
                'metric_name': rule.metric_name,
                'metric_value': metric_value,
                'threshold_value': rule.threshold_value,
                'comparison_operator': rule.comparison_operator,
                'all_metrics': metrics
            }
        )
        
        # Send alert
        self._send_alert(alert, rule.channels)
        
        # Update tracking
        self.last_alert_times[rule_key] = alert.timestamp
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Trim history if needed
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        # Save history
        if self.alert_history_file:
            self._save_alert_history()
        
        logger.info(f"Generated alert: {alert.title}")
        return alert
    
    def _check_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Check if condition is met."""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        else:
            logger.warning(f"Unknown comparison operator: {operator}")
            return False
    
    def _send_alert(self, alert: Alert, channel_names: List[str]) -> None:
        """Send alert through specified channels."""
        for channel_name in channel_names:
            channel = self.channels.get(channel_name)
            if not channel:
                logger.warning(f"Alert channel not found: {channel_name}")
                continue
            
            try:
                success = channel.send_alert(alert)
                if success:
                    logger.info(f"Alert sent successfully via {channel_name}")
                else:
                    logger.error(f"Failed to send alert via {channel_name}")
            except Exception as e:
                logger.error(f"Error sending alert via {channel_name}: {e}")
    
    def resolve_alert(self, alert_id: str, resolution_message: Optional[str] = None) -> bool:
        """Mark an alert as resolved."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            logger.warning(f"Alert not found: {alert_id}")
            return False
        
        alert.resolved = True
        alert.resolved_at = datetime.now()
        
        if resolution_message:
            alert.metadata['resolution_message'] = resolution_message
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert resolved: {alert_id}")
        return True
    
    def suppress_alert(self, alert_id: str, suppress_until: datetime) -> bool:
        """Suppress an alert until specified time."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            logger.warning(f"Alert not found: {alert_id}")
            return False
        
        alert.suppressed = True
        alert.suppressed_until = suppress_until
        
        logger.info(f"Alert suppressed until {suppress_until}: {alert_id}")
        return True
    
    def escalate_alert(self, alert_id: str, escalation_channels: List[str]) -> bool:
        """Escalate an alert to additional channels."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            logger.warning(f"Alert not found: {alert_id}")
            return False
        
        alert.escalated = True
        alert.escalated_at = datetime.now()
        
        # Send to escalation channels
        self._send_alert(alert, escalation_channels)
        
        logger.info(f"Alert escalated: {alert_id}")
        return True
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_history(
        self, 
        limit: Optional[int] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Alert]:
        """Get alert history with optional filters."""
        alerts = self.alert_history.copy()
        
        # Apply filters
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        if alert_type:
            alerts = [alert for alert in alerts if alert.alert_type == alert_type]
        
        if start_time:
            alerts = [alert for alert in alerts if alert.timestamp >= start_time]
        
        if end_time:
            alerts = [alert for alert in alerts if alert.timestamp <= end_time]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        # Count by severity
        severity_counts = {}
        for alert in self.alert_history:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        # Count by type
        type_counts = {}
        for alert in self.alert_history:
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_alerts = [alert for alert in self.alert_history if alert.timestamp >= recent_cutoff]
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'recent_alerts_24h': len(recent_alerts),
            'severity_distribution': severity_counts,
            'type_distribution': type_counts,
            'resolution_rate': (total_alerts - active_alerts) / total_alerts if total_alerts > 0 else 0.0,
        }
    
    def test_channels(self) -> Dict[str, bool]:
        """Test all configured alert channels."""
        results = {}
        
        for name, channel in self.channels.items():
            try:
                results[name] = channel.test_connection()
            except Exception as e:
                logger.error(f"Error testing channel {name}: {e}")
                results[name] = False
        
        return results
    
    def _load_alert_history(self) -> None:
        """Load alert history from file."""
        try:
            history_path = Path(self.alert_history_file)
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history_data = json.load(f)
                
                self.alert_history = []
                for alert_data in history_data:
                    alert = Alert(
                        id=alert_data['id'],
                        alert_type=alert_data['alert_type'],
                        severity=alert_data['severity'],
                        title=alert_data['title'],
                        message=alert_data['message'],
                        timestamp=datetime.fromisoformat(alert_data['timestamp']),
                        source=alert_data['source'],
                        metadata=alert_data.get('metadata', {}),
                        resolved=alert_data.get('resolved', False),
                        resolved_at=datetime.fromisoformat(alert_data['resolved_at']) if alert_data.get('resolved_at') else None,
                        escalated=alert_data.get('escalated', False),
                        escalated_at=datetime.fromisoformat(alert_data['escalated_at']) if alert_data.get('escalated_at') else None,
                        suppressed=alert_data.get('suppressed', False),
                        suppressed_until=datetime.fromisoformat(alert_data['suppressed_until']) if alert_data.get('suppressed_until') else None,
                    )
                    self.alert_history.append(alert)
                
                logger.info(f"Loaded {len(self.alert_history)} alerts from history")
                
        except Exception as e:
            logger.error(f"Failed to load alert history: {e}")
    
    def _save_alert_history(self) -> None:
        """Save alert history to file."""
        try:
            history_data = []
            for alert in self.alert_history:
                alert_data = {
                    'id': alert.id,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'source': alert.source,
                    'metadata': alert.metadata,
                    'resolved': alert.resolved,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                    'escalated': alert.escalated,
                    'escalated_at': alert.escalated_at.isoformat() if alert.escalated_at else None,
                    'suppressed': alert.suppressed,
                    'suppressed_until': alert.suppressed_until.isoformat() if alert.suppressed_until else None,
                }
                history_data.append(alert_data)
            
            history_path = Path(self.alert_history_file)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save alert history: {e}")