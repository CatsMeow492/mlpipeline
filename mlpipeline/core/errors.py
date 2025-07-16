"""Error handling and recovery mechanisms for pipeline execution."""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import json


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of pipeline errors."""
    CONFIGURATION = "configuration"
    DATA = "data"
    MODEL = "model"
    SYSTEM = "system"
    NETWORK = "network"
    RESOURCE = "resource"
    VALIDATION = "validation"
    DEPENDENCY = "dependency"


class RecoveryAction(Enum):
    """Available recovery actions."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ABORT = "abort"
    CONTINUE = "continue"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_id: str
    timestamp: float
    stage_name: str
    component_name: Optional[str]
    experiment_id: str
    execution_id: str
    error_message: str
    exception_type: str
    stack_trace: str
    category: ErrorCategory
    severity: ErrorSeverity
    metadata: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration."""
    action: RecoveryAction
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    fallback_component: Optional[str] = None
    skip_dependencies: bool = False
    continue_on_failure: bool = False


class PipelineError(Exception):
    """Base class for pipeline-specific errors."""
    
    def __init__(self, message: str, category: ErrorCategory, severity: ErrorSeverity, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class ConfigurationError(PipelineError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, context)


class DataError(PipelineError):
    """Data-related errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.DATA, severity, context)


class ModelError(PipelineError):
    """Model-related errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.MODEL, severity, context)


class SystemError(PipelineError):
    """System-related errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.CRITICAL, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.SYSTEM, severity, context)


class ResourceError(PipelineError):
    """Resource-related errors (memory, disk, etc.)."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.RESOURCE, severity, context)


class NetworkError(PipelineError):
    """Network-related errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.NETWORK, severity, context)


class ErrorHandler:
    """Handles error classification, recovery, and reporting."""
    
    def __init__(self, error_log_path: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_log_path = Path(error_log_path) if error_log_path else None
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorCategory, RecoveryStrategy] = self._default_recovery_strategies()
        self.error_handlers: Dict[ErrorCategory, Callable] = {}
        
        if self.error_log_path:
            self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _default_recovery_strategies(self) -> Dict[ErrorCategory, RecoveryStrategy]:
        """Define default recovery strategies for different error categories."""
        return {
            ErrorCategory.CONFIGURATION: RecoveryStrategy(
                action=RecoveryAction.ABORT,
                max_retries=0
            ),
            ErrorCategory.DATA: RecoveryStrategy(
                action=RecoveryAction.RETRY,
                max_retries=2,
                retry_delay=5.0
            ),
            ErrorCategory.MODEL: RecoveryStrategy(
                action=RecoveryAction.FALLBACK,
                max_retries=1,
                retry_delay=2.0
            ),
            ErrorCategory.SYSTEM: RecoveryStrategy(
                action=RecoveryAction.RETRY,
                max_retries=3,
                retry_delay=10.0,
                exponential_backoff=True
            ),
            ErrorCategory.NETWORK: RecoveryStrategy(
                action=RecoveryAction.RETRY,
                max_retries=5,
                retry_delay=2.0,
                exponential_backoff=True
            ),
            ErrorCategory.RESOURCE: RecoveryStrategy(
                action=RecoveryAction.RETRY,
                max_retries=2,
                retry_delay=30.0
            ),
            ErrorCategory.VALIDATION: RecoveryStrategy(
                action=RecoveryAction.SKIP,
                max_retries=0,
                continue_on_failure=True
            ),
            ErrorCategory.DEPENDENCY: RecoveryStrategy(
                action=RecoveryAction.ABORT,
                max_retries=0
            )
        }
    
    def classify_error(self, exception: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Classify an error and create error context."""
        import traceback
        import uuid
        
        # Determine error category and severity
        category, severity = self._categorize_error(exception)
        
        error_context = ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=time.time(),
            stage_name=context.get('stage_name', 'unknown'),
            component_name=context.get('component_name'),
            experiment_id=context.get('experiment_id', 'unknown'),
            execution_id=context.get('execution_id', 'unknown'),
            error_message=str(exception),
            exception_type=type(exception).__name__,
            stack_trace=traceback.format_exc(),
            category=category,
            severity=severity,
            metadata=context.copy()
        )
        
        # Log the error
        self._log_error(error_context)
        
        # Store in history
        self.error_history.append(error_context)
        
        return error_context
    
    def _categorize_error(self, exception: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Categorize an error based on its type and message."""
        if isinstance(exception, PipelineError):
            return exception.category, exception.severity
        
        exception_type = type(exception).__name__
        error_message = str(exception).lower()
        
        # Configuration errors
        if any(keyword in error_message for keyword in ['config', 'parameter', 'setting']):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH
        
        # Data errors
        if any(keyword in error_message for keyword in ['data', 'file not found', 'schema', 'format']):
            return ErrorCategory.DATA, ErrorSeverity.MEDIUM
        
        # Model errors
        if any(keyword in error_message for keyword in ['model', 'training', 'prediction', 'inference']):
            return ErrorCategory.MODEL, ErrorSeverity.HIGH
        
        # Network errors
        if any(keyword in error_message for keyword in ['connection', 'timeout', 'network', 'http']):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        
        # Resource errors
        if any(keyword in error_message for keyword in ['memory', 'disk', 'space', 'resource']):
            return ErrorCategory.RESOURCE, ErrorSeverity.HIGH
        
        # System errors
        if exception_type in ['SystemError', 'OSError', 'PermissionError']:
            return ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL
        
        # Default to system error with medium severity
        return ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error to file and logger."""
        log_entry = {
            'error_id': error_context.error_id,
            'timestamp': error_context.timestamp,
            'stage_name': error_context.stage_name,
            'component_name': error_context.component_name,
            'experiment_id': error_context.experiment_id,
            'execution_id': error_context.execution_id,
            'error_message': error_context.error_message,
            'exception_type': error_context.exception_type,
            'category': error_context.category.value,
            'severity': error_context.severity.value,
            'retry_count': error_context.retry_count,
            'metadata': error_context.metadata
        }
        
        # Log to standard logger
        if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"Pipeline error [{error_context.error_id}]: {error_context.error_message}")
        else:
            self.logger.warning(f"Pipeline warning [{error_context.error_id}]: {error_context.error_message}")
        
        # Log to file if configured
        if self.error_log_path:
            try:
                with open(self.error_log_path, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                self.logger.error(f"Failed to write error log: {str(e)}")
    
    def determine_recovery_action(self, error_context: ErrorContext) -> RecoveryAction:
        """Determine the appropriate recovery action for an error."""
        strategy = self.recovery_strategies.get(error_context.category)
        if not strategy:
            return RecoveryAction.ABORT
        
        # Check if we've exceeded max retries
        if error_context.retry_count >= strategy.max_retries:
            if strategy.action == RecoveryAction.RETRY:
                return RecoveryAction.ABORT
            elif strategy.action == RecoveryAction.FALLBACK and not strategy.fallback_component:
                return RecoveryAction.SKIP if strategy.continue_on_failure else RecoveryAction.ABORT
        
        return strategy.action
    
    def calculate_retry_delay(self, error_context: ErrorContext) -> float:
        """Calculate delay before retry based on strategy and retry count."""
        strategy = self.recovery_strategies.get(error_context.category)
        if not strategy:
            return 1.0
        
        base_delay = strategy.retry_delay
        
        if strategy.exponential_backoff:
            return base_delay * (2 ** error_context.retry_count)
        else:
            return base_delay
    
    def should_continue_pipeline(self, error_context: ErrorContext) -> bool:
        """Determine if pipeline should continue after an error."""
        strategy = self.recovery_strategies.get(error_context.category)
        if not strategy:
            return False
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            return False
        
        return strategy.continue_on_failure
    
    def get_fallback_component(self, error_context: ErrorContext) -> Optional[str]:
        """Get fallback component name if available."""
        strategy = self.recovery_strategies.get(error_context.category)
        return strategy.fallback_component if strategy else None
    
    def register_recovery_strategy(self, category: ErrorCategory, strategy: RecoveryStrategy) -> None:
        """Register a custom recovery strategy for an error category."""
        self.recovery_strategies[category] = strategy
        self.logger.info(f"Registered recovery strategy for {category.value}: {strategy.action.value}")
    
    def register_error_handler(self, category: ErrorCategory, handler: Callable) -> None:
        """Register a custom error handler for an error category."""
        self.error_handlers[category] = handler
        self.logger.info(f"Registered custom error handler for {category.value}")
    
    def handle_error(self, error_context: ErrorContext) -> RecoveryAction:
        """Handle an error using registered handlers or default logic."""
        # Try custom handler first
        if error_context.category in self.error_handlers:
            try:
                custom_action = self.error_handlers[error_context.category](error_context)
                if custom_action:
                    return custom_action
            except Exception as e:
                self.logger.error(f"Custom error handler failed: {str(e)}")
        
        # Use default recovery logic
        return self.determine_recovery_action(error_context)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics from history."""
        if not self.error_history:
            return {"total_errors": 0}
        
        stats = {
            "total_errors": len(self.error_history),
            "by_category": {},
            "by_severity": {},
            "by_stage": {},
            "recent_errors": len([e for e in self.error_history if time.time() - e.timestamp < 3600])
        }
        
        for error in self.error_history:
            # Count by category
            category = error.category.value
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            
            # Count by stage
            stage = error.stage_name
            stats["by_stage"][stage] = stats["by_stage"].get(stage, 0) + 1
        
        return stats
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history.clear()
        self.logger.info("Error history cleared")
    
    def export_error_report(self, output_path: str) -> None:
        """Export detailed error report to file."""
        report = {
            "generated_at": time.time(),
            "statistics": self.get_error_statistics(),
            "errors": [
                {
                    "error_id": error.error_id,
                    "timestamp": error.timestamp,
                    "stage_name": error.stage_name,
                    "component_name": error.component_name,
                    "error_message": error.error_message,
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "retry_count": error.retry_count
                }
                for error in self.error_history
            ]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Error report exported to {output_path}")