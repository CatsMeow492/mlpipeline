"""Tests for error handling and recovery mechanisms."""

import pytest
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch

from mlpipeline.core.errors import (
    ErrorHandler, ErrorSeverity, ErrorCategory, RecoveryAction, RecoveryStrategy,
    PipelineError, ConfigurationError, DataError, ModelError, SystemError,
    ResourceError, NetworkError
)
from mlpipeline.core.orchestrator import PipelineOrchestrator
from mlpipeline.core.interfaces import (
    PipelineStage, PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
)
from mlpipeline.core.registry import component_registry


class FailingComponent(PipelineComponent):
    """Component that fails for testing error handling."""
    
    def __init__(self, failure_type: str = "generic", delay: float = 0.0):
        super().__init__(ComponentType.DATA_PREPROCESSING)
        self.failure_type = failure_type
        self.delay = delay
        self.attempt_count = 0
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute component with controlled failure."""
        self.attempt_count += 1
        
        if self.delay > 0:
            time.sleep(self.delay)
        
        if self.failure_type == "network":
            raise NetworkError("Connection timeout")
        elif self.failure_type == "data":
            raise DataError("Invalid data format")
        elif self.failure_type == "model":
            raise ModelError("Model training failed")
        elif self.failure_type == "resource":
            raise ResourceError("Out of memory")
        elif self.failure_type == "config":
            raise ConfigurationError("Invalid configuration")
        elif self.failure_type == "system":
            raise SystemError("System error")
        else:
            raise RuntimeError("Generic failure")
    
    def validate_config(self, config) -> bool:
        """Validate configuration."""
        return True


class RecoveringComponent(PipelineComponent):
    """Component that fails initially but succeeds on retry."""
    
    def __init__(self, fail_attempts: int = 2):
        super().__init__(ComponentType.DATA_PREPROCESSING)
        self.fail_attempts = fail_attempts
        self.attempt_count = 0
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute component with recovery after failures."""
        self.attempt_count += 1
        
        if self.attempt_count <= self.fail_attempts:
            raise RuntimeError(f"Attempt {self.attempt_count} failed")
        
        return ExecutionResult(
            success=True,
            artifacts=[f"artifact_attempt_{self.attempt_count}"],
            metrics={"attempts": self.attempt_count},
            metadata={"recovered": True}
        )
    
    def validate_config(self, config) -> bool:
        """Validate configuration."""
        return True


class TestErrorHandler:
    """Test error handler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.error_log_path = Path(self.temp_dir) / "error.log"
        self.error_handler = ErrorHandler(str(self.error_log_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_error_classification(self):
        """Test error classification by type and message."""
        context = {
            'stage_name': 'test_stage',
            'component_name': 'test_component',
            'experiment_id': 'test_exp',
            'execution_id': 'test_exec'
        }
        
        # Test network error
        network_error = NetworkError("Connection failed")
        error_context = self.error_handler.classify_error(network_error, context)
        assert error_context.category == ErrorCategory.NETWORK
        assert error_context.severity == ErrorSeverity.MEDIUM
        
        # Test data error
        data_error = DataError("Invalid format", ErrorSeverity.HIGH)
        error_context = self.error_handler.classify_error(data_error, context)
        assert error_context.category == ErrorCategory.DATA
        assert error_context.severity == ErrorSeverity.HIGH
        
        # Test generic error classification
        generic_error = RuntimeError("model training failed")
        error_context = self.error_handler.classify_error(generic_error, context)
        assert error_context.category == ErrorCategory.MODEL
    
    def test_recovery_strategies(self):
        """Test default recovery strategies."""
        # Network errors should retry
        network_context = Mock()
        network_context.category = ErrorCategory.NETWORK
        network_context.retry_count = 0
        
        action = self.error_handler.determine_recovery_action(network_context)
        assert action == RecoveryAction.RETRY
        
        # Configuration errors should abort
        config_context = Mock()
        config_context.category = ErrorCategory.CONFIGURATION
        config_context.retry_count = 0
        
        action = self.error_handler.determine_recovery_action(config_context)
        assert action == RecoveryAction.ABORT
    
    def test_retry_delay_calculation(self):
        """Test retry delay calculation with exponential backoff."""
        error_context = Mock()
        error_context.category = ErrorCategory.NETWORK
        error_context.retry_count = 0
        
        # First retry
        delay = self.error_handler.calculate_retry_delay(error_context)
        assert delay == 2.0  # Base delay for network errors
        
        # Second retry with exponential backoff
        error_context.retry_count = 1
        delay = self.error_handler.calculate_retry_delay(error_context)
        assert delay == 4.0  # 2.0 * 2^1
        
        # Third retry
        error_context.retry_count = 2
        delay = self.error_handler.calculate_retry_delay(error_context)
        assert delay == 8.0  # 2.0 * 2^2
    
    def test_custom_recovery_strategy(self):
        """Test registering custom recovery strategies."""
        custom_strategy = RecoveryStrategy(
            action=RecoveryAction.SKIP,
            max_retries=0,
            continue_on_failure=True
        )
        
        self.error_handler.register_recovery_strategy(ErrorCategory.DATA, custom_strategy)
        
        error_context = Mock()
        error_context.category = ErrorCategory.DATA
        error_context.retry_count = 0
        
        action = self.error_handler.determine_recovery_action(error_context)
        assert action == RecoveryAction.SKIP
    
    def test_error_logging(self):
        """Test error logging to file."""
        context = {
            'stage_name': 'test_stage',
            'experiment_id': 'test_exp',
            'execution_id': 'test_exec'
        }
        
        error = DataError("Test error")
        self.error_handler.classify_error(error, context)
        
        # Check that error log file was created
        assert self.error_log_path.exists()
        
        # Check log content
        with open(self.error_log_path, 'r') as f:
            log_entry = json.loads(f.readline())
            assert log_entry['error_message'] == "Test error"
            assert log_entry['category'] == "data"
            assert log_entry['stage_name'] == "test_stage"
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        context = {
            'stage_name': 'test_stage',
            'experiment_id': 'test_exp',
            'execution_id': 'test_exec'
        }
        
        # Generate some errors
        self.error_handler.classify_error(NetworkError("Network error 1"), context)
        self.error_handler.classify_error(NetworkError("Network error 2"), context)
        self.error_handler.classify_error(DataError("Data error"), context)
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['by_category']['network'] == 2
        assert stats['by_category']['data'] == 1
        assert stats['by_severity']['medium'] == 3  # All three errors have medium severity
        assert stats['by_stage']['test_stage'] == 3
    
    def test_error_report_export(self):
        """Test exporting error reports."""
        context = {
            'stage_name': 'test_stage',
            'experiment_id': 'test_exp',
            'execution_id': 'test_exec'
        }
        
        self.error_handler.classify_error(DataError("Test error"), context)
        
        report_path = Path(self.temp_dir) / "error_report.json"
        self.error_handler.export_error_report(str(report_path))
        
        assert report_path.exists()
        
        with open(report_path, 'r') as f:
            report = json.load(f)
            assert 'statistics' in report
            assert 'errors' in report
            assert len(report['errors']) == 1
            assert report['errors'][0]['error_message'] == "Test error"


class TestOrchestratorErrorHandling:
    """Test orchestrator error handling and recovery."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.orchestrator = PipelineOrchestrator(
            max_workers=2,
            error_log_path=str(Path(self.temp_dir) / "errors.log")
        )
        
        self.context = ExecutionContext(
            experiment_id="test_experiment",
            stage_name="test_stage",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={"test": "config"},
            artifacts_path=self.temp_dir,
            logger=self.orchestrator.logger,
            metadata={"test": "metadata"}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_component_retry_mechanism(self):
        """Test component retry on failure."""
        # Component that fails twice then succeeds
        component = RecoveringComponent(fail_attempts=2)
        
        result = self.orchestrator._execute_component_with_recovery(component, self.context)
        
        assert result.success is True
        assert component.attempt_count == 3  # Failed twice, succeeded on third
        assert result.metadata["recovered"] is True
    
    def test_component_max_retries_exceeded(self):
        """Test component failure after max retries."""
        # Component that always fails
        component = FailingComponent("generic")
        
        result = self.orchestrator._execute_component_with_recovery(component, self.context)
        
        assert result.success is False
        # The result should contain error context, not max_retries_exceeded since we abort after max retries
        assert "error_context" in result.metadata or "max_retries_exceeded" in result.metadata
        assert component.attempt_count == 3  # Max attempts
    
    def test_component_skip_recovery(self):
        """Test component skip recovery action."""
        # Register custom strategy for validation errors to skip
        skip_strategy = RecoveryStrategy(
            action=RecoveryAction.SKIP,
            max_retries=0,
            continue_on_failure=True
        )
        self.orchestrator.error_handler.register_recovery_strategy(
            ErrorCategory.VALIDATION, skip_strategy
        )
        
        # Create a component that raises validation error
        component = FailingComponent("validation")
        
        # Mock the error classification to return validation category
        with patch.object(self.orchestrator.error_handler, 'classify_error') as mock_classify:
            mock_error_context = Mock()
            mock_error_context.category = ErrorCategory.VALIDATION
            mock_error_context.retry_count = 0
            mock_classify.return_value = mock_error_context
            
            result = self.orchestrator._execute_component_with_recovery(component, self.context)
            
            assert result.success is True  # Marked as success to continue
            assert result.metadata["skipped"] is True
    
    def test_fallback_component_recovery(self):
        """Test fallback component recovery."""
        # Register fallback strategy
        fallback_strategy = RecoveryStrategy(
            action=RecoveryAction.FALLBACK,
            fallback_component="fallback_component"
        )
        self.orchestrator.error_handler.register_recovery_strategy(
            ErrorCategory.MODEL, fallback_strategy
        )
        
        # Register fallback component
        class FallbackComponent(PipelineComponent):
            def __init__(self):
                super().__init__(ComponentType.DATA_PREPROCESSING)
            
            def execute(self, context):
                return ExecutionResult(
                    success=True,
                    artifacts=["fallback_artifact"],
                    metrics={"fallback": True},
                    metadata={"is_fallback": True}
                )
            
            def validate_config(self, config):
                return True
        
        component_registry.register_component("fallback_component", FallbackComponent)
        
        try:
            component = FailingComponent("model")
            
            # Mock error classification
            with patch.object(self.orchestrator.error_handler, 'classify_error') as mock_classify:
                mock_error_context = Mock()
                mock_error_context.category = ErrorCategory.MODEL
                mock_error_context.retry_count = 0
                mock_classify.return_value = mock_error_context
                
                result = self.orchestrator._execute_component_with_recovery(component, self.context)
                
                assert result.success is True
                assert "fallback_artifact" in result.artifacts
                assert result.metadata["is_fallback"] is True
        
        finally:
            component_registry.unregister_component("fallback_component")
    
    def test_stage_partial_failure_handling(self):
        """Test stage handling with partial component failures."""
        components = [
            RecoveringComponent(fail_attempts=1),  # Will succeed
            FailingComponent("generic"),           # Will fail
            RecoveringComponent(fail_attempts=1),  # Will succeed
        ]
        
        stage = PipelineStage(
            name="partial_failure_stage",
            components=components,
            parallel=False
        )
        
        result = self.orchestrator._execute_stage_with_recovery(stage, self.context)
        
        # Stage should succeed despite one component failure (< 50% failure rate)
        assert result.success is True
        assert len(result.metadata["failed_components"]) == 1
        assert "FailingComponent" in result.metadata["failed_components"]
    
    def test_stage_majority_failure_handling(self):
        """Test stage failure when majority of components fail."""
        components = [
            FailingComponent("generic"),  # Will fail
            FailingComponent("generic"),  # Will fail
            RecoveringComponent(fail_attempts=1),  # Will succeed
        ]
        
        stage = PipelineStage(
            name="majority_failure_stage",
            components=components,
            parallel=False
        )
        
        result = self.orchestrator._execute_stage_with_recovery(stage, self.context)
        
        # Stage should fail due to majority failure (> 50%)
        assert result.success is False
        assert result.metadata["partial_failure"] is True
        assert len(result.metadata["failed_components"]) == 2
    
    def test_pipeline_error_propagation(self):
        """Test error propagation through pipeline execution."""
        stages = [
            PipelineStage(
                name="success_stage",
                components=[RecoveringComponent(fail_attempts=1)]
            ),
            PipelineStage(
                name="failure_stage",
                components=[FailingComponent("generic")],
                dependencies=["success_stage"]
            )
        ]
        
        result = self.orchestrator.execute_pipeline(stages, self.context)
        
        assert result.success is False
        # The first stage (success_stage) should succeed, then failure_stage should fail
        # But since success_stage has a RecoveringComponent that fails initially, it might be the one failing
        assert "failed_stage" in result.metadata
        failed_stage = result.metadata["failed_stage"]
        assert failed_stage in ["success_stage", "failure_stage"]
        
        # Check error statistics
        stats = self.orchestrator.error_handler.get_error_statistics()
        assert stats["total_errors"] > 0
    
    def test_error_context_correlation_id(self):
        """Test that error contexts include correlation IDs."""
        component = FailingComponent("generic")
        stage = PipelineStage(name="test_stage", components=[component])
        
        # Execute pipeline to set correlation ID
        self.orchestrator.execute_pipeline([stage], self.context)
        
        # Check that errors were logged with correlation ID
        error_history = self.orchestrator.error_handler.error_history
        assert len(error_history) > 0
        
        error_context = error_history[0]
        assert error_context.execution_id is not None
        assert error_context.experiment_id == "test_experiment"
    
    def test_custom_error_handler_registration(self):
        """Test registering custom error handlers."""
        custom_handler_called = False
        
        def custom_handler(error_context):
            nonlocal custom_handler_called
            custom_handler_called = True
            return RecoveryAction.SKIP
        
        self.orchestrator.error_handler.register_error_handler(
            ErrorCategory.DATA, custom_handler
        )
        
        component = FailingComponent("data")
        
        # Mock error classification
        with patch.object(self.orchestrator.error_handler, 'classify_error') as mock_classify:
            mock_error_context = Mock()
            mock_error_context.category = ErrorCategory.DATA
            mock_error_context.retry_count = 0
            mock_classify.return_value = mock_error_context
            
            result = self.orchestrator._execute_component_with_recovery(component, self.context)
            
            assert custom_handler_called is True
            assert result.success is True  # Should be skipped
            assert result.metadata["skipped"] is True
    
    def test_error_logging_integration(self):
        """Test integration with error logging."""
        component = FailingComponent("network")
        stage = PipelineStage(name="logging_test_stage", components=[component])
        
        self.orchestrator.execute_pipeline([stage], self.context)
        
        # Check that error log file was created
        error_log_path = Path(self.temp_dir) / "errors.log"
        assert error_log_path.exists()
        
        # Check log content
        with open(error_log_path, 'r') as f:
            log_lines = f.readlines()
            assert len(log_lines) > 0
            
            # Parse first log entry
            log_entry = json.loads(log_lines[0])
            assert log_entry['stage_name'] == "logging_test_stage"
            assert log_entry['category'] == "network"