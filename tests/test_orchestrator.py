"""Tests for pipeline orchestrator functionality."""

import pytest
import tempfile
import threading
import time
from unittest.mock import Mock, patch
from pathlib import Path

from mlpipeline.core.orchestrator import PipelineOrchestrator
from mlpipeline.core.interfaces import (
    PipelineStage, PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
)
from mlpipeline.core.registry import component_registry


class MockComponent(PipelineComponent):
    """Mock component for testing."""
    
    def __init__(self, name: str = "mock", delay: float = 0.0, should_fail: bool = False):
        super().__init__(ComponentType.DATA_PREPROCESSING)
        self.name = name
        self.delay = delay
        self.should_fail = should_fail
        self.setup_called = False
        self.cleanup_called = False
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute mock component."""
        if self.delay > 0:
            time.sleep(self.delay)
        
        if self.should_fail:
            raise RuntimeError(f"Mock component {self.name} failed")
        
        return ExecutionResult(
            success=True,
            artifacts=[f"artifact_{self.name}"],
            metrics={f"metric_{self.name}": 1.0},
            metadata={"component": self.name}
        )
    
    def validate_config(self, config) -> bool:
        """Validate configuration."""
        return True
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup component."""
        self.setup_called = True
    
    def cleanup(self, context: ExecutionContext) -> None:
        """Cleanup component."""
        self.cleanup_called = True


class TestPipelineOrchestrator:
    """Test pipeline orchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = PipelineOrchestrator(max_workers=2)
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock execution context
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
        self.orchestrator.clear_execution_history()
    
    def test_structured_logger_setup(self):
        """Test structured logger with correlation ID."""
        # Test that logger is properly configured
        assert self.orchestrator.logger is not None
        assert len(self.orchestrator.logger.handlers) > 0
        
        # Test correlation ID in logging
        threading.current_thread().correlation_id = "test-correlation-id"
        
        with patch('logging.StreamHandler.emit') as mock_emit:
            self.orchestrator.logger.info("Test message")
            # Verify that emit was called (logger is working)
            assert mock_emit.called
    
    def test_execute_single_stage(self):
        """Test executing a single stage with one component."""
        component = MockComponent("test_component")
        stage = PipelineStage(
            name="test_stage",
            components=[component]
        )
        
        result = self.orchestrator.execute_stage(stage, self.context)
        
        assert result.success is True
        assert len(result.artifacts) == 1
        assert "artifact_test_component" in result.artifacts
        assert "metric_test_component" in result.metrics
        assert component.setup_called is True
        assert component.cleanup_called is True
    
    def test_execute_stage_with_failure(self):
        """Test stage execution with component failure."""
        component = MockComponent("failing_component", should_fail=True)
        stage = PipelineStage(
            name="failing_stage",
            components=[component]
        )
        
        result = self.orchestrator.execute_stage(stage, self.context)
        
        assert result.success is False
        # With enhanced error handling, the error message might be different
        assert "failed" in result.error_message.lower()
    
    def test_execute_stage_parallel(self):
        """Test parallel execution of components in a stage."""
        components = [
            MockComponent("component_1", delay=0.1),
            MockComponent("component_2", delay=0.1),
            MockComponent("component_3", delay=0.1)
        ]
        
        stage = PipelineStage(
            name="parallel_stage",
            components=components,
            parallel=True
        )
        
        start_time = time.time()
        result = self.orchestrator.execute_stage(stage, self.context)
        execution_time = time.time() - start_time
        
        assert result.success is True
        assert len(result.artifacts) == 3
        # Parallel execution should be faster than sequential
        assert execution_time < 0.25  # Should be much less than 0.3 seconds
        
        # Verify all components were executed
        for component in components:
            assert component.setup_called is True
            assert component.cleanup_called is True
    
    def test_execute_stage_sequential(self):
        """Test sequential execution of components in a stage."""
        components = [
            MockComponent("component_1", delay=0.05),
            MockComponent("component_2", delay=0.05)
        ]
        
        stage = PipelineStage(
            name="sequential_stage",
            components=components,
            parallel=False
        )
        
        start_time = time.time()
        result = self.orchestrator.execute_stage(stage, self.context)
        execution_time = time.time() - start_time
        
        assert result.success is True
        assert len(result.artifacts) == 2
        # Sequential execution should take longer
        assert execution_time >= 0.08
    
    def test_execute_pipeline_simple(self):
        """Test executing a simple pipeline with no dependencies."""
        stages = [
            PipelineStage(
                name="stage_1",
                components=[MockComponent("comp_1")]
            ),
            PipelineStage(
                name="stage_2",
                components=[MockComponent("comp_2")]
            )
        ]
        
        result = self.orchestrator.execute_pipeline(stages, self.context)
        
        assert result.success is True
        assert len(result.artifacts) == 2
        assert len(result.metrics) == 2
        assert "executed_stages" in result.metadata
        assert len(result.metadata["executed_stages"]) == 2
    
    def test_execute_pipeline_with_dependencies(self):
        """Test executing pipeline with stage dependencies."""
        stages = [
            PipelineStage(
                name="stage_1",
                components=[MockComponent("comp_1")]
            ),
            PipelineStage(
                name="stage_2",
                components=[MockComponent("comp_2")],
                dependencies=["stage_1"]
            ),
            PipelineStage(
                name="stage_3",
                components=[MockComponent("comp_3")],
                dependencies=["stage_1", "stage_2"]
            )
        ]
        
        result = self.orchestrator.execute_pipeline(stages, self.context)
        
        assert result.success is True
        assert len(result.artifacts) == 3
        
        # Check execution history
        history = self.orchestrator.get_execution_history()
        assert len(history) == 1
        assert history[0]["experiment_id"] == "test_experiment"
    
    def test_pipeline_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        stages = [
            PipelineStage(
                name="stage_1",
                components=[MockComponent("comp_1")],
                dependencies=["stage_2"]
            ),
            PipelineStage(
                name="stage_2",
                components=[MockComponent("comp_2")],
                dependencies=["stage_1"]
            )
        ]
        
        result = self.orchestrator.execute_pipeline(stages, self.context)
        
        assert result.success is False
        assert "Circular dependency" in result.error_message
    
    def test_validate_pipeline_stages(self):
        """Test pipeline stage validation."""
        stages = [
            PipelineStage(
                name="stage_1",
                components=[MockComponent("comp_1")]
            ),
            PipelineStage(
                name="stage_1",  # Duplicate name
                components=[MockComponent("comp_2")]
            ),
            PipelineStage(
                name="stage_2",
                components=[MockComponent("comp_3")],
                dependencies=["nonexistent_stage"]  # Invalid dependency
            )
        ]
        
        errors = self.orchestrator.validate_pipeline_stages(stages)
        
        assert len(errors) == 2
        assert any("Duplicate stage name" in error for error in errors)
        assert any("depends on non-existent stage" in error for error in errors)
    
    def test_get_stage_execution_order(self):
        """Test getting stage execution order."""
        stages = [
            PipelineStage(name="stage_3", components=[], dependencies=["stage_1", "stage_2"]),
            PipelineStage(name="stage_1", components=[]),
            PipelineStage(name="stage_2", components=[], dependencies=["stage_1"]),
        ]
        
        execution_order = self.orchestrator.get_stage_execution_order(stages)
        
        assert len(execution_order) == 3
        assert execution_order[0] == ["stage_1"]  # First batch
        assert execution_order[1] == ["stage_2"]  # Second batch
        assert execution_order[2] == ["stage_3"]  # Third batch
    
    def test_execution_history_management(self):
        """Test execution history tracking."""
        # Clear any existing history
        self.orchestrator.clear_execution_history()
        
        stages = [PipelineStage(name="test_stage", components=[MockComponent("comp")])]
        
        # Execute pipeline
        self.orchestrator.execute_pipeline(stages, self.context)
        
        # Check history
        history = self.orchestrator.get_execution_history()
        assert len(history) == 1
        assert history[0]["experiment_id"] == "test_experiment"
        assert "execution_id" in history[0]
        
        # Test current execution ID
        current_id = self.orchestrator.get_current_execution_id()
        assert current_id is not None
        
        # Clear history
        self.orchestrator.clear_execution_history()
        history = self.orchestrator.get_execution_history()
        assert len(history) == 0
    
    def test_checkpointing(self):
        """Test checkpoint save and load functionality."""
        execution_id = "test_execution_123"
        checkpoint_data = {
            "stage": "test_stage",
            "progress": 0.5,
            "artifacts": ["artifact1", "artifact2"]
        }
        
        checkpoint_path = Path(self.temp_dir) / "test_checkpoint.json"
        
        # Save checkpoint
        self.orchestrator.save_checkpoint(
            execution_id, 
            checkpoint_data, 
            str(checkpoint_path)
        )
        
        # Verify file exists
        assert checkpoint_path.exists()
        
        # Load checkpoint
        loaded_data = self.orchestrator.load_checkpoint(
            execution_id, 
            str(checkpoint_path)
        )
        
        assert loaded_data is not None
        assert loaded_data["stage"] == "test_stage"
        assert loaded_data["progress"] == 0.5
        assert len(loaded_data["artifacts"]) == 2
    
    def test_checkpointing_disabled(self):
        """Test checkpointing when disabled."""
        orchestrator = PipelineOrchestrator(enable_checkpointing=False)
        
        execution_id = "test_execution"
        checkpoint_data = {"test": "data"}
        
        # Save should do nothing
        orchestrator.save_checkpoint(execution_id, checkpoint_data)
        
        # Load should return None
        result = orchestrator.load_checkpoint(execution_id)
        assert result is None
    
    def test_create_stage_from_config(self):
        """Test creating pipeline stage from configuration."""
        # Register a mock component
        component_registry.register_component("mock_component", MockComponent)
        
        stage_config = {
            "components": [
                {
                    "name": "mock_component",
                    "type": "data_preprocessing",
                    "parameters": {
                        "name": "configured_component"
                    }
                }
            ],
            "dependencies": ["previous_stage"],
            "parallel": True
        }
        
        try:
            stage = self.orchestrator.create_stage_from_config("test_stage", stage_config)
            
            assert stage.name == "test_stage"
            assert len(stage.components) == 1
            assert stage.components[0].name == "configured_component"
            assert stage.dependencies == ["previous_stage"]
            assert stage.parallel is True
            
        finally:
            # Clean up registry
            component_registry.unregister_component("mock_component")
    
    def test_create_stage_from_config_missing_component(self):
        """Test creating stage with missing component in registry."""
        stage_config = {
            "components": [
                {
                    "name": "nonexistent_component",
                    "type": "data_preprocessing",
                    "parameters": {}
                }
            ]
        }
        
        with pytest.raises(ValueError, match="Component nonexistent_component not found in registry"):
            self.orchestrator.create_stage_from_config("test_stage", stage_config)
    
    def test_pipeline_failure_handling(self):
        """Test pipeline behavior when a stage fails."""
        stages = [
            PipelineStage(
                name="success_stage",
                components=[MockComponent("success_comp")]
            ),
            PipelineStage(
                name="failure_stage",
                components=[MockComponent("failure_comp", should_fail=True)],
                dependencies=["success_stage"]
            ),
            PipelineStage(
                name="never_executed_stage",
                components=[MockComponent("never_comp")],
                dependencies=["failure_stage"]
            )
        ]
        
        result = self.orchestrator.execute_pipeline(stages, self.context)
        
        assert result.success is False
        assert "failure_stage" in result.error_message
        assert result.metadata["failed_stage"] == "failure_stage"
        
        # Only one artifact should be present (from success_stage)
        assert len(result.artifacts) == 1
        assert "artifact_success_comp" in result.artifacts
    
    def test_correlation_id_propagation(self):
        """Test that correlation IDs are properly set during execution."""
        stages = [PipelineStage(name="test_stage", components=[MockComponent("comp")])]
        
        # Execute pipeline
        result = self.orchestrator.execute_pipeline(stages, self.context)
        
        # Verify execution was successful
        assert result.success is True
        
        # Verify current execution ID is set
        execution_id = self.orchestrator.get_current_execution_id()
        assert execution_id is not None
        
        # Verify execution history contains the execution ID
        history = self.orchestrator.get_execution_history()
        assert len(history) == 1
        assert history[0]["execution_id"] == execution_id