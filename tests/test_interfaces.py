"""Tests for core interfaces and abstract base classes."""

import pytest
import logging
from unittest.mock import Mock
from dataclasses import FrozenInstanceError

from mlpipeline.core.interfaces import (
    PipelineStatus, ComponentType, ExecutionContext, ExecutionResult,
    PipelineComponent, PipelineStage, PipelineOrchestrator
)


class TestEnums:
    """Test enum classes."""
    
    def test_pipeline_status_values(self):
        """Test PipelineStatus enum values."""
        assert PipelineStatus.NOT_STARTED.value == "not_started"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.FAILED.value == "failed"
        assert PipelineStatus.CANCELLED.value == "cancelled"
    
    def test_component_type_values(self):
        """Test ComponentType enum values."""
        assert ComponentType.DATA_INGESTION.value == "data_ingestion"
        assert ComponentType.DATA_PREPROCESSING.value == "data_preprocessing"
        assert ComponentType.DATA_VALIDATION.value == "data_validation"
        assert ComponentType.MODEL_TRAINING.value == "model_training"
        assert ComponentType.MODEL_EVALUATION.value == "model_evaluation"
        assert ComponentType.MODEL_INFERENCE.value == "model_inference"
        assert ComponentType.DRIFT_DETECTION.value == "drift_detection"
        assert ComponentType.FEW_SHOT_LEARNING.value == "few_shot_learning"
    
    def test_enum_membership(self):
        """Test enum membership checks."""
        assert PipelineStatus.RUNNING in PipelineStatus
        assert ComponentType.MODEL_TRAINING in ComponentType
        
        # Test string comparison
        assert PipelineStatus.COMPLETED.value == "completed"
        assert ComponentType.DATA_PREPROCESSING.value == "data_preprocessing"


class TestExecutionContext:
    """Test ExecutionContext dataclass."""
    
    def test_execution_context_creation(self):
        """Test creating ExecutionContext with all fields."""
        logger = logging.getLogger("test")
        context = ExecutionContext(
            experiment_id="exp_123",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={"param": "value"},
            artifacts_path="/path/to/artifacts",
            logger=logger,
            metadata={"key": "value"}
        )
        
        assert context.experiment_id == "exp_123"
        assert context.stage_name == "preprocessing"
        assert context.component_type == ComponentType.DATA_PREPROCESSING
        assert context.config == {"param": "value"}
        assert context.artifacts_path == "/path/to/artifacts"
        assert context.logger == logger
        assert context.metadata == {"key": "value"}
    
    def test_execution_context_default_metadata(self):
        """Test ExecutionContext with default metadata."""
        logger = logging.getLogger("test")
        context = ExecutionContext(
            experiment_id="exp_123",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={},
            artifacts_path="/path",
            logger=logger
        )
        
        assert context.metadata == {}
    
    def test_execution_context_metadata_post_init(self):
        """Test ExecutionContext metadata initialization in __post_init__."""
        logger = logging.getLogger("test")
        context = ExecutionContext(
            experiment_id="exp_123",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={},
            artifacts_path="/path",
            logger=logger,
            metadata=None
        )
        
        assert context.metadata == {}
    
    def test_execution_context_mutability(self):
        """Test that ExecutionContext fields can be modified."""
        logger = logging.getLogger("test")
        context = ExecutionContext(
            experiment_id="exp_123",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={},
            artifacts_path="/path",
            logger=logger
        )
        
        # Should be able to modify fields since dataclass is not frozen
        context.experiment_id = "new_exp"
        context.stage_name = "new_stage"
        
        assert context.experiment_id == "new_exp"
        assert context.stage_name == "new_stage"
    
    def test_execution_context_mutable_collections(self):
        """Test that mutable collections in ExecutionContext can be modified."""
        logger = logging.getLogger("test")
        context = ExecutionContext(
            experiment_id="exp_123",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={"list": [1, 2, 3]},
            artifacts_path="/path",
            logger=logger,
            metadata={"dict": {"nested": "value"}}
        )
        
        # Should be able to modify mutable collections
        context.config["list"].append(4)
        context.metadata["dict"]["new_key"] = "new_value"
        
        assert context.config["list"] == [1, 2, 3, 4]
        assert context.metadata["dict"]["new_key"] == "new_value"


class TestExecutionResult:
    """Test ExecutionResult dataclass."""
    
    def test_execution_result_creation(self):
        """Test creating ExecutionResult with all fields."""
        result = ExecutionResult(
            success=True,
            artifacts=["artifact1.pkl", "artifact2.json"],
            metrics={"accuracy": 0.95, "loss": 0.05},
            metadata={"model_type": "random_forest"},
            error_message=None,
            execution_time=120.5
        )
        
        assert result.success is True
        assert result.artifacts == ["artifact1.pkl", "artifact2.json"]
        assert result.metrics == {"accuracy": 0.95, "loss": 0.05}
        assert result.metadata == {"model_type": "random_forest"}
        assert result.error_message is None
        assert result.execution_time == 120.5
    
    def test_execution_result_failure(self):
        """Test creating ExecutionResult for failure case."""
        result = ExecutionResult(
            success=False,
            artifacts=[],
            metrics={},
            metadata={},
            error_message="Component execution failed",
            execution_time=30.0
        )
        
        assert result.success is False
        assert result.artifacts == []
        assert result.metrics == {}
        assert result.metadata == {}
        assert result.error_message == "Component execution failed"
        assert result.execution_time == 30.0
    
    def test_execution_result_optional_fields(self):
        """Test ExecutionResult with optional fields as None."""
        result = ExecutionResult(
            success=True,
            artifacts=["artifact.pkl"],
            metrics={"score": 0.8},
            metadata={"info": "test"}
        )
        
        assert result.error_message is None
        assert result.execution_time is None
    
    def test_execution_result_mutability(self):
        """Test that ExecutionResult fields can be modified."""
        result = ExecutionResult(
            success=True,
            artifacts=[],
            metrics={},
            metadata={}
        )
        
        # Should be able to modify fields since dataclass is not frozen
        result.success = False
        result.error_message = "New error"
        
        assert result.success is False
        assert result.error_message == "New error"


class TestPipelineComponent:
    """Test PipelineComponent abstract base class."""
    
    def test_pipeline_component_abstract(self):
        """Test that PipelineComponent cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PipelineComponent(ComponentType.DATA_PREPROCESSING)
    
    def test_concrete_component_implementation(self):
        """Test concrete implementation of PipelineComponent."""
        class ConcreteComponent(PipelineComponent):
            def execute(self, context):
                return ExecutionResult(
                    success=True,
                    artifacts=[],
                    metrics={},
                    metadata={}
                )
            
            def validate_config(self, config):
                return True
        
        component = ConcreteComponent(ComponentType.MODEL_TRAINING)
        
        assert component.component_type == ComponentType.MODEL_TRAINING
        assert isinstance(component.logger, logging.Logger)
        assert component.logger.name == "ConcreteComponent"
    
    def test_component_execute_method(self):
        """Test component execute method."""
        class TestComponent(PipelineComponent):
            def execute(self, context):
                return ExecutionResult(
                    success=True,
                    artifacts=["test_artifact"],
                    metrics={"test_metric": 1.0},
                    metadata={"component_name": "TestComponent"}
                )
            
            def validate_config(self, config):
                return config.get("valid", True)
        
        component = TestComponent(ComponentType.DATA_PREPROCESSING)
        context = Mock()
        
        result = component.execute(context)
        
        assert result.success is True
        assert result.artifacts == ["test_artifact"]
        assert result.metrics == {"test_metric": 1.0}
        assert result.metadata == {"component_name": "TestComponent"}
    
    def test_component_validate_config_method(self):
        """Test component validate_config method."""
        class TestComponent(PipelineComponent):
            def execute(self, context):
                return ExecutionResult(success=True, artifacts=[], metrics={}, metadata={})
            
            def validate_config(self, config):
                required_keys = ["input_path", "output_path"]
                return all(key in config for key in required_keys)
        
        component = TestComponent(ComponentType.DATA_PREPROCESSING)
        
        # Valid config
        valid_config = {"input_path": "/input", "output_path": "/output"}
        assert component.validate_config(valid_config) is True
        
        # Invalid config
        invalid_config = {"input_path": "/input"}
        assert component.validate_config(invalid_config) is False
    
    def test_component_setup_method(self):
        """Test component setup method (optional override)."""
        class TestComponent(PipelineComponent):
            def __init__(self, component_type):
                super().__init__(component_type)
                self.setup_called = False
            
            def execute(self, context):
                return ExecutionResult(success=True, artifacts=[], metrics={}, metadata={})
            
            def validate_config(self, config):
                return True
            
            def setup(self, context):
                self.setup_called = True
        
        component = TestComponent(ComponentType.DATA_PREPROCESSING)
        context = Mock()
        
        # Setup should not be called initially
        assert component.setup_called is False
        
        # Call setup
        component.setup(context)
        assert component.setup_called is True
    
    def test_component_cleanup_method(self):
        """Test component cleanup method (optional override)."""
        class TestComponent(PipelineComponent):
            def __init__(self, component_type):
                super().__init__(component_type)
                self.cleanup_called = False
            
            def execute(self, context):
                return ExecutionResult(success=True, artifacts=[], metrics={}, metadata={})
            
            def validate_config(self, config):
                return True
            
            def cleanup(self, context):
                self.cleanup_called = True
        
        component = TestComponent(ComponentType.DATA_PREPROCESSING)
        context = Mock()
        
        # Cleanup should not be called initially
        assert component.cleanup_called is False
        
        # Call cleanup
        component.cleanup(context)
        assert component.cleanup_called is True
    
    def test_component_default_setup_cleanup(self):
        """Test component default setup and cleanup methods."""
        class MinimalComponent(PipelineComponent):
            def execute(self, context):
                return ExecutionResult(success=True, artifacts=[], metrics={}, metadata={})
            
            def validate_config(self, config):
                return True
        
        component = MinimalComponent(ComponentType.DATA_PREPROCESSING)
        context = Mock()
        
        # Default setup and cleanup should not raise exceptions
        component.setup(context)
        component.cleanup(context)
    
    def test_component_logger_configuration(self):
        """Test component logger configuration."""
        class CustomNameComponent(PipelineComponent):
            def execute(self, context):
                return ExecutionResult(success=True, artifacts=[], metrics={}, metadata={})
            
            def validate_config(self, config):
                return True
        
        component = CustomNameComponent(ComponentType.MODEL_EVALUATION)
        
        assert component.logger.name == "CustomNameComponent"
        assert isinstance(component.logger, logging.Logger)


class TestPipelineStage:
    """Test PipelineStage dataclass."""
    
    def test_pipeline_stage_creation(self):
        """Test creating PipelineStage with all fields."""
        mock_component1 = Mock(spec=PipelineComponent)
        mock_component2 = Mock(spec=PipelineComponent)
        
        stage = PipelineStage(
            name="test_stage",
            components=[mock_component1, mock_component2],
            dependencies=["previous_stage"],
            parallel=True
        )
        
        assert stage.name == "test_stage"
        assert stage.components == [mock_component1, mock_component2]
        assert stage.dependencies == ["previous_stage"]
        assert stage.parallel is True
    
    def test_pipeline_stage_default_dependencies(self):
        """Test PipelineStage with default dependencies."""
        mock_component = Mock(spec=PipelineComponent)
        
        stage = PipelineStage(
            name="test_stage",
            components=[mock_component]
        )
        
        assert stage.dependencies == []
        assert stage.parallel is False
    
    def test_pipeline_stage_dependencies_post_init(self):
        """Test PipelineStage dependencies initialization in __post_init__."""
        mock_component = Mock(spec=PipelineComponent)
        
        stage = PipelineStage(
            name="test_stage",
            components=[mock_component],
            dependencies=None
        )
        
        assert stage.dependencies == []
    
    def test_pipeline_stage_empty_components(self):
        """Test PipelineStage with empty components list."""
        stage = PipelineStage(
            name="empty_stage",
            components=[]
        )
        
        assert stage.name == "empty_stage"
        assert stage.components == []
        assert stage.dependencies == []
        assert stage.parallel is False
    
    def test_pipeline_stage_single_component(self):
        """Test PipelineStage with single component."""
        mock_component = Mock(spec=PipelineComponent)
        
        stage = PipelineStage(
            name="single_stage",
            components=[mock_component],
            dependencies=["dep1", "dep2"],
            parallel=False
        )
        
        assert len(stage.components) == 1
        assert stage.components[0] == mock_component
        assert stage.dependencies == ["dep1", "dep2"]
        assert stage.parallel is False
    
    def test_pipeline_stage_mutability(self):
        """Test that PipelineStage fields can be modified."""
        mock_component = Mock(spec=PipelineComponent)
        
        stage = PipelineStage(
            name="test_stage",
            components=[mock_component]
        )
        
        # Should be able to modify fields since dataclass is not frozen
        stage.name = "new_name"
        stage.parallel = True
        
        assert stage.name == "new_name"
        assert stage.parallel is True


class TestPipelineOrchestrator:
    """Test PipelineOrchestrator abstract base class."""
    
    def test_pipeline_orchestrator_abstract(self):
        """Test that PipelineOrchestrator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PipelineOrchestrator()
    
    def test_concrete_orchestrator_implementation(self):
        """Test concrete implementation of PipelineOrchestrator."""
        class ConcreteOrchestrator(PipelineOrchestrator):
            def execute_pipeline(self, stages, context):
                return ExecutionResult(
                    success=True,
                    artifacts=[],
                    metrics={},
                    metadata={"stages_executed": len(stages)}
                )
            
            def execute_stage(self, stage, context):
                return ExecutionResult(
                    success=True,
                    artifacts=[],
                    metrics={},
                    metadata={"stage_name": stage.name}
                )
        
        orchestrator = ConcreteOrchestrator()
        
        # Test execute_pipeline
        mock_stages = [Mock(), Mock()]
        mock_context = Mock()
        
        result = orchestrator.execute_pipeline(mock_stages, mock_context)
        assert result.success is True
        assert result.metadata["stages_executed"] == 2
        
        # Test execute_stage
        mock_stage = Mock()
        mock_stage.name = "test_stage"
        
        result = orchestrator.execute_stage(mock_stage, mock_context)
        assert result.success is True
        assert result.metadata["stage_name"] == "test_stage"
    
    def test_orchestrator_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        class IncompleteOrchestrator(PipelineOrchestrator):
            def execute_pipeline(self, stages, context):
                pass
            # Missing execute_stage implementation
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteOrchestrator()
        
        class AnotherIncompleteOrchestrator(PipelineOrchestrator):
            def execute_stage(self, stage, context):
                pass
            # Missing execute_pipeline implementation
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AnotherIncompleteOrchestrator()


class TestInterfaceIntegration:
    """Test integration between different interface components."""
    
    def test_component_in_stage(self):
        """Test using components within stages."""
        class TestComponent(PipelineComponent):
            def __init__(self, name):
                super().__init__(ComponentType.DATA_PREPROCESSING)
                self.name = name
            
            def execute(self, context):
                return ExecutionResult(
                    success=True,
                    artifacts=[f"{self.name}_artifact"],
                    metrics={f"{self.name}_metric": 1.0},
                    metadata={"component": self.name}
                )
            
            def validate_config(self, config):
                return True
        
        component1 = TestComponent("comp1")
        component2 = TestComponent("comp2")
        
        stage = PipelineStage(
            name="integration_stage",
            components=[component1, component2],
            dependencies=["previous_stage"],
            parallel=True
        )
        
        assert len(stage.components) == 2
        assert stage.components[0].name == "comp1"
        assert stage.components[1].name == "comp2"
        assert stage.dependencies == ["previous_stage"]
        assert stage.parallel is True
    
    def test_execution_context_with_component(self):
        """Test ExecutionContext usage with components."""
        class TestComponent(PipelineComponent):
            def execute(self, context):
                # Use context information
                return ExecutionResult(
                    success=True,
                    artifacts=[f"{context.experiment_id}_artifact"],
                    metrics={"stage": context.stage_name},
                    metadata={
                        "component_type": context.component_type.value,
                        "config_keys": list(context.config.keys())
                    }
                )
            
            def validate_config(self, config):
                return "required_param" in config
        
        component = TestComponent(ComponentType.MODEL_TRAINING)
        logger = logging.getLogger("test")
        
        context = ExecutionContext(
            experiment_id="exp_123",
            stage_name="training_stage",
            component_type=ComponentType.MODEL_TRAINING,
            config={"required_param": "value", "optional_param": 42},
            artifacts_path="/artifacts",
            logger=logger
        )
        
        # Validate config
        assert component.validate_config(context.config) is True
        
        # Execute component
        result = component.execute(context)
        
        assert result.success is True
        assert result.artifacts == ["exp_123_artifact"]
        assert result.metrics == {"stage": "training_stage"}
        assert result.metadata["component_type"] == "model_training"
        assert "required_param" in result.metadata["config_keys"]
        assert "optional_param" in result.metadata["config_keys"]
    
    def test_orchestrator_with_stages_and_components(self):
        """Test orchestrator integration with stages and components."""
        class TestComponent(PipelineComponent):
            def __init__(self, name):
                super().__init__(ComponentType.DATA_PREPROCESSING)
                self.name = name
            
            def execute(self, context):
                return ExecutionResult(
                    success=True,
                    artifacts=[f"{self.name}_output"],
                    metrics={f"{self.name}_score": 0.9},
                    metadata={"processed_by": self.name}
                )
            
            def validate_config(self, config):
                return True
        
        class TestOrchestrator(PipelineOrchestrator):
            def execute_pipeline(self, stages, context):
                all_artifacts = []
                all_metrics = {}
                
                for stage in stages:
                    stage_result = self.execute_stage(stage, context)
                    all_artifacts.extend(stage_result.artifacts)
                    all_metrics.update(stage_result.metrics)
                
                return ExecutionResult(
                    success=True,
                    artifacts=all_artifacts,
                    metrics=all_metrics,
                    metadata={"total_stages": len(stages)}
                )
            
            def execute_stage(self, stage, context):
                stage_artifacts = []
                stage_metrics = {}
                
                for component in stage.components:
                    comp_result = component.execute(context)
                    stage_artifacts.extend(comp_result.artifacts)
                    stage_metrics.update(comp_result.metrics)
                
                return ExecutionResult(
                    success=True,
                    artifacts=stage_artifacts,
                    metrics=stage_metrics,
                    metadata={"stage_name": stage.name}
                )
        
        # Create components and stages
        comp1 = TestComponent("preprocessor")
        comp2 = TestComponent("validator")
        
        stage1 = PipelineStage("preprocessing", [comp1])
        stage2 = PipelineStage("validation", [comp2], dependencies=["preprocessing"])
        
        # Create orchestrator and context
        orchestrator = TestOrchestrator()
        logger = logging.getLogger("test")
        context = ExecutionContext(
            experiment_id="integration_test",
            stage_name="pipeline",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={},
            artifacts_path="/tmp",
            logger=logger
        )
        
        # Execute pipeline
        result = orchestrator.execute_pipeline([stage1, stage2], context)
        
        assert result.success is True
        assert "preprocessor_output" in result.artifacts
        assert "validator_output" in result.artifacts
        assert result.metrics["preprocessor_score"] == 0.9
        assert result.metrics["validator_score"] == 0.9
        assert result.metadata["total_stages"] == 2