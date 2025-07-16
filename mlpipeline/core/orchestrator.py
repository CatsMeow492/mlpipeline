"""Pipeline orchestrator implementation."""

import time
import logging
import uuid
import threading
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .interfaces import (
    PipelineOrchestrator as BasePipelineOrchestrator,
    PipelineStage,
    ExecutionContext,
    ExecutionResult,
    PipelineStatus,
    ComponentType
)
from .registry import component_registry
from .errors import ErrorHandler, RecoveryAction, ErrorContext


class PipelineOrchestrator(BasePipelineOrchestrator):
    """Enhanced pipeline orchestrator with component registry integration and structured logging."""
    
    def __init__(self, max_workers: int = 4, enable_checkpointing: bool = True, 
                 error_log_path: Optional[str] = None):
        self.max_workers = max_workers
        self.enable_checkpointing = enable_checkpointing
        self.logger = self._setup_structured_logger()
        self._execution_history: List[Dict[str, Any]] = []
        self._current_execution_id: Optional[str] = None
        self._lock = threading.Lock()
        self.error_handler = ErrorHandler(error_log_path)
        
    def _setup_structured_logger(self) -> logging.Logger:
        """Setup structured logger with correlation ID support."""
        logger = logging.getLogger(self.__class__.__name__)
        
        # Create custom formatter that includes correlation ID
        class CorrelationFormatter(logging.Formatter):
            def format(self, record):
                # Add correlation ID to log record if available
                correlation_id = getattr(threading.current_thread(), 'correlation_id', None)
                if correlation_id:
                    record.correlation_id = correlation_id
                else:
                    record.correlation_id = 'N/A'
                return super().format(record)
        
        # Set up formatter
        formatter = CorrelationFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
        )
        
        # Configure handler if not already configured
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def execute_pipeline(self, stages: List[PipelineStage], context: ExecutionContext) -> ExecutionResult:
        """Execute a complete pipeline with dependency resolution."""
        # Set up correlation ID for this execution
        execution_id = str(uuid.uuid4())
        self._current_execution_id = execution_id
        threading.current_thread().correlation_id = execution_id
        
        self.logger.info(f"Starting pipeline execution for experiment: {context.experiment_id}")
        
        start_time = time.time()
        executed_stages = set()
        all_artifacts = []
        all_metrics = {}
        
        # Record execution start
        execution_record = {
            "execution_id": execution_id,
            "experiment_id": context.experiment_id,
            "start_time": start_time,
            "status": PipelineStatus.RUNNING,
            "stages": [stage.name for stage in stages],
            "executed_stages": [],
            "artifacts": [],
            "metrics": {}
        }
        
        with self._lock:
            self._execution_history.append(execution_record)
        
        try:
            # Build dependency graph
            stage_map = {stage.name: stage for stage in stages}
            
            # Execute stages in dependency order
            while len(executed_stages) < len(stages):
                ready_stages = []
                
                for stage in stages:
                    if stage.name not in executed_stages:
                        # Check if all dependencies are satisfied
                        if all(dep in executed_stages for dep in stage.dependencies):
                            ready_stages.append(stage)
                
                if not ready_stages:
                    raise RuntimeError("Circular dependency detected or unresolvable dependencies")
                
                # Execute ready stages
                for stage in ready_stages:
                    stage_context = ExecutionContext(
                        experiment_id=context.experiment_id,
                        stage_name=stage.name,
                        component_type=context.component_type,
                        config=context.config,
                        artifacts_path=context.artifacts_path,
                        logger=context.logger,
                        metadata=context.metadata.copy()
                    )
                    
                    stage_result = self.execute_stage(stage, stage_context)
                    
                    if not stage_result.success:
                        return ExecutionResult(
                            success=False,
                            artifacts=all_artifacts,
                            metrics=all_metrics,
                            metadata={"failed_stage": stage.name},
                            error_message=f"Stage {stage.name} failed: {stage_result.error_message}",
                            execution_time=time.time() - start_time
                        )
                    
                    all_artifacts.extend(stage_result.artifacts)
                    all_metrics.update(stage_result.metrics)
                    executed_stages.add(stage.name)
            
            execution_time = time.time() - start_time
            self.logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")
            
            return ExecutionResult(
                success=True,
                artifacts=all_artifacts,
                metrics=all_metrics,
                metadata={"executed_stages": list(executed_stages)},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            return ExecutionResult(
                success=False,
                artifacts=all_artifacts,
                metrics=all_metrics,
                metadata={"executed_stages": list(executed_stages)},
                error_message=error_msg,
                execution_time=execution_time
            )
    
    def execute_stage(self, stage: PipelineStage, context: ExecutionContext) -> ExecutionResult:
        """Execute a single pipeline stage with error handling and recovery."""
        return self._execute_stage_with_recovery(stage, context)
    
    def _execute_components_sequential(self, components, context):
        """Execute components sequentially."""
        results = []
        for component in components:
            try:
                component.setup(context)
                result = component.execute(context)
                component.cleanup(context)
                results.append(result)
            except Exception as e:
                results.append(ExecutionResult(
                    success=False,
                    artifacts=[],
                    metrics={},
                    metadata={},
                    error_message=str(e)
                ))
        return results
    
    def _execute_components_parallel(self, components, context):
        """Execute components in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(components))) as executor:
            # Submit all components for execution
            future_to_component = {}
            for component in components:
                future = executor.submit(self._execute_single_component, component, context)
                future_to_component[future] = component
            
            # Collect results as they complete
            for future in as_completed(future_to_component):
                component = future_to_component[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Component {component.__class__.__name__} failed: {str(e)}")
                    results.append(ExecutionResult(
                        success=False,
                        artifacts=[],
                        metrics={},
                        metadata={},
                        error_message=str(e)
                    ))
        
        return results
    
    def _execute_single_component(self, component, context):
        """Execute a single component with setup and cleanup."""
        try:
            component.setup(context)
            result = component.execute(context)
            component.cleanup(context)
            return result
        except Exception as e:
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def _execute_component_with_recovery(self, component, context: ExecutionContext) -> ExecutionResult:
        """Execute a component with error handling and recovery mechanisms."""
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                component.setup(context)
                result = component.execute(context)
                component.cleanup(context)
                return result
                
            except Exception as e:
                attempt += 1
                
                # Create error context
                error_context = self.error_handler.classify_error(e, {
                    'stage_name': context.stage_name,
                    'component_name': getattr(component, 'name', component.__class__.__name__),
                    'experiment_id': context.experiment_id,
                    'execution_id': self._current_execution_id,
                    'attempt': attempt
                })
                
                # Determine recovery action
                recovery_action = self.error_handler.handle_error(error_context)
                
                if recovery_action == RecoveryAction.RETRY and attempt < max_attempts:
                    # Calculate retry delay
                    delay = self.error_handler.calculate_retry_delay(error_context)
                    self.logger.warning(f"Retrying component {component.__class__.__name__} in {delay} seconds (attempt {attempt}/{max_attempts})")
                    time.sleep(delay)
                    continue
                    
                elif recovery_action == RecoveryAction.FALLBACK:
                    # Try fallback component if available
                    fallback_name = self.error_handler.get_fallback_component(error_context)
                    if fallback_name:
                        fallback_component = component_registry.get_component_class(fallback_name)
                        if fallback_component:
                            self.logger.info(f"Using fallback component: {fallback_name}")
                            try:
                                fallback_instance = fallback_component()
                                return self._execute_single_component(fallback_instance, context)
                            except Exception as fallback_error:
                                self.logger.error(f"Fallback component also failed: {str(fallback_error)}")
                
                elif recovery_action == RecoveryAction.SKIP:
                    self.logger.warning(f"Skipping component {component.__class__.__name__} due to error")
                    return ExecutionResult(
                        success=True,  # Mark as success to continue pipeline
                        artifacts=[],
                        metrics={},
                        metadata={"skipped": True, "reason": str(e)},
                        error_message=f"Component skipped: {str(e)}"
                    )
                
                # If we reach here, either ABORT or max retries exceeded
                return ExecutionResult(
                    success=False,
                    artifacts=[],
                    metrics={},
                    metadata={"error_context": error_context.error_id},
                    error_message=str(e)
                )
        
        # Max attempts exceeded
        return ExecutionResult(
            success=False,
            artifacts=[],
            metrics={},
            metadata={"max_retries_exceeded": True},
            error_message=f"Component failed after {max_attempts} attempts"
        )
    
    def _execute_stage_with_recovery(self, stage: PipelineStage, context: ExecutionContext) -> ExecutionResult:
        """Execute a stage with comprehensive error handling and recovery."""
        self.logger.info(f"Executing stage: {stage.name}")
        
        start_time = time.time()
        stage_artifacts = []
        stage_metrics = {}
        failed_components = []
        
        try:
            if stage.parallel and len(stage.components) > 1:
                # Execute components in parallel with recovery
                results = self._execute_components_parallel_with_recovery(stage.components, context)
            else:
                # Execute components sequentially with recovery
                results = self._execute_components_sequential_with_recovery(stage.components, context)
            
            # Aggregate results and handle partial failures
            for i, result in enumerate(results):
                if not result.success:
                    failed_components.append(stage.components[i].__class__.__name__)
                    
                    # Check if we should continue despite failure
                    if not result.metadata.get("skipped", False):
                        # This is a real failure, not a skip
                        if len(failed_components) > len(stage.components) * 0.5:  # More than 50% failed
                            return ExecutionResult(
                                success=False,
                                artifacts=stage_artifacts,
                                metrics=stage_metrics,
                                metadata={
                                    "stage": stage.name,
                                    "failed_components": failed_components,
                                    "partial_failure": True
                                },
                                error_message=f"Stage {stage.name} failed: too many component failures",
                                execution_time=time.time() - start_time
                            )
                
                stage_artifacts.extend(result.artifacts)
                stage_metrics.update(result.metrics)
            
            execution_time = time.time() - start_time
            
            if failed_components:
                self.logger.warning(f"Stage {stage.name} completed with {len(failed_components)} failed components")
            else:
                self.logger.info(f"Stage {stage.name} completed successfully in {execution_time:.2f} seconds")
            
            return ExecutionResult(
                success=True,
                artifacts=stage_artifacts,
                metrics=stage_metrics,
                metadata={
                    "stage": stage.name,
                    "failed_components": failed_components if failed_components else None
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle stage-level errors
            error_context = self.error_handler.classify_error(e, {
                'stage_name': stage.name,
                'experiment_id': context.experiment_id,
                'execution_id': self._current_execution_id
            })
            
            recovery_action = self.error_handler.handle_error(error_context)
            
            if recovery_action == RecoveryAction.SKIP:
                self.logger.warning(f"Skipping stage {stage.name} due to error")
                return ExecutionResult(
                    success=True,  # Mark as success to continue pipeline
                    artifacts=[],
                    metrics={},
                    metadata={"stage": stage.name, "skipped": True, "reason": str(e)},
                    error_message=f"Stage skipped: {str(e)}",
                    execution_time=execution_time
                )
            
            error_msg = f"Stage {stage.name} failed: {str(e)}"
            self.logger.error(error_msg)
            
            return ExecutionResult(
                success=False,
                artifacts=stage_artifacts,
                metrics=stage_metrics,
                metadata={"stage": stage.name, "error_context": error_context.error_id},
                error_message=error_msg,
                execution_time=execution_time
            )
    
    def _execute_components_sequential_with_recovery(self, components, context):
        """Execute components sequentially with recovery mechanisms."""
        results = []
        for component in components:
            result = self._execute_component_with_recovery(component, context)
            results.append(result)
            
            # If component failed and it's critical, stop execution
            if not result.success and not result.metadata.get("skipped", False):
                # Check if this is a critical failure that should stop the stage
                if hasattr(component, 'critical') and component.critical:
                    break
        
        return results
    
    def _execute_components_parallel_with_recovery(self, components, context):
        """Execute components in parallel with recovery mechanisms."""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(components))) as executor:
            # Submit all components for execution with recovery
            future_to_component = {}
            for component in components:
                future = executor.submit(self._execute_component_with_recovery, component, context)
                future_to_component[future] = component
            
            # Collect results as they complete
            for future in as_completed(future_to_component):
                component = future_to_component[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Component {component.__class__.__name__} failed: {str(e)}")
                    results.append(ExecutionResult(
                        success=False,
                        artifacts=[],
                        metrics={},
                        metadata={},
                        error_message=str(e)
                    ))
        
        return results
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history."""
        with self._lock:
            return self._execution_history.copy()
    
    def get_current_execution_id(self) -> Optional[str]:
        """Get the current execution ID."""
        return self._current_execution_id
    
    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        with self._lock:
            self._execution_history.clear()
    
    def create_stage_from_config(self, stage_name: str, stage_config: Dict[str, Any]) -> PipelineStage:
        """Create a pipeline stage from configuration using component registry."""
        components = []
        
        for component_config in stage_config.get('components', []):
            component_name = component_config.get('name')
            component_type = component_config.get('type')
            component_params = component_config.get('parameters', {})
            
            # Try to get component from registry
            component_class = component_registry.get_component_class(component_name)
            if component_class:
                try:
                    component = component_class(**component_params)
                    components.append(component)
                    self.logger.info(f"Created component {component_name} for stage {stage_name}")
                except Exception as e:
                    self.logger.error(f"Failed to create component {component_name}: {str(e)}")
                    raise
            else:
                self.logger.error(f"Component {component_name} not found in registry")
                raise ValueError(f"Component {component_name} not found in registry")
        
        return PipelineStage(
            name=stage_name,
            components=components,
            dependencies=stage_config.get('dependencies', []),
            parallel=stage_config.get('parallel', False)
        )
    
    def validate_pipeline_stages(self, stages: List[PipelineStage]) -> List[str]:
        """Validate pipeline stages and return list of validation errors."""
        errors = []
        stage_names = {stage.name for stage in stages}
        
        # Check for duplicate stage names
        seen_names = set()
        for stage in stages:
            if stage.name in seen_names:
                errors.append(f"Duplicate stage name: {stage.name}")
            seen_names.add(stage.name)
        
        # Check for invalid dependencies
        for stage in stages:
            for dep in stage.dependencies:
                if dep not in stage_names:
                    errors.append(f"Stage {stage.name} depends on non-existent stage: {dep}")
        
        # Check for circular dependencies
        def has_circular_dependency(stage_name: str, visited: set, path: set) -> bool:
            if stage_name in path:
                return True
            if stage_name in visited:
                return False
            
            visited.add(stage_name)
            path.add(stage_name)
            
            stage = next((s for s in stages if s.name == stage_name), None)
            if stage:
                for dep in stage.dependencies:
                    if has_circular_dependency(dep, visited, path):
                        return True
            
            path.remove(stage_name)
            return False
        
        visited = set()
        for stage in stages:
            if stage.name not in visited:
                if has_circular_dependency(stage.name, visited, set()):
                    errors.append(f"Circular dependency detected involving stage: {stage.name}")
        
        return errors
    
    def save_checkpoint(self, execution_id: str, checkpoint_data: Dict[str, Any], 
                       checkpoint_path: Optional[str] = None) -> None:
        """Save execution checkpoint."""
        if not self.enable_checkpointing:
            return
        
        if checkpoint_path is None:
            checkpoint_path = f"checkpoints/{execution_id}.json"
        
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self, execution_id: str, checkpoint_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load execution checkpoint."""
        if not self.enable_checkpointing:
            return None
        
        if checkpoint_path is None:
            checkpoint_path = f"checkpoints/{execution_id}.json"
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return None
        
        import json
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            return None
    
    def get_stage_execution_order(self, stages: List[PipelineStage]) -> List[List[str]]:
        """Get the execution order of stages considering dependencies."""
        stage_map = {stage.name: stage for stage in stages}
        executed = set()
        execution_order = []
        
        while len(executed) < len(stages):
            ready_stages = []
            
            for stage in stages:
                if stage.name not in executed:
                    if all(dep in executed for dep in stage.dependencies):
                        ready_stages.append(stage.name)
            
            if not ready_stages:
                raise RuntimeError("Circular dependency detected or unresolvable dependencies")
            
            execution_order.append(ready_stages)
            executed.update(ready_stages)
        
        return execution_order