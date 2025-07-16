"""Pipeline orchestrator implementation."""

import time
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .interfaces import (
    PipelineOrchestrator as BasePipelineOrchestrator,
    PipelineStage,
    ExecutionContext,
    ExecutionResult,
    PipelineStatus
)


class PipelineOrchestrator(BasePipelineOrchestrator):
    """Default implementation of pipeline orchestrator."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)
        self._execution_history: List[Dict[str, Any]] = []
    
    def execute_pipeline(self, stages: List[PipelineStage], context: ExecutionContext) -> ExecutionResult:
        """Execute a complete pipeline with dependency resolution."""
        self.logger.info(f"Starting pipeline execution for experiment: {context.experiment_id}")
        
        start_time = time.time()
        executed_stages = set()
        all_artifacts = []
        all_metrics = {}
        
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
        """Execute a single pipeline stage."""
        self.logger.info(f"Executing stage: {stage.name}")
        
        start_time = time.time()
        stage_artifacts = []
        stage_metrics = {}
        
        try:
            if stage.parallel and len(stage.components) > 1:
                # Execute components in parallel
                results = self._execute_components_parallel(stage.components, context)
            else:
                # Execute components sequentially
                results = self._execute_components_sequential(stage.components, context)
            
            # Aggregate results
            for result in results:
                if not result.success:
                    return ExecutionResult(
                        success=False,
                        artifacts=stage_artifacts,
                        metrics=stage_metrics,
                        metadata={"stage": stage.name},
                        error_message=result.error_message,
                        execution_time=time.time() - start_time
                    )
                
                stage_artifacts.extend(result.artifacts)
                stage_metrics.update(result.metrics)
            
            execution_time = time.time() - start_time
            self.logger.info(f"Stage {stage.name} completed in {execution_time:.2f} seconds")
            
            return ExecutionResult(
                success=True,
                artifacts=stage_artifacts,
                metrics=stage_metrics,
                metadata={"stage": stage.name},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Stage {stage.name} failed: {str(e)}"
            self.logger.error(error_msg)
            
            return ExecutionResult(
                success=False,
                artifacts=stage_artifacts,
                metrics=stage_metrics,
                metadata={"stage": stage.name},
                error_message=error_msg,
                execution_time=execution_time
            )
    
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