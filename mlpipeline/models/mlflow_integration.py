"""MLflow integration for experiment tracking, model registry, and artifact management."""

import logging
import json
import os
import tempfile
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    # Create mock classes for type hints when MLflow is not available
    class MockMLflow:
        class MlflowClient:
            pass
    
    mlflow = MockMLflow()
    MlflowClient = MockMLflow.MlflowClient
    MLFLOW_AVAILABLE = False

from .training import ModelTrainer, ModelConfig, TrainingMetrics
from .evaluation import ModelEvaluator, EvaluationMetrics
from .hyperparameter_optimization import HyperparameterOptimizedTrainer, OptimizationResult
from ..core.interfaces import PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
from ..core.errors import ModelError, ConfigurationError


@dataclass
class MLflowConfig:
    """Configuration for MLflow integration."""
    tracking_uri: Optional[str] = None
    experiment_name: str = "ml-pipeline-experiment"
    run_name: Optional[str] = None
    artifact_location: Optional[str] = None
    registry_uri: Optional[str] = None
    
    # Logging configuration
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    log_model: bool = True
    log_input_example: bool = False
    log_model_signature: bool = True
    
    # Model registry configuration
    register_model: bool = False
    model_name: Optional[str] = None
    model_stage: str = "None"  # None, Staging, Production, Archived
    model_description: Optional[str] = None
    
    # Tags
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class MLflowRunInfo:
    """Information about an MLflow run."""
    run_id: str
    experiment_id: str
    run_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    artifact_uri: str
    lifecycle_stage: str
    user_id: str
    tags: Dict[str, str]
    params: Dict[str, str]
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'experiment_id': self.experiment_id,
            'run_name': self.run_name,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'artifact_uri': self.artifact_uri,
            'lifecycle_stage': self.lifecycle_stage,
            'user_id': self.user_id,
            'tags': self.tags,
            'params': self.params,
            'metrics': self.metrics
        }


class MLflowTracker:
    """MLflow experiment tracking and model registry integration."""
    
    def __init__(self, config: MLflowConfig):
        if not MLFLOW_AVAILABLE:
            raise ModelError("MLflow is not installed. Please install it with: pip install mlflow")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = None
        self.experiment_id = None
        self.current_run = None
        
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking and registry."""
        # Set tracking URI
        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            self.logger.info(f"MLflow tracking URI set to: {self.config.tracking_uri}")
        
        # Set registry URI if different from tracking URI
        if self.config.registry_uri:
            mlflow.set_registry_uri(self.config.registry_uri)
            self.logger.info(f"MLflow registry URI set to: {self.config.registry_uri}")
        
        # Create MLflow client
        self.client = MlflowClient()
        
        # Setup experiment
        self._setup_experiment()
    
    def _setup_experiment(self) -> None:
        """Setup or create MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                # Create new experiment
                self.experiment_id = mlflow.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                    tags=self.config.tags
                )
                self.logger.info(f"Created new MLflow experiment: {self.config.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                self.logger.info(f"Using existing MLflow experiment: {self.config.experiment_name}")
            
            # Set experiment
            mlflow.set_experiment(self.config.experiment_name)
            
        except Exception as e:
            self.logger.error(f"Failed to setup MLflow experiment: {str(e)}")
            raise ModelError(f"MLflow experiment setup failed: {str(e)}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new MLflow run."""
        try:
            # Combine default tags with provided tags
            all_tags = self.config.tags.copy()
            if tags:
                all_tags.update(tags)
            
            # Add system tags
            all_tags.update({
                'mlflow.source.type': 'LOCAL',
                'mlflow.user': os.getenv('USER', 'unknown'),
                'pipeline.timestamp': datetime.now().isoformat()
            })
            
            # Start run
            self.current_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name or self.config.run_name,
                tags=all_tags
            )
            
            run_id = self.current_run.info.run_id
            self.logger.info(f"Started MLflow run: {run_id}")
            
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to start MLflow run: {str(e)}")
            raise ModelError(f"MLflow run start failed: {str(e)}")
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        try:
            if self.current_run:
                mlflow.end_run(status=status)
                self.logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")
                self.current_run = None
            
        except Exception as e:
            self.logger.error(f"Failed to end MLflow run: {str(e)}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self.config.log_params or not self.current_run:
            return
        
        try:
            # Convert all parameters to strings (MLflow requirement)
            str_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    str_params[key] = json.dumps(value)
                else:
                    str_params[key] = str(value)
            
            mlflow.log_params(str_params)
            self.logger.debug(f"Logged {len(str_params)} parameters to MLflow")
            
        except Exception as e:
            self.logger.warning(f"Failed to log parameters: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        if not self.config.log_metrics or not self.current_run:
            return
        
        try:
            # Filter out non-numeric metrics
            numeric_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    numeric_metrics[key] = float(value)
            
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)
                self.logger.debug(f"Logged {len(numeric_metrics)} metrics to MLflow")
            
        except Exception as e:
            self.logger.warning(f"Failed to log metrics: {str(e)}")
    
    def log_artifacts(self, artifacts: List[str], artifact_path: Optional[str] = None) -> None:
        """Log artifacts to MLflow."""
        if not self.config.log_artifacts or not self.current_run:
            return
        
        try:
            for artifact in artifacts:
                artifact_path_obj = Path(artifact)
                if artifact_path_obj.exists():
                    if artifact_path_obj.is_file():
                        mlflow.log_artifact(str(artifact_path_obj), artifact_path)
                    elif artifact_path_obj.is_dir():
                        mlflow.log_artifacts(str(artifact_path_obj), artifact_path)
            
            self.logger.debug(f"Logged {len(artifacts)} artifacts to MLflow")
            
        except Exception as e:
            self.logger.warning(f"Failed to log artifacts: {str(e)}")
    
    def log_model(self, model: Any, model_name: str, framework: str, 
                  input_example: Optional[Any] = None, signature: Optional[Any] = None,
                  conda_env: Optional[str] = None, pip_requirements: Optional[List[str]] = None) -> None:
        """Log model to MLflow."""
        if not self.config.log_model or not self.current_run:
            return
        
        try:
            # Log model based on framework
            if framework == 'sklearn':
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_name,
                    conda_env=conda_env,
                    pip_requirements=pip_requirements,
                    input_example=input_example if self.config.log_input_example else None,
                    signature=signature if self.config.log_model_signature else None
                )
            elif framework == 'xgboost':
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path=model_name,
                    conda_env=conda_env,
                    pip_requirements=pip_requirements,
                    input_example=input_example if self.config.log_input_example else None,
                    signature=signature if self.config.log_model_signature else None
                )
            elif framework == 'pytorch':
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=model_name,
                    conda_env=conda_env,
                    pip_requirements=pip_requirements,
                    input_example=input_example if self.config.log_input_example else None,
                    signature=signature if self.config.log_model_signature else None
                )
            else:
                # Generic model logging
                mlflow.log_artifact(str(model), model_name)
            
            self.logger.info(f"Logged {framework} model '{model_name}' to MLflow")
            
        except Exception as e:
            self.logger.warning(f"Failed to log model: {str(e)}")
    
    def register_model(self, model_uri: str, model_name: str, 
                      description: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Register model in MLflow Model Registry."""
        if not self.config.register_model or not model_name:
            return None
        
        try:
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Update model version description
            if description or self.config.model_description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description or self.config.model_description
                )
            
            # Transition to specified stage if not None
            if self.config.model_stage and self.config.model_stage != "None":
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage=self.config.model_stage
                )
            
            self.logger.info(f"Registered model '{model_name}' version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {str(e)}")
            return None
    
    def get_run_info(self, run_id: Optional[str] = None) -> Optional[MLflowRunInfo]:
        """Get information about a specific run."""
        try:
            target_run_id = run_id or (self.current_run.info.run_id if self.current_run else None)
            if not target_run_id:
                return None
            
            run = self.client.get_run(target_run_id)
            
            return MLflowRunInfo(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                run_name=run.data.tags.get('mlflow.runName', ''),
                status=run.info.status,
                start_time=datetime.fromtimestamp(run.info.start_time / 1000),
                end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                artifact_uri=run.info.artifact_uri,
                lifecycle_stage=run.info.lifecycle_stage,
                user_id=run.info.user_id,
                tags=run.data.tags,
                params=run.data.params,
                metrics=run.data.metrics
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get run info: {str(e)}")
            return None
    
    def search_runs(self, filter_string: Optional[str] = None, 
                   max_results: int = 100) -> List[MLflowRunInfo]:
        """Search for runs in the current experiment."""
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=max_results
            )
            
            run_infos = []
            for run in runs:
                run_info = MLflowRunInfo(
                    run_id=run.info.run_id,
                    experiment_id=run.info.experiment_id,
                    run_name=run.data.tags.get('mlflow.runName', ''),
                    status=run.info.status,
                    start_time=datetime.fromtimestamp(run.info.start_time / 1000),
                    end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    artifact_uri=run.info.artifact_uri,
                    lifecycle_stage=run.info.lifecycle_stage,
                    user_id=run.info.user_id,
                    tags=run.data.tags,
                    params=run.data.params,
                    metrics=run.data.metrics
                )
                run_infos.append(run_info)
            
            return run_infos
            
        except Exception as e:
            self.logger.error(f"Failed to search runs: {str(e)}")
            return []
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs and return comparison DataFrame."""
        try:
            runs_data = []
            
            for run_id in run_ids:
                run_info = self.get_run_info(run_id)
                if run_info:
                    run_data = {
                        'run_id': run_info.run_id,
                        'run_name': run_info.run_name,
                        'status': run_info.status,
                        'start_time': run_info.start_time,
                        **run_info.params,
                        **run_info.metrics
                    }
                    runs_data.append(run_data)
            
            return pd.DataFrame(runs_data)
            
        except Exception as e:
            self.logger.error(f"Failed to compare runs: {str(e)}")
            return pd.DataFrame()


class MLflowIntegratedTrainer(ModelTrainer):
    """Model trainer with MLflow integration."""
    
    def __init__(self, mlflow_config: Optional[MLflowConfig] = None):
        super().__init__()
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.mlflow_tracker = None
        self.run_id = None
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration including MLflow settings."""
        if not super().validate_config(config):
            return False
        
        # Validate MLflow configuration
        mlflow_config = config.get('mlflow', {})
        if mlflow_config.get('enabled', False):
            if not MLFLOW_AVAILABLE:
                self.logger.error("MLflow is not installed but MLflow integration is enabled")
                return False
        
        return True
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup model training with MLflow integration."""
        super().setup(context)
        
        # Setup MLflow if enabled
        config = context.config.get('mlflow', {})
        if config.get('enabled', False):
            # Update MLflow configuration from context
            self.mlflow_config.tracking_uri = config.get('tracking_uri', self.mlflow_config.tracking_uri)
            self.mlflow_config.experiment_name = config.get('experiment_name', self.mlflow_config.experiment_name)
            self.mlflow_config.run_name = config.get('run_name', self.mlflow_config.run_name)
            self.mlflow_config.register_model = config.get('register_model', self.mlflow_config.register_model)
            self.mlflow_config.model_name = config.get('model_name', self.mlflow_config.model_name)
            
            # Create MLflow tracker
            self.mlflow_tracker = MLflowTracker(self.mlflow_config)
            self.logger.info("MLflow integration enabled")
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute model training with MLflow tracking."""
        if self.mlflow_tracker:
            return self._execute_with_mlflow(context)
        else:
            return super().execute(context)
    
    def _execute_with_mlflow(self, context: ExecutionContext) -> ExecutionResult:
        """Execute training with MLflow tracking."""
        try:
            # Start MLflow run
            run_tags = {
                'pipeline.component': 'model_training',
                'pipeline.experiment_id': context.experiment_id,
                'pipeline.stage': context.stage_name
            }
            
            self.run_id = self.mlflow_tracker.start_run(
                run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=run_tags
            )
            
            # Log configuration parameters
            self._log_training_config(context)
            
            # Execute training
            result = super().execute(context)
            
            if result.success:
                # Log training results
                self._log_training_results(result, context)
                
                # End run successfully
                self.mlflow_tracker.end_run("FINISHED")
                
                # Add MLflow information to result metadata
                result.metadata.update({
                    'mlflow_run_id': self.run_id,
                    'mlflow_experiment_id': self.mlflow_tracker.experiment_id,
                    'mlflow_tracking_uri': self.mlflow_config.tracking_uri
                })
            else:
                # End run with failure
                self.mlflow_tracker.end_run("FAILED")
            
            return result
            
        except Exception as e:
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run("FAILED")
            
            self.logger.error(f"MLflow integrated training failed: {str(e)}")
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def _log_training_config(self, context: ExecutionContext) -> None:
        """Log training configuration to MLflow."""
        if not self.mlflow_tracker:
            return
        
        # Log model configuration
        model_config = {
            'model_type': self.model_adapter.config.model_type,
            'framework': self.model_adapter.config.framework,
            'task_type': self.model_adapter.config.task_type,
            'random_state': self.model_adapter.config.random_state,
            **self.model_adapter.config.parameters
        }
        
        # Log training configuration
        training_config = context.config.get('training', {})
        
        # Combine all parameters
        all_params = {
            **model_config,
            'target_column': training_config.get('target_column', 'target'),
            'experiment_id': context.experiment_id,
            'stage_name': context.stage_name
        }
        
        self.mlflow_tracker.log_params(all_params)
    
    def _log_training_results(self, result: ExecutionResult, context: ExecutionContext) -> None:
        """Log training results to MLflow."""
        if not self.mlflow_tracker:
            return
        
        # Log metrics
        self.mlflow_tracker.log_metrics(result.metrics)
        
        # Log artifacts
        if result.artifacts:
            self.mlflow_tracker.log_artifacts(result.artifacts)
        
        # Log model if available
        if hasattr(self, 'model_adapter') and self.model_adapter and self.model_adapter.model:
            model_name = f"{self.model_adapter.config.framework}_{self.model_adapter.config.model_type}"
            
            # Create input example if requested
            input_example = None
            if self.mlflow_config.log_input_example:
                try:
                    # Load a small sample of training data for input example
                    artifacts_path = Path(context.artifacts_path)
                    train_file = artifacts_path / "train_preprocessed.parquet"
                    if train_file.exists():
                        sample_data = pd.read_parquet(train_file).head(5)
                        target_column = context.config.get('training', {}).get('target_column', 'target')
                        if target_column in sample_data.columns:
                            input_example = sample_data.drop(columns=[target_column])
                except Exception as e:
                    self.logger.warning(f"Could not create input example: {str(e)}")
            
            # Log model
            self.mlflow_tracker.log_model(
                model=self.model_adapter.model,
                model_name=model_name,
                framework=self.model_adapter.config.framework,
                input_example=input_example
            )
            
            # Register model if configured
            if self.mlflow_config.register_model and self.mlflow_config.model_name:
                model_uri = f"runs:/{self.run_id}/{model_name}"
                model_version = self.mlflow_tracker.register_model(
                    model_uri=model_uri,
                    model_name=self.mlflow_config.model_name,
                    description=f"Model trained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                if model_version:
                    result.metadata['mlflow_model_version'] = model_version


class MLflowIntegratedEvaluator(ModelEvaluator):
    """Model evaluator with MLflow integration."""
    
    def __init__(self, mlflow_config: Optional[MLflowConfig] = None):
        super().__init__()
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.mlflow_tracker = None
        self.run_id = None
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup model evaluation with MLflow integration."""
        super().setup(context)
        
        # Setup MLflow if enabled
        config = context.config.get('mlflow', {})
        if config.get('enabled', False):
            # Update MLflow configuration from context
            self.mlflow_config.tracking_uri = config.get('tracking_uri', self.mlflow_config.tracking_uri)
            self.mlflow_config.experiment_name = config.get('experiment_name', self.mlflow_config.experiment_name)
            
            # Create MLflow tracker
            self.mlflow_tracker = MLflowTracker(self.mlflow_config)
            self.logger.info("MLflow integration enabled for evaluation")
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute model evaluation with MLflow tracking."""
        if self.mlflow_tracker:
            return self._execute_with_mlflow(context)
        else:
            return super().execute(context)
    
    def _execute_with_mlflow(self, context: ExecutionContext) -> ExecutionResult:
        """Execute evaluation with MLflow tracking."""
        try:
            # Start MLflow run
            run_tags = {
                'pipeline.component': 'model_evaluation',
                'pipeline.experiment_id': context.experiment_id,
                'pipeline.stage': context.stage_name
            }
            
            self.run_id = self.mlflow_tracker.start_run(
                run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=run_tags
            )
            
            # Log evaluation configuration
            eval_config = context.config.get('evaluation', {})
            self.mlflow_tracker.log_params({
                'evaluation_metrics': eval_config.get('metrics', []),
                'experiment_id': context.experiment_id,
                'stage_name': context.stage_name
            })
            
            # Execute evaluation
            result = super().execute(context)
            
            if result.success:
                # Log evaluation results
                self._log_evaluation_results(result)
                
                # End run successfully
                self.mlflow_tracker.end_run("FINISHED")
                
                # Add MLflow information to result metadata
                result.metadata.update({
                    'mlflow_run_id': self.run_id,
                    'mlflow_experiment_id': self.mlflow_tracker.experiment_id
                })
            else:
                # End run with failure
                self.mlflow_tracker.end_run("FAILED")
            
            return result
            
        except Exception as e:
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run("FAILED")
            
            self.logger.error(f"MLflow integrated evaluation failed: {str(e)}")
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def _log_evaluation_results(self, result: ExecutionResult) -> None:
        """Log evaluation results to MLflow."""
        if not self.mlflow_tracker:
            return
        
        # Log metrics
        self.mlflow_tracker.log_metrics(result.metrics)
        
        # Log artifacts (plots, reports, etc.)
        if result.artifacts:
            self.mlflow_tracker.log_artifacts(result.artifacts, "evaluation")


class MLflowIntegratedHyperparameterTrainer(HyperparameterOptimizedTrainer):
    """Hyperparameter optimized trainer with MLflow integration."""
    
    def __init__(self, mlflow_config: Optional[MLflowConfig] = None):
        super().__init__()
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.mlflow_tracker = None
        self.parent_run_id = None
        self.child_runs = []
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup hyperparameter optimization with MLflow integration."""
        super().setup(context)
        
        # Setup MLflow if enabled
        config = context.config.get('mlflow', {})
        if config.get('enabled', False):
            # Update MLflow configuration from context
            self.mlflow_config.tracking_uri = config.get('tracking_uri', self.mlflow_config.tracking_uri)
            self.mlflow_config.experiment_name = config.get('experiment_name', self.mlflow_config.experiment_name)
            
            # Create MLflow tracker
            self.mlflow_tracker = MLflowTracker(self.mlflow_config)
            self.logger.info("MLflow integration enabled for hyperparameter optimization")
    
    def _execute_with_optimization(self, context: ExecutionContext) -> ExecutionResult:
        """Execute hyperparameter optimization with MLflow tracking."""
        if not self.mlflow_tracker:
            return super()._execute_with_optimization(context)
        
        try:
            # Start parent MLflow run for the entire optimization
            run_tags = {
                'pipeline.component': 'hyperparameter_optimization',
                'pipeline.experiment_id': context.experiment_id,
                'pipeline.stage': context.stage_name,
                'optimization.method': self.hyperparameter_optimizer.optimization_config.method
            }
            
            self.parent_run_id = self.mlflow_tracker.start_run(
                run_name=f"hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=run_tags
            )
            
            # Log optimization configuration
            opt_config = self.hyperparameter_optimizer.optimization_config
            self.mlflow_tracker.log_params({
                'optimization_method': opt_config.method,
                'n_trials': opt_config.n_trials,
                'sampler': opt_config.sampler,
                'pruner': opt_config.pruner,
                'direction': opt_config.direction,
                'metric': opt_config.metric,
                'cv_folds': opt_config.cv_folds
            })
            
            # Execute optimization with MLflow tracking for each trial
            result = self._execute_optimization_with_mlflow_trials(context)
            
            if result.success:
                # Log final optimization results
                self._log_optimization_results(result)
                
                # End parent run successfully
                self.mlflow_tracker.end_run("FINISHED")
                
                # Add MLflow information to result metadata
                result.metadata.update({
                    'mlflow_parent_run_id': self.parent_run_id,
                    'mlflow_child_runs': self.child_runs,
                    'mlflow_experiment_id': self.mlflow_tracker.experiment_id
                })
            else:
                # End parent run with failure
                self.mlflow_tracker.end_run("FAILED")
            
            return result
            
        except Exception as e:
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run("FAILED")
            
            self.logger.error(f"MLflow integrated hyperparameter optimization failed: {str(e)}")
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def _execute_optimization_with_mlflow_trials(self, context: ExecutionContext) -> ExecutionResult:
        """Execute optimization with MLflow tracking for individual trials."""
        # This would require modifying the Optuna objective function to log each trial
        # For now, we'll execute the standard optimization and log the results
        result = super()._execute_with_optimization(context)
        
        # Log individual trials if optimization was successful
        if result.success and self.optimization_result:
            self._log_individual_trials()
        
        return result
    
    def _log_individual_trials(self) -> None:
        """Log individual optimization trials as child runs."""
        if not self.mlflow_tracker or not self.optimization_result:
            return
        
        try:
            for trial_data in self.optimization_result.all_trials:
                # Create child run for each trial
                with mlflow.start_run(nested=True) as child_run:
                    child_run_id = child_run.info.run_id
                    self.child_runs.append(child_run_id)
                    
                    # Log trial parameters
                    mlflow.log_params(trial_data['params'])
                    
                    # Log trial metrics
                    if trial_data['value'] is not None:
                        mlflow.log_metric('objective_value', trial_data['value'])
                    
                    # Log trial metadata
                    mlflow.log_params({
                        'trial_number': trial_data['number'],
                        'trial_state': trial_data['state'],
                        'trial_duration': trial_data.get('duration', 0)
                    })
                    
                    # Add tags
                    mlflow.set_tags({
                        'trial.number': str(trial_data['number']),
                        'trial.state': trial_data['state'],
                        'optimization.parent_run': self.parent_run_id
                    })
        
        except Exception as e:
            self.logger.warning(f"Failed to log individual trials: {str(e)}")
    
    def _log_optimization_results(self, result: ExecutionResult) -> None:
        """Log optimization results to parent MLflow run."""
        if not self.mlflow_tracker or not self.optimization_result:
            return
        
        # Log optimization summary metrics
        opt_metrics = {
            'best_objective_value': self.optimization_result.best_value,
            'total_trials': self.optimization_result.n_trials,
            'pruned_trials': self.optimization_result.pruned_trials,
            'failed_trials': self.optimization_result.failed_trials,
            'optimization_time_seconds': self.optimization_result.optimization_time
        }
        
        self.mlflow_tracker.log_metrics(opt_metrics)
        
        # Log best parameters
        best_params = {f'best_{k}': str(v) for k, v in self.optimization_result.best_params.items()}
        self.mlflow_tracker.log_params(best_params)
        
        # Log artifacts
        if result.artifacts:
            self.mlflow_tracker.log_artifacts(result.artifacts, "optimization")