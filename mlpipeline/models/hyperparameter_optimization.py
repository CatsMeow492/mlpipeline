"""Hyperparameter optimization using Optuna with early stopping and pruning."""

import logging
import json
import pickle
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import TPESampler, RandomSampler, GridSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    # Create mock classes for type hints when Optuna is not available
    class MockOptuna:
        class Trial:
            pass
        class Study:
            pass
    
    optuna = MockOptuna()
    MedianPruner = None
    SuccessiveHalvingPruner = None
    TPESampler = None
    RandomSampler = None
    GridSampler = None
    OPTUNA_AVAILABLE = False

from .training import ModelTrainer, ModelConfig, TrainingMetrics, ModelAdapter
from ..core.interfaces import ExecutionContext, ExecutionResult
from ..core.errors import ModelError, ConfigurationError


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""
    method: str = "optuna"  # optuna, grid_search, random_search
    n_trials: int = 50
    timeout: Optional[int] = None  # seconds
    sampler: str = "tpe"  # tpe, random, grid
    pruner: str = "median"  # median, successive_halving, none
    direction: str = "maximize"  # maximize or minimize
    metric: str = "accuracy"  # metric to optimize
    early_stopping_rounds: Optional[int] = None
    cv_folds: int = 5
    random_state: Optional[int] = None
    
    # Parameter search spaces
    parameter_space: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parameter_space is None:
            self.parameter_space = {}


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_value: float
    best_trial_number: int
    n_trials: int
    optimization_time: float
    study_name: str
    all_trials: List[Dict[str, Any]]
    pruned_trials: int
    failed_trials: int
    
    def __post_init__(self):
        if self.all_trials is None:
            self.all_trials = []


class HyperparameterOptimizer:
    """Hyperparameter optimization engine using Optuna."""
    
    def __init__(self, model_trainer: ModelTrainer):
        self.model_trainer = model_trainer
        self.logger = logging.getLogger(self.__class__.__name__)
        self.study = None
        self.optimization_config = None
        self.training_data = None
        self.validation_data = None
        
        if not OPTUNA_AVAILABLE:
            raise ModelError("Optuna is not installed. Please install it with: pip install optuna")
    
    def setup_optimization(self, config: HyperparameterConfig, context: ExecutionContext) -> None:
        """Setup hyperparameter optimization."""
        self.optimization_config = config
        
        # Load training data
        self.training_data, self.validation_data, _ = self.model_trainer._load_training_data(context)
        
        if self.validation_data is None:
            # Create validation split from training data
            from sklearn.model_selection import train_test_split
            
            X_train, y_train = self.model_trainer._split_features_target(self.training_data, context.config)
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, 
                test_size=0.2, 
                random_state=config.random_state,
                stratify=y_train if self.model_trainer.model_adapter.config.task_type == 'classification' else None
            )
            
            # Recreate dataframes with target column
            target_column = context.config.get('training', {}).get('target_column', 'target')
            self.training_data = pd.concat([X_train_split, y_train_split.rename(target_column)], axis=1)
            self.validation_data = pd.concat([X_val_split, y_val_split.rename(target_column)], axis=1)
        
        # Setup Optuna study
        study_name = f"hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configure sampler
        sampler = self._create_sampler(config)
        
        # Configure pruner
        pruner = self._create_pruner(config)
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=config.direction,
            sampler=sampler,
            pruner=pruner
        )
        
        self.logger.info(f"Setup hyperparameter optimization study: {study_name}")
    
    def _create_sampler(self, config: HyperparameterConfig):
        """Create Optuna sampler based on configuration."""
        if config.sampler == "tpe":
            return TPESampler(seed=config.random_state)
        elif config.sampler == "random":
            return RandomSampler(seed=config.random_state)
        elif config.sampler == "grid":
            # Grid sampler requires search space to be defined
            if not config.parameter_space:
                raise ConfigurationError("Grid sampler requires parameter_space to be defined")
            return GridSampler(config.parameter_space)
        else:
            raise ConfigurationError(f"Unsupported sampler: {config.sampler}")
    
    def _create_pruner(self, config: HyperparameterConfig):
        """Create Optuna pruner based on configuration."""
        if config.pruner == "median":
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        elif config.pruner == "successive_halving":
            return SuccessiveHalvingPruner()
        elif config.pruner == "none":
            return optuna.pruners.NopPruner()
        else:
            raise ConfigurationError(f"Unsupported pruner: {config.pruner}")
    
    def optimize(self, context: ExecutionContext) -> OptimizationResult:
        """Run hyperparameter optimization."""
        if self.study is None:
            raise ModelError("Optimization not setup. Call setup_optimization first.")
        
        start_time = datetime.now()
        
        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            return self._objective_function(trial, context)
        
        # Run optimization
        try:
            self.study.optimize(
                objective,
                n_trials=self.optimization_config.n_trials,
                timeout=self.optimization_config.timeout,
                callbacks=[self._logging_callback] if self.logger.isEnabledFor(logging.INFO) else None
            )
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Extract results
        best_trial = self.study.best_trial
        all_trials = []
        pruned_trials = 0
        failed_trials = 0
        
        for trial in self.study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            all_trials.append(trial_data)
            
            if trial.state == optuna.trial.TrialState.PRUNED:
                pruned_trials += 1
            elif trial.state == optuna.trial.TrialState.FAIL:
                failed_trials += 1
        
        result = OptimizationResult(
            best_params=best_trial.params,
            best_value=best_trial.value,
            best_trial_number=best_trial.number,
            n_trials=len(self.study.trials),
            optimization_time=optimization_time,
            study_name=self.study.study_name,
            all_trials=all_trials,
            pruned_trials=pruned_trials,
            failed_trials=failed_trials
        )
        
        self.logger.info(f"Optimization completed. Best {self.optimization_config.metric}: {best_trial.value:.4f}")
        self.logger.info(f"Best parameters: {best_trial.params}")
        
        return result
    
    def _objective_function(self, trial, context: ExecutionContext) -> float:
        """Objective function for Optuna optimization."""
        try:
            # Suggest hyperparameters
            suggested_params = self._suggest_parameters(trial)
            
            # Create model config with suggested parameters
            base_config = self.model_trainer.model_adapter.config
            optimized_config = ModelConfig(
                model_type=base_config.model_type,
                framework=base_config.framework,
                parameters={**base_config.parameters, **suggested_params},
                task_type=base_config.task_type,
                random_state=base_config.random_state
            )
            
            # Create temporary model adapter
            adapter_class = self.model_trainer.adapter_registry[optimized_config.framework]
            temp_adapter = adapter_class(optimized_config)
            
            # Perform cross-validation or single validation
            if self.optimization_config.cv_folds > 1:
                score = self._cross_validate(temp_adapter, trial)
            else:
                score = self._single_validation(temp_adapter, trial)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.TrialPruned()
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters based on model type and framework."""
        suggested_params = {}
        
        # Use predefined parameter space if available
        if self.optimization_config.parameter_space:
            for param_name, param_config in self.optimization_config.parameter_space.items():
                param_type = param_config.get('type', 'float')
                
                if param_type == 'float':
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'int':
                    suggested_params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
        else:
            # Use default parameter spaces based on model type
            suggested_params = self._get_default_parameter_space(trial)
        
        return suggested_params
    
    def _get_default_parameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get default parameter search space for different model types."""
        model_type = self.model_trainer.model_adapter.config.model_type
        framework = self.model_trainer.model_adapter.config.framework
        suggested_params = {}
        
        if framework == 'sklearn':
            if 'random_forest' in model_type:
                suggested_params.update({
                    'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                })
            elif 'gradient_boosting' in model_type:
                suggested_params.update({
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                })
            elif 'svm' in model_type:
                suggested_params.update({
                    'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('kernel', ['rbf', 'poly']) == 'rbf' else 'scale'
                })
            elif 'logistic_regression' in model_type:
                suggested_params.update({
                    'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
                })
        
        elif framework == 'xgboost':
            suggested_params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
            })
        
        elif framework == 'pytorch':
            suggested_params.update({
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'hidden_sizes': [trial.suggest_int(f'hidden_size_{i}', 32, 512) for i in range(trial.suggest_int('n_layers', 1, 3))]
            })
        
        return suggested_params
    
    def _cross_validate(self, model_adapter: ModelAdapter, trial: optuna.Trial) -> float:
        """Perform cross-validation for hyperparameter optimization."""
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # Prepare data
        X_train, y_train = self.model_trainer._split_features_target(self.training_data, {'training': {'target_column': 'target'}})
        
        # Setup cross-validation
        if model_adapter.config.task_type == 'classification':
            cv = StratifiedKFold(n_splits=self.optimization_config.cv_folds, shuffle=True, random_state=self.optimization_config.random_state)
        else:
            cv = KFold(n_splits=self.optimization_config.cv_folds, shuffle=True, random_state=self.optimization_config.random_state)
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            # Split data for this fold
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Create fresh model adapter for this fold
            adapter_class = self.model_trainer.adapter_registry[model_adapter.config.framework]
            fold_adapter = adapter_class(model_adapter.config)
            
            try:
                # Train model
                fold_adapter.fit(X_fold_train, y_fold_train)
                
                # Evaluate
                fold_score = self._evaluate_fold(fold_adapter, X_fold_val, y_fold_val)
                scores.append(fold_score)
                
                # Report intermediate value for pruning
                trial.report(fold_score, fold)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
            except Exception as e:
                self.logger.warning(f"Fold {fold} failed: {str(e)}")
                continue
        
        if not scores:
            raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    def _single_validation(self, model_adapter: ModelAdapter, trial: optuna.Trial) -> float:
        """Perform single validation split for hyperparameter optimization."""
        # Prepare data
        X_train, y_train = self.model_trainer._split_features_target(self.training_data, {'training': {'target_column': 'target'}})
        X_val, y_val = self.model_trainer._split_features_target(self.validation_data, {'training': {'target_column': 'target'}})
        
        try:
            # Train model
            model_adapter.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate
            score = self._evaluate_fold(model_adapter, X_val, y_val)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Validation failed: {str(e)}")
            raise optuna.TrialPruned()
    
    def _evaluate_fold(self, model_adapter: ModelAdapter, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Evaluate model performance for a single fold."""
        predictions = model_adapter.predict(X_val)
        
        metric_name = self.optimization_config.metric
        
        if model_adapter.config.task_type == 'classification':
            if metric_name == 'accuracy':
                from sklearn.metrics import accuracy_score
                return accuracy_score(y_val, predictions)
            elif metric_name == 'f1_score':
                from sklearn.metrics import f1_score
                return f1_score(y_val, predictions, average='weighted')
            elif metric_name == 'precision':
                from sklearn.metrics import precision_score
                return precision_score(y_val, predictions, average='weighted', zero_division=0)
            elif metric_name == 'recall':
                from sklearn.metrics import recall_score
                return recall_score(y_val, predictions, average='weighted', zero_division=0)
            elif metric_name == 'roc_auc':
                probabilities = model_adapter.predict_proba(X_val)
                if probabilities is not None and len(np.unique(y_val)) == 2:
                    from sklearn.metrics import roc_auc_score
                    if probabilities.shape[1] == 2:
                        return roc_auc_score(y_val, probabilities[:, 1])
                    else:
                        return roc_auc_score(y_val, probabilities)
                else:
                    return accuracy_score(y_val, predictions)  # Fallback
        else:
            # Regression metrics (note: these should be minimized, so return negative)
            if metric_name == 'mse':
                from sklearn.metrics import mean_squared_error
                return -mean_squared_error(y_val, predictions)
            elif metric_name == 'mae':
                from sklearn.metrics import mean_absolute_error
                return -mean_absolute_error(y_val, predictions)
            elif metric_name == 'r2_score':
                from sklearn.metrics import r2_score
                return r2_score(y_val, predictions)
            elif metric_name == 'rmse':
                from sklearn.metrics import mean_squared_error
                return -np.sqrt(mean_squared_error(y_val, predictions))
        
        raise ValueError(f"Unsupported metric: {metric_name}")
    
    def _logging_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback for logging optimization progress."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.logger.info(f"Trial {trial.number}: {self.optimization_config.metric}={trial.value:.4f}, params={trial.params}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            self.logger.debug(f"Trial {trial.number}: pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            self.logger.warning(f"Trial {trial.number}: failed")
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            return pd.DataFrame()
        
        trials_data = []
        for trial in self.study.trials:
            trial_data = {
                'trial_number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete,
                'duration_seconds': trial.duration.total_seconds() if trial.duration else None,
                **trial.params
            }
            trials_data.append(trial_data)
        
        return pd.DataFrame(trials_data)
    
    def save_study(self, path: str) -> None:
        """Save Optuna study to file."""
        if self.study is None:
            raise ModelError("No study to save")
        
        study_data = {
            'study_name': self.study.study_name,
            'direction': self.study.direction.name,
            'best_trial': {
                'number': self.study.best_trial.number,
                'value': self.study.best_trial.value,
                'params': self.study.best_trial.params
            },
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
                }
                for trial in self.study.trials
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(study_data, f, indent=2, default=str)
        
        self.logger.info(f"Study saved to {path}")


class HyperparameterOptimizedTrainer(ModelTrainer):
    """Extended ModelTrainer with hyperparameter optimization capabilities."""
    
    def __init__(self):
        super().__init__()
        self.hyperparameter_optimizer = None
        self.optimization_result = None
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration including hyperparameter optimization."""
        if not super().validate_config(config):
            return False
        
        # Validate hyperparameter optimization config if present
        hyperopt_config = config.get('hyperparameter_optimization', {})
        if hyperopt_config.get('enabled', False):
            if not OPTUNA_AVAILABLE:
                self.logger.error("Optuna is not installed but hyperparameter optimization is enabled")
                return False
            
            method = hyperopt_config.get('method', 'optuna')
            if method not in ['optuna', 'grid_search', 'random_search']:
                self.logger.error(f"Unsupported optimization method: {method}")
                return False
        
        return True
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup model training with optional hyperparameter optimization."""
        super().setup(context)
        
        # Setup hyperparameter optimization if enabled
        config = context.config.get('training', {})
        hyperopt_config = config.get('hyperparameter_optimization', {})
        
        if hyperopt_config.get('enabled', False):
            # Create hyperparameter configuration
            hp_config = HyperparameterConfig(
                method=hyperopt_config.get('method', 'optuna'),
                n_trials=hyperopt_config.get('n_trials', 50),
                timeout=hyperopt_config.get('timeout'),
                sampler=hyperopt_config.get('sampler', 'tpe'),
                pruner=hyperopt_config.get('pruner', 'median'),
                direction=hyperopt_config.get('direction', 'maximize'),
                metric=hyperopt_config.get('metric', 'accuracy'),
                early_stopping_rounds=hyperopt_config.get('early_stopping_rounds'),
                cv_folds=hyperopt_config.get('cv_folds', 5),
                random_state=hyperopt_config.get('random_state'),
                parameter_space=hyperopt_config.get('parameter_space', {})
            )
            
            # Create and setup optimizer
            self.hyperparameter_optimizer = HyperparameterOptimizer(self)
            self.hyperparameter_optimizer.setup_optimization(hp_config, context)
            
            self.logger.info("Hyperparameter optimization enabled")
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute model training with optional hyperparameter optimization."""
        # Setup the trainer if not already done
        if self.model_adapter is None:
            self.setup(context)
        
        if self.hyperparameter_optimizer is not None:
            return self._execute_with_optimization(context)
        else:
            return super().execute(context)
    
    def _execute_with_optimization(self, context: ExecutionContext) -> ExecutionResult:
        """Execute training with hyperparameter optimization."""
        try:
            start_time = datetime.now()
            
            # Run hyperparameter optimization
            self.logger.info("Starting hyperparameter optimization...")
            self.optimization_result = self.hyperparameter_optimizer.optimize(context)
            
            # Update model configuration with best parameters
            best_params = self.optimization_result.best_params
            self.model_adapter.config.parameters.update(best_params)
            
            # Recreate model adapter with optimized parameters
            adapter_class = self.adapter_registry[self.model_adapter.config.framework]
            self.model_adapter = adapter_class(self.model_adapter.config)
            
            self.logger.info(f"Optimization completed. Best parameters: {best_params}")
            
            # Train final model with best parameters on full training data
            self.logger.info("Training final model with optimized parameters...")
            result = super().execute(context)
            
            # Add optimization results to metadata
            if result.success:
                optimization_time = (datetime.now() - start_time).total_seconds()
                
                result.metadata.update({
                    'hyperparameter_optimization': True,
                    'optimization_trials': self.optimization_result.n_trials,
                    'optimization_time_seconds': optimization_time,
                    'best_hyperparameters': best_params,
                    'best_optimization_score': self.optimization_result.best_value,
                    'pruned_trials': self.optimization_result.pruned_trials,
                    'failed_trials': self.optimization_result.failed_trials
                })
                
                # Save optimization artifacts
                optimization_artifacts = self._save_optimization_artifacts(context)
                result.artifacts.extend(optimization_artifacts)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {str(e)}")
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def _save_optimization_artifacts(self, context: ExecutionContext) -> List[str]:
        """Save hyperparameter optimization artifacts."""
        artifacts = []
        artifacts_path = Path(context.artifacts_path)
        
        if self.optimization_result:
            # Save optimization results
            results_path = artifacts_path / "hyperparameter_optimization_results.json"
            with open(results_path, 'w') as f:
                # Convert optimization result to dict manually if it's not a dataclass
                if hasattr(self.optimization_result, '__dict__'):
                    result_dict = self.optimization_result.__dict__
                else:
                    result_dict = asdict(self.optimization_result)
                json.dump(result_dict, f, indent=2, default=str)
            artifacts.append(str(results_path))
            
            # Save optimization history
            history_df = self.hyperparameter_optimizer.get_optimization_history()
            if not history_df.empty:
                history_path = artifacts_path / "optimization_history.csv"
                history_df.to_csv(history_path, index=False)
                artifacts.append(str(history_path))
            
            # Save Optuna study
            study_path = artifacts_path / "optuna_study.json"
            self.hyperparameter_optimizer.save_study(str(study_path))
            artifacts.append(str(study_path))
        
        return artifacts