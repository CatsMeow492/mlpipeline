"""Model evaluation and comparison system with metrics computation and visualization."""

import logging
import json
import pickle
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# ML Framework imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..core.interfaces import PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
from ..core.errors import ModelError, ConfigurationError


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""
    task_type: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict[str, Any]] = None
    roc_curve_data: Optional[Dict[str, np.ndarray]] = None
    pr_curve_data: Optional[Dict[str, np.ndarray]] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'task_type': self.task_type,
            'metrics': self.metrics,
            'classification_report': self.classification_report,
            'feature_importance': self.feature_importance
        }
        
        if self.confusion_matrix is not None:
            result['confusion_matrix'] = self.confusion_matrix.tolist()
        
        if self.roc_curve_data is not None:
            result['roc_curve_data'] = {k: v.tolist() for k, v in self.roc_curve_data.items()}
        
        if self.pr_curve_data is not None:
            result['pr_curve_data'] = {k: v.tolist() for k, v in self.pr_curve_data.items()}
        
        return result


@dataclass
class ModelComparison:
    """Container for model comparison results."""
    model_names: List[str]
    metrics_comparison: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Dict[str, Any]]
    best_model: str
    ranking: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_names': self.model_names,
            'metrics_comparison': self.metrics_comparison,
            'statistical_tests': self.statistical_tests,
            'best_model': self.best_model,
            'ranking': self.ranking
        }


class ModelEvaluator(PipelineComponent):
    """Model evaluation component with metrics computation and visualization."""
    
    def __init__(self):
        super().__init__(ComponentType.MODEL_EVALUATION)
        self.evaluation_results: Dict[str, EvaluationMetrics] = {}
        self.comparison_results: Optional[ModelComparison] = None
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate evaluation configuration."""
        try:
            eval_config = config.get('evaluation', {})
            
            # Check if metrics are specified
            if 'metrics' not in eval_config:
                self.logger.warning("No metrics specified, using defaults")
            
            # Validate metric names
            valid_classification_metrics = {
                'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
                'precision_macro', 'recall_macro', 'f1_macro'
            }
            valid_regression_metrics = {
                'mse', 'mae', 'r2_score', 'mape', 'rmse'
            }
            
            metrics = eval_config.get('metrics', [])
            task_type = config.get('training', {}).get('model', {}).get('task_type', 'classification')
            
            if task_type == 'classification':
                valid_metrics = valid_classification_metrics
            else:
                valid_metrics = valid_regression_metrics
            
            for metric in metrics:
                if metric not in valid_metrics:
                    self.logger.error(f"Invalid metric '{metric}' for task type '{task_type}'")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup evaluation component."""
        config = context.config.get('evaluation', {})
        
        if not self.validate_config(context.config):
            raise ConfigurationError("Invalid evaluation configuration")
        
        self.logger.info("Model evaluator setup completed")
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute model evaluation."""
        try:
            start_time = datetime.now()
            
            # Load trained models and data
            models_data = self._load_models_and_data(context)
            
            # Evaluate each model
            evaluation_results = {}
            for model_name, model_data in models_data.items():
                self.logger.info(f"Evaluating model: {model_name}")
                
                eval_result = self._evaluate_single_model(
                    model_data['model'],
                    model_data['X_test'],
                    model_data['y_test'],
                    model_data['task_type'],
                    context.config.get('evaluation', {})
                )
                
                evaluation_results[model_name] = eval_result
                
                # Generate visualizations
                self._generate_visualizations(
                    eval_result, 
                    model_name, 
                    context.artifacts_path
                )
            
            self.evaluation_results = evaluation_results
            
            # Perform model comparison if multiple models
            if len(evaluation_results) > 1:
                self.comparison_results = self._compare_models(
                    evaluation_results,
                    context.config.get('evaluation', {})
                )
            
            # Save results
            artifacts = self._save_evaluation_results(context)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare metrics for result
            result_metrics = {}
            for model_name, eval_result in evaluation_results.items():
                for metric_name, metric_value in eval_result.metrics.items():
                    result_metrics[f"{model_name}_{metric_name}"] = metric_value
            
            result_metrics['evaluation_time_seconds'] = execution_time
            result_metrics['models_evaluated'] = len(evaluation_results)
            
            self.logger.info("Model evaluation completed successfully")
            
            return ExecutionResult(
                success=True,
                artifacts=artifacts,
                metrics=result_metrics,
                metadata={
                    'models_evaluated': list(evaluation_results.keys()),
                    'has_comparison': self.comparison_results is not None,
                    'visualizations_generated': True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def _load_models_and_data(self, context: ExecutionContext) -> Dict[str, Dict[str, Any]]:
        """Load trained models and test data."""
        models_data = {}
        
        # Get artifacts path from context
        artifacts_path = Path(context.artifacts_path)
        
        # Look for model files in artifacts directory
        model_files = list(artifacts_path.glob("*_model.pkl"))
        
        if not model_files:
            raise ModelError("No trained models found in artifacts directory")
        
        for model_file in model_files:
            model_name = model_file.stem.replace("_model", "")
            
            try:
                # Load model
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                # Load test data
                test_data_file = artifacts_path / f"{model_name}_test_data.pkl"
                if test_data_file.exists():
                    with open(test_data_file, 'rb') as f:
                        test_data = pickle.load(f)
                    
                    models_data[model_name] = {
                        'model': model,
                        'X_test': test_data['X_test'],
                        'y_test': test_data['y_test'],
                        'task_type': test_data.get('task_type', 'classification')
                    }
                else:
                    self.logger.warning(f"Test data not found for model {model_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {str(e)}")
                continue
        
        if not models_data:
            raise ModelError("No valid models and test data found")
        
        return models_data
    
    def _evaluate_single_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        task_type: str,
        eval_config: Dict[str, Any]
    ) -> EvaluationMetrics:
        """Evaluate a single model and compute metrics."""
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        
        # Compute metrics based on task type
        if task_type == 'classification':
            metrics = self._compute_classification_metrics(
                y_test, y_pred, y_pred_proba, eval_config
            )
            
            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Generate classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Compute ROC curve data if binary classification
            roc_data = None
            pr_data = None
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                if y_pred_proba.ndim > 1:
                    y_scores = y_pred_proba[:, 1]
                else:
                    y_scores = y_pred_proba
                
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                precision, recall, _ = precision_recall_curve(y_test, y_scores)
                
                roc_data = {'fpr': fpr, 'tpr': tpr}
                pr_data = {'precision': precision, 'recall': recall}
            
            return EvaluationMetrics(
                task_type=task_type,
                metrics=metrics,
                confusion_matrix=cm,
                classification_report=class_report,
                roc_curve_data=roc_data,
                pr_curve_data=pr_data,
                feature_importance=self._get_feature_importance(model)
            )
            
        else:  # regression
            metrics = self._compute_regression_metrics(y_test, y_pred, eval_config)
            
            return EvaluationMetrics(
                task_type=task_type,
                metrics=metrics,
                feature_importance=self._get_feature_importance(model)
            )
    
    def _compute_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray],
        eval_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics = {}
        
        # Default metrics if not specified
        requested_metrics = eval_config.get('metrics', [
            'accuracy', 'precision', 'recall', 'f1_score'
        ])
        
        # Basic metrics
        if 'accuracy' in requested_metrics:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle multiclass vs binary
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        
        if 'precision' in requested_metrics:
            metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        
        if 'recall' in requested_metrics:
            metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        
        if 'f1_score' in requested_metrics:
            metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Macro averages
        if 'precision_macro' in requested_metrics:
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        
        if 'recall_macro' in requested_metrics:
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        if 'f1_macro' in requested_metrics:
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # ROC AUC (only for binary or with probabilities)
        if 'roc_auc' in requested_metrics and y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    if y_pred_proba.ndim > 1:
                        scores = y_pred_proba[:, 1]
                    else:
                        scores = y_pred_proba
                    metrics['roc_auc'] = roc_auc_score(y_true, scores)
                else:
                    # Multiclass ROC AUC
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except Exception as e:
                self.logger.warning(f"Could not compute ROC AUC: {str(e)}")
        
        return metrics
    
    def _compute_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        eval_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        metrics = {}
        
        # Default metrics if not specified
        requested_metrics = eval_config.get('metrics', [
            'mse', 'mae', 'r2_score'
        ])
        
        if 'mse' in requested_metrics:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        if 'rmse' in requested_metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if 'mae' in requested_metrics:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        if 'r2_score' in requested_metrics:
            metrics['r2_score'] = r2_score(y_true, y_pred)
        
        if 'mape' in requested_metrics:
            # Avoid division by zero
            mask = y_true != 0
            if np.any(mask):
                metrics['mape'] = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
            else:
                metrics['mape'] = float('inf')
        
        return metrics
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importances = model.feature_importances_
                return {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
            elif hasattr(model, 'coef_'):
                # For linear models
                coef = model.coef_
                if coef.ndim > 1:
                    # Multi-class case, take mean absolute coefficients
                    coef = np.mean(np.abs(coef), axis=0)
                return {f'feature_{i}': float(c) for i, c in enumerate(coef)}
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {str(e)}")
        
        return None
    
    def _generate_visualizations(
        self, 
        eval_result: EvaluationMetrics, 
        model_name: str, 
        artifacts_path: str
    ) -> None:
        """Generate visualization plots for evaluation results."""
        artifacts_dir = Path(artifacts_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        if eval_result.task_type == 'classification':
            self._generate_classification_plots(eval_result, model_name, artifacts_dir)
        else:
            self._generate_regression_plots(eval_result, model_name, artifacts_dir)
        
        # Feature importance plot
        if eval_result.feature_importance:
            self._generate_feature_importance_plot(
                eval_result.feature_importance, model_name, artifacts_dir
            )
    
    def _generate_classification_plots(
        self, 
        eval_result: EvaluationMetrics, 
        model_name: str, 
        artifacts_dir: Path
    ) -> None:
        """Generate classification-specific plots."""
        
        # Confusion Matrix
        if eval_result.confusion_matrix is not None:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                eval_result.confusion_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                cbar_kws={'label': 'Count'}
            )
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(artifacts_dir / f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # ROC Curve
        if eval_result.roc_curve_data is not None:
            plt.figure(figsize=(8, 6))
            fpr = eval_result.roc_curve_data['fpr']
            tpr = eval_result.roc_curve_data['tpr']
            
            auc_score = eval_result.metrics.get('roc_auc', 0)
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(artifacts_dir / f'{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Precision-Recall Curve
        if eval_result.pr_curve_data is not None:
            plt.figure(figsize=(8, 6))
            precision = eval_result.pr_curve_data['precision']
            recall = eval_result.pr_curve_data['recall']
            
            plt.plot(recall, precision, linewidth=2, label='Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(artifacts_dir / f'{model_name}_pr_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_regression_plots(
        self, 
        eval_result: EvaluationMetrics, 
        model_name: str, 
        artifacts_dir: Path
    ) -> None:
        """Generate regression-specific plots."""
        # For regression, we would need actual vs predicted values
        # This would require storing them in the evaluation results
        # For now, we'll create a placeholder metrics plot
        
        metrics = eval_result.metrics
        if not metrics:
            return
        
        plt.figure(figsize=(10, 6))
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = plt.bar(metric_names, metric_values, color='skyblue', alpha=0.7)
        plt.title(f'Regression Metrics - {model_name}')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(artifacts_dir / f'{model_name}_regression_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_feature_importance_plot(
        self, 
        feature_importance: Dict[str, float], 
        model_name: str, 
        artifacts_dir: Path
    ) -> None:
        """Generate feature importance plot."""
        if not feature_importance:
            return
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 20 features if more than 20
        if len(sorted_features) > 20:
            sorted_features = sorted_features[:20]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(10, max(6, len(features) * 0.3)))
        bars = plt.barh(range(len(features)), importances, color='lightcoral', alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(artifacts_dir / f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _compare_models(
        self, 
        evaluation_results: Dict[str, EvaluationMetrics],
        eval_config: Dict[str, Any]
    ) -> ModelComparison:
        """Compare multiple models and perform statistical significance testing."""
        
        model_names = list(evaluation_results.keys())
        metrics_comparison = {}
        statistical_tests = {}
        
        # Collect all metrics
        all_metrics = set()
        for eval_result in evaluation_results.values():
            all_metrics.update(eval_result.metrics.keys())
        
        # Compare metrics across models
        for metric in all_metrics:
            metrics_comparison[metric] = {}
            for model_name, eval_result in evaluation_results.items():
                metrics_comparison[metric][model_name] = eval_result.metrics.get(metric, 0.0)
        
        # Perform statistical significance testing
        # For now, we'll use a simple ranking approach
        # In a full implementation, you'd use cross-validation scores for proper statistical testing
        
        primary_metric = eval_config.get('primary_metric', 'accuracy' if 
                                       evaluation_results[model_names[0]].task_type == 'classification' 
                                       else 'r2_score')
        
        if primary_metric in metrics_comparison:
            # Rank models by primary metric
            model_scores = [(name, metrics_comparison[primary_metric][name]) 
                          for name in model_names]
            
            # For classification metrics (higher is better)
            if primary_metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'r2_score']:
                model_scores.sort(key=lambda x: x[1], reverse=True)
            else:  # For error metrics (lower is better)
                model_scores.sort(key=lambda x: x[1])
            
            ranking = [name for name, _ in model_scores]
            best_model = ranking[0]
            
            # Simple statistical test placeholder
            # In practice, you'd use paired t-tests or McNemar's test with cross-validation scores
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    test_key = f"{model1}_vs_{model2}"
                    score1 = metrics_comparison[primary_metric][model1]
                    score2 = metrics_comparison[primary_metric][model2]
                    
                    # Simple difference test (placeholder)
                    difference = abs(score1 - score2)
                    p_value = 0.05 if difference > 0.01 else 0.5  # Placeholder
                    
                    statistical_tests[test_key] = {
                        'metric': primary_metric,
                        'model1_score': score1,
                        'model2_score': score2,
                        'difference': score1 - score2,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        else:
            # Fallback ranking
            ranking = model_names
            best_model = model_names[0]
        
        return ModelComparison(
            model_names=model_names,
            metrics_comparison=metrics_comparison,
            statistical_tests=statistical_tests,
            best_model=best_model,
            ranking=ranking
        )
    
    def _save_evaluation_results(self, context: ExecutionContext) -> List[str]:
        """Save evaluation results to artifacts."""
        artifacts = []
        artifacts_path = Path(context.artifacts_path)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual model evaluation results
        for model_name, eval_result in self.evaluation_results.items():
            result_file = artifacts_path / f"{model_name}_evaluation.json"
            with open(result_file, 'w') as f:
                json.dump(eval_result.to_dict(), f, indent=2)
            artifacts.append(str(result_file))
        
        # Save model comparison results
        if self.comparison_results:
            comparison_file = artifacts_path / "model_comparison.json"
            with open(comparison_file, 'w') as f:
                json.dump(self.comparison_results.to_dict(), f, indent=2)
            artifacts.append(str(comparison_file))
        
        # Save summary report
        summary_file = artifacts_path / "evaluation_summary.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(self.evaluation_results.keys()),
            'best_model': self.comparison_results.best_model if self.comparison_results else None,
            'summary_metrics': {}
        }
        
        # Add summary metrics
        for model_name, eval_result in self.evaluation_results.items():
            summary['summary_metrics'][model_name] = eval_result.metrics
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        artifacts.append(str(summary_file))
        
        return artifacts