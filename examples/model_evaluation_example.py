#!/usr/bin/env python3
"""
Example demonstrating the model evaluation and comparison system.

This example shows how to:
1. Set up trained models and test data
2. Configure the model evaluator
3. Run evaluation with metrics computation
4. Generate visualizations and comparison reports
5. Analyze the results
"""

import os
import sys
import tempfile
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from unittest.mock import Mock

# Add the parent directory to the path so we can import mlpipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mlpipeline.models.evaluation import ModelEvaluator
from mlpipeline.core.interfaces import ExecutionContext, ComponentType


def create_sample_data():
    """Create sample classification dataset."""
    print("Creating sample classification dataset...")
    
    # Generate a multi-class classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Dataset created: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """Train multiple models for comparison."""
    print("\nTraining multiple models...")
    
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            multi_class='ovr'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        ),
        'svm': SVC(
            random_state=42,
            probability=True,  # Enable probability estimates for ROC curves
            kernel='rbf',
            C=1.0
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Quick accuracy check
        train_accuracy = model.score(X_train, y_train)
        print(f"    Training accuracy: {train_accuracy:.3f}")
    
    return trained_models


def setup_evaluation_environment(trained_models, X_test, y_test):
    """Set up the evaluation environment with models and test data."""
    print("\nSetting up evaluation environment...")
    
    # Create temporary directory for artifacts
    temp_dir = tempfile.mkdtemp()
    artifacts_path = Path(temp_dir)
    
    print(f"Artifacts directory: {temp_dir}")
    
    # Save trained models
    for name, model in trained_models.items():
        model_file = artifacts_path / f"{name}_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved {name} model")
        
        # Save test data for each model
        test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'task_type': 'classification'
        }
        test_data_file = artifacts_path / f"{name}_test_data.pkl"
        with open(test_data_file, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"  Saved {name} test data")
    
    return temp_dir


def run_evaluation(artifacts_path):
    """Run the model evaluation process."""
    print("\nRunning model evaluation...")
    
    # Create evaluation configuration
    config = {
        'evaluation': {
            'metrics': [
                'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
                'precision_macro', 'recall_macro', 'f1_macro'
            ],
            'primary_metric': 'f1_macro'  # Use macro F1 for multi-class
        },
        'training': {
            'model': {
                'task_type': 'classification'
            }
        }
    }
    
    # Create execution context
    context = ExecutionContext(
        experiment_id="evaluation_example",
        stage_name="evaluation",
        component_type=ComponentType.MODEL_EVALUATION,
        config=config,
        artifacts_path=artifacts_path,
        logger=Mock(),
        metadata={}
    )
    
    # Initialize and run evaluator
    evaluator = ModelEvaluator()
    result = evaluator.execute(context)
    
    if result.success:
        print("✓ Evaluation completed successfully!")
        print(f"  Models evaluated: {result.metrics['models_evaluated']}")
        print(f"  Evaluation time: {result.metrics['evaluation_time_seconds']:.2f} seconds")
        print(f"  Artifacts created: {len(result.artifacts)}")
    else:
        print(f"✗ Evaluation failed: {result.error_message}")
        return None
    
    return evaluator, result


def analyze_results(evaluator, artifacts_path):
    """Analyze and display the evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS ANALYSIS")
    print("="*60)
    
    # Display individual model results
    print("\nIndividual Model Performance:")
    print("-" * 40)
    
    for model_name, eval_result in evaluator.evaluation_results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        
        # Display key metrics
        metrics = eval_result.metrics
        print(f"  Accuracy:     {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision:    {metrics.get('precision', 0):.4f}")
        print(f"  Recall:       {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:     {metrics.get('f1_score', 0):.4f}")
        print(f"  ROC AUC:      {metrics.get('roc_auc', 0):.4f}")
        print(f"  F1 Macro:     {metrics.get('f1_macro', 0):.4f}")
        
        # Display confusion matrix shape
        if eval_result.confusion_matrix is not None:
            cm_shape = eval_result.confusion_matrix.shape
            print(f"  Confusion Matrix: {cm_shape[0]}x{cm_shape[1]}")
    
    # Display model comparison
    if evaluator.comparison_results:
        print("\nModel Comparison:")
        print("-" * 40)
        
        comparison = evaluator.comparison_results
        print(f"Best Model: {comparison.best_model}")
        print(f"Ranking: {' > '.join(comparison.ranking)}")
        
        # Display statistical tests
        print("\nStatistical Significance Tests:")
        for test_name, test_result in comparison.statistical_tests.items():
            models = test_name.split('_vs_')
            diff = test_result['difference']
            p_val = test_result['p_value']
            significant = test_result['significant']
            
            print(f"  {models[0]} vs {models[1]}:")
            print(f"    Difference: {diff:+.4f}")
            print(f"    P-value: {p_val:.4f}")
            print(f"    Significant: {'Yes' if significant else 'No'}")
    
    # List generated artifacts
    print("\nGenerated Artifacts:")
    print("-" * 40)
    
    artifacts_dir = Path(artifacts_path)
    
    # Evaluation reports
    eval_files = list(artifacts_dir.glob("*_evaluation.json"))
    print(f"  Evaluation Reports: {len(eval_files)}")
    for file in eval_files:
        print(f"    - {file.name}")
    
    # Visualizations
    viz_files = list(artifacts_dir.glob("*.png"))
    print(f"  Visualizations: {len(viz_files)}")
    for file in sorted(viz_files):
        print(f"    - {file.name}")
    
    # Other files
    other_files = [f for f in artifacts_dir.glob("*.json") 
                   if not f.name.endswith("_evaluation.json")]
    if other_files:
        print(f"  Other Reports: {len(other_files)}")
        for file in other_files:
            print(f"    - {file.name}")


def demonstrate_configuration_options():
    """Show different configuration options for evaluation."""
    print("\n" + "="*60)
    print("CONFIGURATION OPTIONS")
    print("="*60)
    
    print("\nClassification Metrics:")
    classification_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
        'precision_macro', 'recall_macro', 'f1_macro'
    ]
    for metric in classification_metrics:
        print(f"  - {metric}")
    
    print("\nRegression Metrics:")
    regression_metrics = [
        'mse', 'rmse', 'mae', 'r2_score', 'mape'
    ]
    for metric in regression_metrics:
        print(f"  - {metric}")
    
    print("\nExample Configuration:")
    example_config = {
        'evaluation': {
            'metrics': ['accuracy', 'f1_macro', 'roc_auc'],
            'primary_metric': 'f1_macro'
        }
    }
    
    import json
    print(json.dumps(example_config, indent=2))


def main():
    """Main example execution."""
    print("Model Evaluation and Comparison Example")
    print("="*50)
    
    try:
        # Step 1: Create sample data
        X_train, X_test, y_train, y_test = create_sample_data()
        
        # Step 2: Train multiple models
        trained_models = train_models(X_train, y_train)
        
        # Step 3: Set up evaluation environment
        artifacts_path = setup_evaluation_environment(trained_models, X_test, y_test)
        
        # Step 4: Run evaluation
        evaluator, result = run_evaluation(artifacts_path)
        
        if evaluator and result:
            # Step 5: Analyze results
            analyze_results(evaluator, artifacts_path)
            
            # Step 6: Show configuration options
            demonstrate_configuration_options()
            
            print(f"\n✓ Example completed successfully!")
            print(f"Check the artifacts directory for detailed results: {artifacts_path}")
            print("\nGenerated files include:")
            print("  - Individual model evaluation reports (JSON)")
            print("  - Model comparison report (JSON)")
            print("  - Confusion matrices (PNG)")
            print("  - ROC curves (PNG)")
            print("  - Feature importance plots (PNG)")
        
    except Exception as e:
        print(f"\n✗ Example failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()