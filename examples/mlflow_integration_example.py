"""Example usage of MLflow integration for experiment tracking and model registry."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add the parent directory to the path to import mlpipeline
import sys
sys.path.append(str(Path(__file__).parent.parent))

from mlpipeline.models.mlflow_integration import (
    MLflowConfig, MLflowTracker, MLflowIntegratedTrainer, 
    MLflowIntegratedEvaluator, MLflowIntegratedHyperparameterTrainer
)
from mlpipeline.models.training import ModelConfig
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
from mlpipeline.config.manager import ConfigManager


def create_sample_data():
    """Create sample dataset for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic classification data
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Create target with some relationship to features
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df


def setup_sample_data_files(temp_dir: Path):
    """Setup sample data files for training."""
    data = create_sample_data()
    
    # Split data
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Save data files
    train_data.to_parquet(temp_dir / "train_preprocessed.parquet", index=False)
    val_data.to_parquet(temp_dir / "val_preprocessed.parquet", index=False)
    test_data.to_parquet(temp_dir / "test_preprocessed.parquet", index=False)
    
    return train_data, val_data, test_data


def example_basic_mlflow_tracking():
    """Example of basic MLflow tracking functionality."""
    print("=== Basic MLflow Tracking Example ===")
    
    # Configure MLflow
    mlflow_config = MLflowConfig(
        experiment_name="ml-pipeline-basic-example",
        log_params=True,
        log_metrics=True,
        log_artifacts=True,
        log_model=True,
        tags={'example': 'basic_tracking', 'version': '1.0'}
    )
    
    # Create MLflow tracker
    tracker = MLflowTracker(mlflow_config)
    
    # Start a run
    run_id = tracker.start_run(
        run_name="basic_example_run",
        tags={'model_type': 'sklearn', 'task': 'classification'}
    )
    
    print(f"Started MLflow run: {run_id}")
    
    # Log some parameters
    params = {
        'algorithm': 'random_forest',
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    tracker.log_params(params)
    
    # Log some metrics
    metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1_score': 0.85
    }
    tracker.log_metrics(metrics)
    
    # End the run
    tracker.end_run()
    
    # Get run information
    run_info = tracker.get_run_info(run_id)
    if run_info:
        print(f"Run completed: {run_info.run_name}")
        print(f"Status: {run_info.status}")
        print(f"Metrics: {run_info.metrics}")
    
    print("Basic MLflow tracking example completed!\n")


def example_integrated_training():
    """Example of integrated model training with MLflow."""
    print("=== Integrated Training with MLflow Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Setup sample data
        train_data, val_data, test_data = setup_sample_data_files(temp_path)
        print(f"Created sample data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
        
        # Configure MLflow
        mlflow_config = MLflowConfig(
            experiment_name="ml-pipeline-training-example",
            log_model=True,
            log_input_example=True,
            register_model=False,  # Set to True to register model
            model_name="sample_classifier",
            tags={'example': 'integrated_training'}
        )
        
        # Create integrated trainer
        trainer = MLflowIntegratedTrainer(mlflow_config)
        
        # Create execution context
        config = {
            'mlflow': {
                'enabled': True,
                'experiment_name': 'ml-pipeline-training-example',
                'register_model': False
            },
            'training': {
                'model': {
                    'framework': 'sklearn',
                    'model_type': 'random_forest_classifier',
                    'task_type': 'classification',
                    'parameters': {
                        'n_estimators': 50,
                        'max_depth': 8,
                        'random_state': 42
                    }
                },
                'target_column': 'target'
            }
        }
        
        context = ExecutionContext(
            experiment_id="example_experiment_001",
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config=config,
            artifacts_path=str(temp_path),
            logger=trainer.logger
        )
        
        # Execute training
        result = trainer.execute(context)
        
        if result.success:
            print("Training completed successfully!")
            print(f"MLflow run ID: {result.metadata.get('mlflow_run_id')}")
            print(f"Training metrics: {result.metrics}")
        else:
            print(f"Training failed: {result.error_message}")
    
    print("Integrated training example completed!\n")


def example_hyperparameter_optimization_with_mlflow():
    """Example of hyperparameter optimization with MLflow tracking."""
    print("=== Hyperparameter Optimization with MLflow Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Setup sample data
        train_data, val_data, test_data = setup_sample_data_files(temp_path)
        print(f"Created sample data for optimization")
        
        # Configure MLflow
        mlflow_config = MLflowConfig(
            experiment_name="ml-pipeline-hyperopt-example",
            log_model=True,
            register_model=False,
            tags={'example': 'hyperparameter_optimization'}
        )
        
        # Create integrated hyperparameter trainer
        trainer = MLflowIntegratedHyperparameterTrainer(mlflow_config)
        
        # Create execution context with hyperparameter optimization
        config = {
            'mlflow': {
                'enabled': True,
                'experiment_name': 'ml-pipeline-hyperopt-example'
            },
            'training': {
                'model': {
                    'framework': 'sklearn',
                    'model_type': 'random_forest_classifier',
                    'task_type': 'classification',
                    'parameters': {
                        'random_state': 42
                    }
                },
                'target_column': 'target',
                'hyperparameter_optimization': {
                    'enabled': True,
                    'method': 'optuna',
                    'n_trials': 10,  # Small number for example
                    'metric': 'accuracy',
                    'direction': 'maximize',
                    'cv_folds': 3,
                    'parameter_space': {
                        'n_estimators': {
                            'type': 'int',
                            'low': 10,
                            'high': 100
                        },
                        'max_depth': {
                            'type': 'int',
                            'low': 3,
                            'high': 15
                        },
                        'min_samples_split': {
                            'type': 'int',
                            'low': 2,
                            'high': 10
                        }
                    }
                }
            }
        }
        
        context = ExecutionContext(
            experiment_id="example_hyperopt_001",
            stage_name="hyperparameter_optimization",
            component_type=ComponentType.MODEL_TRAINING,
            config=config,
            artifacts_path=str(temp_path),
            logger=trainer.logger
        )
        
        # Execute hyperparameter optimization
        result = trainer.execute(context)
        
        if result.success:
            print("Hyperparameter optimization completed successfully!")
            print(f"MLflow parent run ID: {result.metadata.get('mlflow_parent_run_id')}")
            print(f"Number of child runs: {len(result.metadata.get('mlflow_child_runs', []))}")
            print(f"Best hyperparameters: {result.metadata.get('best_hyperparameters')}")
            print(f"Final metrics: {result.metrics}")
        else:
            print(f"Hyperparameter optimization failed: {result.error_message}")
    
    print("Hyperparameter optimization example completed!\n")


def example_model_registry():
    """Example of model registry functionality."""
    print("=== Model Registry Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Setup sample data
        setup_sample_data_files(temp_path)
        
        # Configure MLflow with model registry
        mlflow_config = MLflowConfig(
            experiment_name="ml-pipeline-registry-example",
            log_model=True,
            register_model=True,
            model_name="sample_production_classifier",
            model_stage="Staging",
            model_description="Sample classifier for production deployment",
            tags={'example': 'model_registry', 'environment': 'staging'}
        )
        
        # Create integrated trainer
        trainer = MLflowIntegratedTrainer(mlflow_config)
        
        # Create execution context
        config = {
            'mlflow': {
                'enabled': True,
                'experiment_name': 'ml-pipeline-registry-example',
                'register_model': True,
                'model_name': 'sample_production_classifier'
            },
            'training': {
                'model': {
                    'framework': 'sklearn',
                    'model_type': 'random_forest_classifier',
                    'task_type': 'classification',
                    'parameters': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'random_state': 42
                    }
                },
                'target_column': 'target'
            }
        }
        
        context = ExecutionContext(
            experiment_id="example_registry_001",
            stage_name="training_for_registry",
            component_type=ComponentType.MODEL_TRAINING,
            config=config,
            artifacts_path=str(temp_path),
            logger=trainer.logger
        )
        
        # Execute training with model registration
        result = trainer.execute(context)
        
        if result.success:
            print("Training and model registration completed successfully!")
            print(f"MLflow run ID: {result.metadata.get('mlflow_run_id')}")
            print(f"Model version: {result.metadata.get('mlflow_model_version')}")
        else:
            print(f"Training with registry failed: {result.error_message}")
    
    print("Model registry example completed!\n")


def example_run_comparison():
    """Example of comparing multiple MLflow runs."""
    print("=== Run Comparison Example ===")
    
    # Configure MLflow
    mlflow_config = MLflowConfig(
        experiment_name="ml-pipeline-comparison-example",
        tags={'example': 'run_comparison'}
    )
    
    # Create MLflow tracker
    tracker = MLflowTracker(mlflow_config)
    
    # Create multiple runs with different parameters
    run_ids = []
    
    for i, params in enumerate([
        {'algorithm': 'random_forest', 'n_estimators': 50, 'max_depth': 5},
        {'algorithm': 'random_forest', 'n_estimators': 100, 'max_depth': 10},
        {'algorithm': 'gradient_boosting', 'n_estimators': 50, 'learning_rate': 0.1}
    ]):
        run_id = tracker.start_run(run_name=f"comparison_run_{i+1}")
        run_ids.append(run_id)
        
        # Log parameters
        tracker.log_params(params)
        
        # Simulate different performance metrics
        metrics = {
            'accuracy': 0.80 + i * 0.02 + np.random.random() * 0.05,
            'f1_score': 0.78 + i * 0.03 + np.random.random() * 0.04,
            'training_time': 10 + i * 5 + np.random.random() * 3
        }
        tracker.log_metrics(metrics)
        
        tracker.end_run()
    
    print(f"Created {len(run_ids)} runs for comparison")
    
    # Compare runs
    comparison_df = tracker.compare_runs(run_ids)
    if not comparison_df.empty:
        print("\nRun Comparison:")
        print(comparison_df[['run_name', 'algorithm', 'n_estimators', 'accuracy', 'f1_score']].to_string(index=False))
    
    # Search for runs
    runs = tracker.search_runs(filter_string="metrics.accuracy > 0.8", max_results=10)
    print(f"\nFound {len(runs)} runs with accuracy > 0.8")
    
    print("Run comparison example completed!\n")


def main():
    """Run all MLflow integration examples."""
    print("MLflow Integration Examples")
    print("=" * 50)
    
    try:
        # Check if MLflow is available
        import mlflow
        print(f"MLflow version: {mlflow.__version__}")
        print()
        
        # Run examples
        example_basic_mlflow_tracking()
        example_integrated_training()
        
        # Check if Optuna is available for hyperparameter optimization
        try:
            import optuna
            example_hyperparameter_optimization_with_mlflow()
        except ImportError:
            print("Skipping hyperparameter optimization example (Optuna not installed)")
        
        example_model_registry()
        example_run_comparison()
        
        print("All MLflow integration examples completed successfully!")
        print("\nTo view the results:")
        print("1. Run 'mlflow ui' in your terminal")
        print("2. Open http://localhost:5000 in your browser")
        print("3. Browse the experiments and runs created by these examples")
        
    except ImportError as e:
        print(f"MLflow is not installed: {e}")
        print("Please install MLflow with: pip install mlflow")
    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    main()