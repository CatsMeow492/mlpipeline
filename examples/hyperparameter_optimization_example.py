"""Example demonstrating hyperparameter optimization with the ML pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from mlpipeline.models.hyperparameter_optimization import (
    HyperparameterOptimizedTrainer,
    HyperparameterConfig
)
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
import logging


def create_sample_data():
    """Create sample dataset for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some pattern
    y = (X[:, 0] + X[:, 1] - X[:, 2] + 0.5 * X[:, 3] > 0).astype(int)
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    return df


def prepare_data_splits(data, artifacts_path):
    """Prepare train/validation/test splits."""
    # Simple split: 60% train, 20% validation, 20% test
    n_samples = len(data)
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    # Save to parquet files
    train_data.to_parquet(artifacts_path / "train_preprocessed.parquet")
    val_data.to_parquet(artifacts_path / "val_preprocessed.parquet")
    test_data.to_parquet(artifacts_path / "test_preprocessed.parquet")
    
    print(f"Data splits created:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")


def example_sklearn_random_forest_optimization():
    """Example: Optimize Random Forest hyperparameters."""
    print("=" * 60)
    print("Random Forest Hyperparameter Optimization Example")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_path = Path(temp_dir)
        
        # Prepare data
        prepare_data_splits(data, artifacts_path)
        
        # Create execution context
        context = ExecutionContext(
            experiment_id="rf_optimization_example",
            stage_name="model_training",
            component_type=ComponentType.MODEL_TRAINING,
            artifacts_path=str(artifacts_path),
            logger=logging.getLogger("example"),
            config={
                'training': {
                    'target_column': 'target',
                    'model': {
                        'framework': 'sklearn',
                        'model_type': 'random_forest_classifier',
                        'task_type': 'classification',
                        'parameters': {
                            'random_state': 42
                        }
                    },
                    'hyperparameter_optimization': {
                        'enabled': True,
                        'method': 'optuna',
                        'n_trials': 20,
                        'timeout': 60,  # 1 minute timeout
                        'sampler': 'tpe',
                        'pruner': 'median',
                        'direction': 'maximize',
                        'metric': 'accuracy',
                        'cv_folds': 3,
                        'random_state': 42,
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
                                'high': 20
                            },
                            'min_samples_leaf': {
                                'type': 'int',
                                'low': 1,
                                'high': 10
                            },
                            'max_features': {
                                'type': 'categorical',
                                'choices': ['sqrt', 'log2', None]
                            }
                        }
                    }
                }
            }
        )
        
        # Create and run optimized trainer
        trainer = HyperparameterOptimizedTrainer()
        result = trainer.execute(context)
        
        if result.success:
            print("\n✅ Optimization completed successfully!")
            print(f"Training time: {result.metrics.get('training_time_seconds', 0):.2f} seconds")
            print(f"Optimization trials: {result.metadata.get('optimization_trials', 0)}")
            print(f"Best score: {result.metadata.get('best_optimization_score', 0):.4f}")
            print(f"Best hyperparameters: {result.metadata.get('best_hyperparameters', {})}")
            
            # Print final model performance
            print(f"\nFinal model performance:")
            print(f"  Train accuracy: {result.metrics.get('train_accuracy', 0):.4f}")
            print(f"  Validation accuracy: {result.metrics.get('val_accuracy', 0):.4f}")
            print(f"  Test accuracy: {result.metrics.get('test_accuracy', 0):.4f}")
            
            # Show optimization artifacts
            print(f"\nOptimization artifacts created:")
            for artifact in result.artifacts:
                if 'optimization' in artifact or 'optuna' in artifact:
                    print(f"  - {Path(artifact).name}")
        else:
            print(f"❌ Optimization failed: {result.error_message}")


def example_xgboost_optimization():
    """Example: Optimize XGBoost hyperparameters."""
    print("\n" + "=" * 60)
    print("XGBoost Hyperparameter Optimization Example")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_path = Path(temp_dir)
        
        # Prepare data
        prepare_data_splits(data, artifacts_path)
        
        # Create execution context
        context = ExecutionContext(
            experiment_id="xgb_optimization_example",
            stage_name="model_training",
            component_type=ComponentType.MODEL_TRAINING,
            artifacts_path=str(artifacts_path),
            logger=logging.getLogger("example"),
            config={
                'training': {
                    'target_column': 'target',
                    'model': {
                        'framework': 'xgboost',
                        'model_type': 'xgb_classifier',
                        'task_type': 'classification',
                        'parameters': {
                            'random_state': 42,
                            'eval_metric': 'logloss'
                        }
                    },
                    'hyperparameter_optimization': {
                        'enabled': True,
                        'method': 'optuna',
                        'n_trials': 15,
                        'timeout': 45,  # 45 seconds timeout
                        'sampler': 'tpe',
                        'pruner': 'median',
                        'direction': 'maximize',
                        'metric': 'accuracy',
                        'cv_folds': 3,
                        'random_state': 42,
                        'parameter_space': {
                            'n_estimators': {
                                'type': 'int',
                                'low': 50,
                                'high': 200
                            },
                            'learning_rate': {
                                'type': 'float',
                                'low': 0.01,
                                'high': 0.3,
                                'log': True
                            },
                            'max_depth': {
                                'type': 'int',
                                'low': 3,
                                'high': 10
                            },
                            'subsample': {
                                'type': 'float',
                                'low': 0.6,
                                'high': 1.0
                            },
                            'colsample_bytree': {
                                'type': 'float',
                                'low': 0.6,
                                'high': 1.0
                            },
                            'reg_alpha': {
                                'type': 'float',
                                'low': 1e-8,
                                'high': 1.0,
                                'log': True
                            },
                            'reg_lambda': {
                                'type': 'float',
                                'low': 1e-8,
                                'high': 1.0,
                                'log': True
                            }
                        }
                    }
                }
            }
        )
        
        # Create and run optimized trainer
        trainer = HyperparameterOptimizedTrainer()
        result = trainer.execute(context)
        
        if result.success:
            print("\n✅ XGBoost optimization completed successfully!")
            print(f"Training time: {result.metrics.get('training_time_seconds', 0):.2f} seconds")
            print(f"Optimization trials: {result.metadata.get('optimization_trials', 0)}")
            print(f"Best score: {result.metadata.get('best_optimization_score', 0):.4f}")
            print(f"Best hyperparameters: {result.metadata.get('best_hyperparameters', {})}")
            
            # Print final model performance
            print(f"\nFinal model performance:")
            print(f"  Train accuracy: {result.metrics.get('train_accuracy', 0):.4f}")
            print(f"  Validation accuracy: {result.metrics.get('val_accuracy', 0):.4f}")
            print(f"  Test accuracy: {result.metrics.get('test_accuracy', 0):.4f}")
        else:
            print(f"❌ XGBoost optimization failed: {result.error_message}")


def example_regression_optimization():
    """Example: Optimize hyperparameters for regression task."""
    print("\n" + "=" * 60)
    print("Regression Hyperparameter Optimization Example")
    print("=" * 60)
    
    # Create regression dataset
    np.random.seed(42)
    n_samples = 800
    n_features = 8
    
    X = np.random.randn(n_samples, n_features)
    # Create continuous target
    y = (2 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] + 
         0.3 * X[:, 3] + np.random.normal(0, 0.5, n_samples))
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_cols)
    data['target'] = y
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_path = Path(temp_dir)
        
        # Prepare data
        prepare_data_splits(data, artifacts_path)
        
        # Create execution context for regression
        context = ExecutionContext(
            experiment_id="regression_optimization_example",
            stage_name="model_training",
            component_type=ComponentType.MODEL_TRAINING,
            artifacts_path=str(artifacts_path),
            logger=logging.getLogger("example"),
            config={
                'training': {
                    'target_column': 'target',
                    'model': {
                        'framework': 'sklearn',
                        'model_type': 'random_forest_regressor',
                        'task_type': 'regression',
                        'parameters': {
                            'random_state': 42
                        }
                    },
                    'hyperparameter_optimization': {
                        'enabled': True,
                        'method': 'optuna',
                        'n_trials': 15,
                        'timeout': 30,
                        'sampler': 'tpe',
                        'pruner': 'median',
                        'direction': 'maximize',  # Maximize R² score
                        'metric': 'r2_score',
                        'cv_folds': 3,
                        'random_state': 42,
                        'parameter_space': {
                            'n_estimators': {
                                'type': 'int',
                                'low': 20,
                                'high': 100
                            },
                            'max_depth': {
                                'type': 'int',
                                'low': 3,
                                'high': 12
                            },
                            'min_samples_split': {
                                'type': 'int',
                                'low': 2,
                                'high': 15
                            },
                            'min_samples_leaf': {
                                'type': 'int',
                                'low': 1,
                                'high': 8
                            }
                        }
                    }
                }
            }
        )
        
        # Create and run optimized trainer
        trainer = HyperparameterOptimizedTrainer()
        result = trainer.execute(context)
        
        if result.success:
            print("\n✅ Regression optimization completed successfully!")
            print(f"Training time: {result.metrics.get('training_time_seconds', 0):.2f} seconds")
            print(f"Optimization trials: {result.metadata.get('optimization_trials', 0)}")
            print(f"Best R² score: {result.metadata.get('best_optimization_score', 0):.4f}")
            print(f"Best hyperparameters: {result.metadata.get('best_hyperparameters', {})}")
            
            # Print final model performance
            print(f"\nFinal model performance:")
            print(f"  Train R²: {result.metrics.get('train_r2_score', 0):.4f}")
            print(f"  Train RMSE: {result.metrics.get('train_rmse', 0):.4f}")
            print(f"  Validation R²: {result.metrics.get('val_r2_score', 0):.4f}")
            print(f"  Validation RMSE: {result.metrics.get('val_rmse', 0):.4f}")
            print(f"  Test R²: {result.metrics.get('test_r2_score', 0):.4f}")
            print(f"  Test RMSE: {result.metrics.get('test_rmse', 0):.4f}")
        else:
            print(f"❌ Regression optimization failed: {result.error_message}")


def example_custom_parameter_space():
    """Example: Using custom parameter space with different parameter types."""
    print("\n" + "=" * 60)
    print("Custom Parameter Space Example")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_path = Path(temp_dir)
        
        # Prepare data
        prepare_data_splits(data, artifacts_path)
        
        # Create execution context with complex parameter space
        context = ExecutionContext(
            experiment_id="custom_param_space_example",
            stage_name="model_training",
            component_type=ComponentType.MODEL_TRAINING,
            artifacts_path=str(artifacts_path),
            logger=logging.getLogger("example"),
            config={
                'training': {
                    'target_column': 'target',
                    'model': {
                        'framework': 'sklearn',
                        'model_type': 'gradient_boosting_classifier',
                        'task_type': 'classification',
                        'parameters': {
                            'random_state': 42
                        }
                    },
                    'hyperparameter_optimization': {
                        'enabled': True,
                        'method': 'optuna',
                        'n_trials': 12,
                        'timeout': 30,
                        'sampler': 'tpe',
                        'pruner': 'successive_halving',
                        'direction': 'maximize',
                        'metric': 'f1_score',
                        'cv_folds': 3,
                        'random_state': 42,
                        'parameter_space': {
                            'n_estimators': {
                                'type': 'int',
                                'low': 50,
                                'high': 150
                            },
                            'learning_rate': {
                                'type': 'float',
                                'low': 0.05,
                                'high': 0.3,
                                'log': True
                            },
                            'max_depth': {
                                'type': 'int',
                                'low': 3,
                                'high': 8
                            },
                            'subsample': {
                                'type': 'float',
                                'low': 0.7,
                                'high': 1.0
                            },
                            'loss': {
                                'type': 'categorical',
                                'choices': ['deviance', 'exponential']
                            }
                        }
                    }
                }
            }
        )
        
        # Create and run optimized trainer
        trainer = HyperparameterOptimizedTrainer()
        result = trainer.execute(context)
        
        if result.success:
            print("\n✅ Custom parameter space optimization completed!")
            print(f"Training time: {result.metrics.get('training_time_seconds', 0):.2f} seconds")
            print(f"Optimization trials: {result.metadata.get('optimization_trials', 0)}")
            print(f"Pruned trials: {result.metadata.get('pruned_trials', 0)}")
            print(f"Best F1 score: {result.metadata.get('best_optimization_score', 0):.4f}")
            print(f"Best hyperparameters: {result.metadata.get('best_hyperparameters', {})}")
            
            # Print final model performance
            print(f"\nFinal model performance:")
            print(f"  Train F1: {result.metrics.get('train_f1_score', 0):.4f}")
            print(f"  Validation F1: {result.metrics.get('val_f1_score', 0):.4f}")
            print(f"  Test F1: {result.metrics.get('test_f1_score', 0):.4f}")
        else:
            print(f"❌ Custom parameter space optimization failed: {result.error_message}")


def main():
    """Run all hyperparameter optimization examples."""
    print("🚀 ML Pipeline Hyperparameter Optimization Examples")
    print("This script demonstrates various hyperparameter optimization scenarios.")
    
    try:
        # Check if required packages are available
        import optuna
        import xgboost
        print("✅ All required packages are available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install with: pip install optuna xgboost")
        return
    
    # Run examples
    example_sklearn_random_forest_optimization()
    example_xgboost_optimization()
    example_regression_optimization()
    example_custom_parameter_space()
    
    print("\n" + "=" * 60)
    print("🎉 All hyperparameter optimization examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()