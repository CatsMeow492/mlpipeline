"""Example demonstrating model training framework with multiple ML frameworks."""

import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

from mlpipeline.models import ModelTrainer
from mlpipeline.core.interfaces import ExecutionContext, ComponentType


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Create classification data
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some pattern
    weights = np.random.randn(n_features)
    linear_combination = X @ weights
    probabilities = 1 / (1 + np.exp(-linear_combination))
    y = (probabilities > 0.5).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    
    return data


def create_regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 5
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with linear relationship + noise
    weights = np.array([2.5, -1.8, 3.2, -0.5, 1.1])
    y = X @ weights + np.random.randn(n_samples) * 0.5
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    
    return data


def split_and_save_data(data, temp_dir, train_ratio=0.7, val_ratio=0.15):
    """Split data and save to files."""
    n_samples = len(data)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Shuffle data
    data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    train_data = data_shuffled[:n_train]
    val_data = data_shuffled[n_train:n_train + n_val]
    test_data = data_shuffled[n_train + n_val:]
    
    # Save to files
    train_data.to_parquet(temp_dir / "train_preprocessed.parquet", index=False)
    val_data.to_parquet(temp_dir / "val_preprocessed.parquet", index=False)
    test_data.to_parquet(temp_dir / "test_preprocessed.parquet", index=False)
    
    return len(train_data), len(val_data), len(test_data)


def demonstrate_sklearn_models():
    """Demonstrate various scikit-learn models."""
    print("\n" + "="*60)
    print("SCIKIT-LEARN MODELS DEMONSTRATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create and save classification data
        print("\n1. Classification Models")
        print("-" * 30)
        
        data = create_sample_data()
        train_size, val_size, test_size = split_and_save_data(data, temp_path)
        print(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Test different classification models
        classification_models = [
            {
                'name': 'Logistic Regression',
                'model_type': 'logistic_regression',
                'parameters': {'C': 1.0, 'max_iter': 1000}
            },
            {
                'name': 'Random Forest',
                'model_type': 'random_forest_classifier',
                'parameters': {'n_estimators': 50, 'max_depth': 10}
            },
            {
                'name': 'Gradient Boosting',
                'model_type': 'gradient_boosting_classifier',
                'parameters': {'n_estimators': 50, 'learning_rate': 0.1}
            }
        ]
        
        for model_config in classification_models:
            print(f"\nTraining {model_config['name']}...")
            
            trainer = ModelTrainer()
            context = ExecutionContext(
                experiment_id=f"sklearn_classification_{model_config['model_type']}",
                stage_name="training",
                component_type=ComponentType.MODEL_TRAINING,
                config={
                    'training': {
                        'model': {
                            'framework': 'sklearn',
                            'model_type': model_config['model_type'],
                            'task_type': 'classification',
                            'parameters': model_config['parameters'],
                            'random_state': 42
                        },
                        'target_column': 'target'
                    }
                },
                artifacts_path=str(temp_path),
                logger=None,
                metadata={}
            )
            
            result = trainer.execute(context)
            
            if result.success:
                print(f"  ✓ Training successful!")
                print(f"  ✓ Training time: {result.metrics['training_time_seconds']:.2f}s")
                print(f"  ✓ Train accuracy: {result.metrics['train_accuracy']:.3f}")
                print(f"  ✓ Val accuracy: {result.metrics['val_accuracy']:.3f}")
                print(f"  ✓ Test accuracy: {result.metrics.get('test_accuracy', 'N/A')}")
                print(f"  ✓ Model size: {result.metrics['model_size_bytes']} bytes")
            else:
                print(f"  ✗ Training failed: {result.error_message}")
        
        # Test regression models
        print("\n2. Regression Models")
        print("-" * 30)
        
        reg_data = create_regression_data()
        split_and_save_data(reg_data, temp_path)
        
        regression_models = [
            {
                'name': 'Linear Regression',
                'model_type': 'linear_regression',
                'parameters': {}
            },
            {
                'name': 'Ridge Regression',
                'model_type': 'ridge_regression',
                'parameters': {'alpha': 1.0}
            },
            {
                'name': 'Random Forest Regressor',
                'model_type': 'random_forest_regressor',
                'parameters': {'n_estimators': 50, 'max_depth': 10}
            }
        ]
        
        for model_config in regression_models:
            print(f"\nTraining {model_config['name']}...")
            
            trainer = ModelTrainer()
            context = ExecutionContext(
                experiment_id=f"sklearn_regression_{model_config['model_type']}",
                stage_name="training",
                component_type=ComponentType.MODEL_TRAINING,
                config={
                    'training': {
                        'model': {
                            'framework': 'sklearn',
                            'model_type': model_config['model_type'],
                            'task_type': 'regression',
                            'parameters': model_config['parameters'],
                            'random_state': 42
                        },
                        'target_column': 'target'
                    }
                },
                artifacts_path=str(temp_path),
                logger=None,
                metadata={}
            )
            
            result = trainer.execute(context)
            
            if result.success:
                print(f"  ✓ Training successful!")
                print(f"  ✓ Training time: {result.metrics['training_time_seconds']:.2f}s")
                print(f"  ✓ Train R²: {result.metrics['train_r2_score']:.3f}")
                print(f"  ✓ Val R²: {result.metrics['val_r2_score']:.3f}")
                print(f"  ✓ Train RMSE: {result.metrics['train_rmse']:.3f}")
                print(f"  ✓ Val RMSE: {result.metrics['val_rmse']:.3f}")
            else:
                print(f"  ✗ Training failed: {result.error_message}")


def demonstrate_xgboost_models():
    """Demonstrate XGBoost models."""
    print("\n" + "="*60)
    print("XGBOOST MODELS DEMONSTRATION")
    print("="*60)
    
    try:
        import xgboost
        print("XGBoost is available!")
    except ImportError:
        print("XGBoost is not installed. Skipping XGBoost demonstration.")
        print("To install XGBoost: pip install xgboost")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create and save data
        data = create_sample_data()
        split_and_save_data(data, temp_path)
        
        print(f"\nTraining XGBoost Classifier...")
        
        trainer = ModelTrainer()
        context = ExecutionContext(
            experiment_id="xgboost_classification",
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config={
                'training': {
                    'model': {
                        'framework': 'xgboost',
                        'model_type': 'xgb_classifier',
                        'task_type': 'classification',
                        'parameters': {
                            'n_estimators': 100,
                            'max_depth': 6,
                            'learning_rate': 0.1
                        },
                        'random_state': 42
                    },
                    'target_column': 'target'
                }
            },
            artifacts_path=str(temp_path),
            logger=None,
            metadata={}
        )
        
        result = trainer.execute(context)
        
        if result.success:
            print(f"  ✓ Training successful!")
            print(f"  ✓ Training time: {result.metrics['training_time_seconds']:.2f}s")
            print(f"  ✓ Train accuracy: {result.metrics['train_accuracy']:.3f}")
            print(f"  ✓ Val accuracy: {result.metrics['val_accuracy']:.3f}")
            print(f"  ✓ Feature importance available: {result.metadata['has_feature_importance']}")
        else:
            print(f"  ✗ Training failed: {result.error_message}")


def demonstrate_pytorch_models():
    """Demonstrate PyTorch models."""
    print("\n" + "="*60)
    print("PYTORCH MODELS DEMONSTRATION")
    print("="*60)
    
    try:
        import torch
        print("PyTorch is available!")
    except ImportError:
        print("PyTorch is not installed. Skipping PyTorch demonstration.")
        print("To install PyTorch: pip install torch")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create and save data
        data = create_sample_data()
        split_and_save_data(data, temp_path)
        
        print(f"\nTraining Neural Network...")
        
        trainer = ModelTrainer()
        context = ExecutionContext(
            experiment_id="pytorch_neural_network",
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config={
                'training': {
                    'model': {
                        'framework': 'pytorch',
                        'model_type': 'neural_network',
                        'task_type': 'classification',
                        'parameters': {
                            'hidden_sizes': [64, 32, 16],
                            'dropout_rate': 0.2,
                            'epochs': 50,
                            'batch_size': 32,
                            'learning_rate': 0.001
                        },
                        'random_state': 42
                    },
                    'target_column': 'target'
                }
            },
            artifacts_path=str(temp_path),
            logger=None,
            metadata={}
        )
        
        result = trainer.execute(context)
        
        if result.success:
            print(f"  ✓ Training successful!")
            print(f"  ✓ Training time: {result.metrics['training_time_seconds']:.2f}s")
            print(f"  ✓ Train accuracy: {result.metrics['train_accuracy']:.3f}")
            print(f"  ✓ Val accuracy: {result.metrics['val_accuracy']:.3f}")
            print(f"  ✓ Model size: {result.metrics['model_size_bytes']} bytes")
        else:
            print(f"  ✗ Training failed: {result.error_message}")


def demonstrate_model_comparison():
    """Demonstrate model comparison across frameworks."""
    print("\n" + "="*60)
    print("MODEL COMPARISON ACROSS FRAMEWORKS")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create and save data
        data = create_sample_data()
        split_and_save_data(data, temp_path)
        
        models_to_compare = [
            {
                'name': 'Logistic Regression',
                'framework': 'sklearn',
                'model_type': 'logistic_regression',
                'parameters': {'C': 1.0, 'max_iter': 1000}
            },
            {
                'name': 'Random Forest',
                'framework': 'sklearn',
                'model_type': 'random_forest_classifier',
                'parameters': {'n_estimators': 100, 'max_depth': 10}
            }
        ]
        
        # Add XGBoost if available
        try:
            import xgboost
            models_to_compare.append({
                'name': 'XGBoost',
                'framework': 'xgboost',
                'model_type': 'xgb_classifier',
                'parameters': {'n_estimators': 100, 'max_depth': 6}
            })
        except ImportError:
            pass
        
        # Add PyTorch if available
        try:
            import torch
            models_to_compare.append({
                'name': 'Neural Network',
                'framework': 'pytorch',
                'model_type': 'neural_network',
                'parameters': {
                    'hidden_sizes': [32, 16],
                    'epochs': 30,
                    'batch_size': 32,
                    'learning_rate': 0.01
                }
            })
        except ImportError:
            pass
        
        results = []
        
        for model_config in models_to_compare:
            print(f"\nTraining {model_config['name']} ({model_config['framework']})...")
            
            trainer = ModelTrainer()
            context = ExecutionContext(
                experiment_id=f"comparison_{model_config['framework']}_{model_config['model_type']}",
                stage_name="training",
                component_type=ComponentType.MODEL_TRAINING,
                config={
                    'training': {
                        'model': {
                            'framework': model_config['framework'],
                            'model_type': model_config['model_type'],
                            'task_type': 'classification',
                            'parameters': model_config['parameters'],
                            'random_state': 42
                        },
                        'target_column': 'target'
                    }
                },
                artifacts_path=str(temp_path),
                logger=None,
                metadata={}
            )
            
            result = trainer.execute(context)
            
            if result.success:
                results.append({
                    'name': model_config['name'],
                    'framework': model_config['framework'],
                    'train_accuracy': result.metrics['train_accuracy'],
                    'val_accuracy': result.metrics['val_accuracy'],
                    'training_time': result.metrics['training_time_seconds'],
                    'model_size': result.metrics['model_size_bytes']
                })
                print(f"  ✓ Success! Val accuracy: {result.metrics['val_accuracy']:.3f}")
            else:
                print(f"  ✗ Failed: {result.error_message}")
        
        # Display comparison results
        if results:
            print("\n" + "="*80)
            print("MODEL COMPARISON RESULTS")
            print("="*80)
            print(f"{'Model':<20} {'Framework':<10} {'Train Acc':<10} {'Val Acc':<10} {'Time (s)':<10} {'Size (KB)':<10}")
            print("-" * 80)
            
            for result in results:
                print(f"{result['name']:<20} {result['framework']:<10} "
                      f"{result['train_accuracy']:<10.3f} {result['val_accuracy']:<10.3f} "
                      f"{result['training_time']:<10.2f} {result['model_size']/1024:<10.1f}")
            
            # Find best model
            best_model = max(results, key=lambda x: x['val_accuracy'])
            print(f"\n🏆 Best model: {best_model['name']} ({best_model['framework']}) "
                  f"with validation accuracy: {best_model['val_accuracy']:.3f}")


def main():
    """Run the model training framework demonstration."""
    print("MODEL TRAINING FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    print("\nThis example demonstrates the model training framework with:")
    print("- Multiple ML frameworks (scikit-learn, XGBoost, PyTorch)")
    print("- Classification and regression tasks")
    print("- Automatic model evaluation and comparison")
    print("- Flexible configuration system")
    print("- Model artifact management")
    
    # Run demonstrations
    demonstrate_sklearn_models()
    demonstrate_xgboost_models()
    demonstrate_pytorch_models()
    demonstrate_model_comparison()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED!")
    print("="*60)
    print("\nKey features demonstrated:")
    print("✓ Multi-framework support (sklearn, XGBoost, PyTorch)")
    print("✓ Classification and regression tasks")
    print("✓ Automatic model evaluation with standard metrics")
    print("✓ Feature importance extraction (where available)")
    print("✓ Model serialization and artifact management")
    print("✓ Flexible configuration system")
    print("✓ Comprehensive error handling")
    print("✓ Performance metrics tracking")


if __name__ == "__main__":
    main()