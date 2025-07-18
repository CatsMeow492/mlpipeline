"""Model training framework with adapter pattern for multiple ML frameworks."""

import logging
import json
import pickle
import joblib
from typing import Dict, Any, List, Optional, Union, Tuple, Type
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

# ML Framework imports
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..core.interfaces import PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
from ..core.errors import ModelError, ConfigurationError


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str
    framework: str
    parameters: Dict[str, Any]
    task_type: str = "classification"  # classification or regression
    random_state: Optional[int] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class TrainingMetrics:
    """Training metrics and evaluation results."""
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    model_size_bytes: int = 0
    feature_importance: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.train_metrics is None:
            self.train_metrics = {}
        if self.val_metrics is None:
            self.val_metrics = {}


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create and return the model instance."""
        pass
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on input data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities (for classification)."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        pass
    
    def get_model_size(self) -> int:
        """Get model size in bytes."""
        if self.model is None:
            return 0
        
        try:
            import tempfile
            with tempfile.NamedTemporaryFile() as tmp:
                self.save_model(tmp.name)
                return Path(tmp.name).stat().st_size
        except Exception:
            return 0


class SklearnAdapter(ModelAdapter):
    """Adapter for scikit-learn models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_classes = {
            # Classification
            'logistic_regression': 'sklearn.linear_model.LogisticRegression',
            'random_forest_classifier': 'sklearn.ensemble.RandomForestClassifier',
            'gradient_boosting_classifier': 'sklearn.ensemble.GradientBoostingClassifier',
            'svm_classifier': 'sklearn.svm.SVC',
            'decision_tree_classifier': 'sklearn.tree.DecisionTreeClassifier',
            'naive_bayes': 'sklearn.naive_bayes.GaussianNB',
            'knn_classifier': 'sklearn.neighbors.KNeighborsClassifier',
            
            # Regression
            'linear_regression': 'sklearn.linear_model.LinearRegression',
            'ridge_regression': 'sklearn.linear_model.Ridge',
            'lasso_regression': 'sklearn.linear_model.Lasso',
            'random_forest_regressor': 'sklearn.ensemble.RandomForestRegressor',
            'gradient_boosting_regressor': 'sklearn.ensemble.GradientBoostingRegressor',
            'svm_regressor': 'sklearn.svm.SVR',
            'decision_tree_regressor': 'sklearn.tree.DecisionTreeRegressor',
            'knn_regressor': 'sklearn.neighbors.KNeighborsRegressor',
        }
    
    def create_model(self) -> BaseEstimator:
        """Create scikit-learn model instance."""
        model_type = self.config.model_type
        
        if model_type not in self.model_classes:
            raise ModelError(f"Unsupported sklearn model type: {model_type}")
        
        # Import the model class dynamically
        module_path, class_name = self.model_classes[model_type].rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        
        # Set random state if provided
        params = self.config.parameters.copy()
        if self.config.random_state is not None and 'random_state' in model_class().get_params():
            params['random_state'] = self.config.random_state
        
        self.model = model_class(**params)
        return self.model
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """Fit the scikit-learn model."""
        if self.model is None:
            self.create_model()
        
        self.logger.info(f"Training {self.config.model_type} model...")
        
        # Some models support validation data during training
        if hasattr(self.model, 'fit') and X_val is not None and y_val is not None:
            # Check if model supports early stopping or validation
            if hasattr(self.model, 'validation_fraction'):
                # Combine train and validation for models that handle validation internally
                X_combined = pd.concat([X_train, X_val])
                y_combined = pd.concat([y_train, y_val])
                self.model.fit(X_combined, y_combined)
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        self.logger.info("Model training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with scikit-learn model."""
        if not self.is_fitted or self.model is None:
            raise ModelError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities."""
        if not self.is_fitted or self.model is None:
            raise ModelError("Model must be fitted before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For SVM and other models that have decision_function
            scores = self.model.decision_function(X)
            if scores.ndim == 1:
                # Binary classification
                from scipy.special import expit
                proba_pos = expit(scores)
                return np.column_stack([1 - proba_pos, proba_pos])
            else:
                # Multi-class
                from scipy.special import softmax
                return softmax(scores, axis=1)
        else:
            return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from scikit-learn model."""
        if not self.is_fitted or self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            return dict(zip(feature_names, importances.astype(float)))
        elif hasattr(self.model, 'coef_'):
            # Linear models
            coef = self.model.coef_
            if coef.ndim == 1:
                feature_names = [f'feature_{i}' for i in range(len(coef))]
                return dict(zip(feature_names, np.abs(coef).astype(float)))
            else:
                # Multi-class: use mean absolute coefficient
                feature_names = [f'feature_{i}' for i in range(coef.shape[1])]
                importance = np.mean(np.abs(coef), axis=0)
                return dict(zip(feature_names, importance.astype(float)))
        else:
            return None
    
    def save_model(self, path: str) -> None:
        """Save scikit-learn model."""
        if self.model is None:
            raise ModelError("No model to save")
        
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load scikit-learn model."""
        self.model = joblib.load(path)
        self.is_fitted = True
        self.logger.info(f"Model loaded from {path}")


class XGBoostAdapter(ModelAdapter):
    """Adapter for XGBoost models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_classes = {
            'xgb_classifier': 'XGBClassifier',
            'xgb_regressor': 'XGBRegressor'
        }
    
    def create_model(self) -> Any:
        """Create XGBoost model instance."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ModelError("XGBoost is not installed. Please install it with: pip install xgboost")
        
        model_type = self.config.model_type
        
        if model_type not in self.model_classes:
            raise ModelError(f"Unsupported XGBoost model type: {model_type}")
        
        model_class_name = self.model_classes[model_type]
        model_class = getattr(xgb, model_class_name)
        
        # Set random state if provided
        params = self.config.parameters.copy()
        if self.config.random_state is not None:
            params['random_state'] = self.config.random_state
        
        self.model = model_class(**params)
        return self.model
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """Fit XGBoost model."""
        if self.model is None:
            self.create_model()
        
        self.logger.info(f"Training {self.config.model_type} model...")
        
        # XGBoost supports validation data for early stopping
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        self.logger.info("XGBoost model training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost model."""
        if not self.is_fitted or self.model is None:
            raise ModelError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities."""
        if not self.is_fitted or self.model is None:
            raise ModelError("Model must be fitted before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from XGBoost model."""
        if not self.is_fitted or self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            return dict(zip(feature_names, importances.astype(float)))
        else:
            return None
    
    def save_model(self, path: str) -> None:
        """Save XGBoost model."""
        if self.model is None:
            raise ModelError("No model to save")
        
        self.model.save_model(path)
        self.logger.info(f"XGBoost model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load XGBoost model."""
        if self.model is None:
            self.create_model()
        
        self.model.load_model(path)
        self.is_fitted = True
        self.logger.info(f"XGBoost model loaded from {path}")


class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = None
        self.criterion = None
        self.optimizer = None
    
    def create_model(self) -> Any:
        """Create PyTorch model instance."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ModelError("PyTorch is not installed. Please install it with: pip install torch")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a simple neural network based on configuration
        model_type = self.config.model_type
        params = self.config.parameters
        
        if model_type == 'neural_network':
            input_size = params.get('input_size', 10)
            hidden_sizes = params.get('hidden_sizes', [64, 32])
            output_size = params.get('output_size', 1)
            dropout_rate = params.get('dropout_rate', 0.2)
            
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, output_size))
            
            if self.config.task_type == 'classification' and output_size > 1:
                layers.append(nn.Softmax(dim=1))
            elif self.config.task_type == 'classification' and output_size == 1:
                layers.append(nn.Sigmoid())
            
            self.model = nn.Sequential(*layers).to(self.device)
        else:
            raise ModelError(f"Unsupported PyTorch model type: {model_type}")
        
        return self.model
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """Fit PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ModelError("PyTorch is not installed")
        
        if self.model is None:
            # Need to set input size based on data
            params = self.config.parameters.copy()
            params['input_size'] = X_train.shape[1]
            if self.config.task_type == 'classification':
                params['output_size'] = len(y_train.unique())
            else:
                params['output_size'] = 1
            
            self.config.parameters = params
            self.create_model()
        
        self.logger.info(f"Training {self.config.model_type} model...")
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
        
        # Setup loss and optimizer
        if self.config.task_type == 'classification':
            if len(y_train.unique()) > 2:
                self.criterion = nn.CrossEntropyLoss()
                y_train_tensor = y_train_tensor.long()
                if X_val is not None:
                    y_val_tensor = y_val_tensor.long()
            else:
                self.criterion = nn.BCELoss()
                y_train_tensor = y_train_tensor.unsqueeze(1)
                if X_val is not None:
                    y_val_tensor = y_val_tensor.unsqueeze(1)
        else:
            self.criterion = nn.MSELoss()
            y_train_tensor = y_train_tensor.unsqueeze(1)
            if X_val is not None:
                y_val_tensor = y_val_tensor.unsqueeze(1)
        
        learning_rate = self.config.parameters.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        epochs = self.config.parameters.get('epochs', 100)
        batch_size = self.config.parameters.get('batch_size', 32)
        
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            
            # Validation
            if X_val is not None and epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor)
                    self.logger.debug(f"Epoch {epoch}, Val Loss: {val_loss.item():.4f}")
        
        self.is_fitted = True
        self.logger.info("PyTorch model training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with PyTorch model."""
        if not self.is_fitted or self.model is None:
            raise ModelError("Model must be fitted before making predictions")
        
        try:
            import torch
        except ImportError:
            raise ModelError("PyTorch is not installed")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            outputs = self.model(X_tensor)
            
            if self.config.task_type == 'classification':
                if outputs.shape[1] > 1:
                    # Multi-class
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    # Binary
                    predictions = (outputs > 0.5).float().squeeze()
            else:
                predictions = outputs.squeeze()
            
            return predictions.cpu().numpy()
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities."""
        if not self.is_fitted or self.model is None:
            raise ModelError("Model must be fitted before making predictions")
        
        if self.config.task_type != 'classification':
            return None
        
        try:
            import torch
        except ImportError:
            raise ModelError("PyTorch is not installed")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (not directly available for neural networks)."""
        return None
    
    def save_model(self, path: str) -> None:
        """Save PyTorch model."""
        if self.model is None:
            raise ModelError("No model to save")
        
        try:
            import torch
        except ImportError:
            raise ModelError("PyTorch is not installed")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        self.logger.info(f"PyTorch model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load PyTorch model."""
        try:
            import torch
        except ImportError:
            raise ModelError("PyTorch is not installed")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
        self.logger.info(f"PyTorch model loaded from {path}")


class ModelTrainer(PipelineComponent):
    """Main model training component with adapter pattern support."""
    
    def __init__(self):
        super().__init__(ComponentType.MODEL_TRAINING)
        self.adapter_registry = {
            'sklearn': SklearnAdapter,
            'xgboost': XGBoostAdapter,
            'pytorch': PyTorchAdapter
        }
        self.model_adapter: Optional[ModelAdapter] = None
        self.training_metrics: Optional[TrainingMetrics] = None
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate model training configuration."""
        try:
            model_config = config.get('model', {})
            
            if not model_config:
                self.logger.error("Model configuration not found")
                return False
            
            # Check for framework or type field
            framework = model_config.get('type', model_config.get('framework'))
            if not framework:
                self.logger.error("Model framework/type not specified")
                return False
            
            if framework not in self.adapter_registry:
                self.logger.error(f"Unsupported framework: {framework}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup model training component."""
        # Pass the entire context config to validate_config
        if not self.validate_config(context.config):
            raise ConfigurationError("Invalid model training configuration")
        
        # Create model configuration
        model_config_dict = context.config['model']
        
        # Map 'type' to 'framework' and algorithm to model_type for sklearn
        framework = model_config_dict.get('type', model_config_dict.get('framework', 'sklearn'))
        
        # For sklearn, convert algorithm parameter to model_type
        if framework == 'sklearn' and 'algorithm' in model_config_dict.get('parameters', {}):
            algorithm = model_config_dict['parameters']['algorithm']
            # Map algorithm names to model_type format
            algorithm_mapping = {
                'RandomForestRegressor': 'random_forest_regressor',
                'RandomForestClassifier': 'random_forest_classifier',
                'LogisticRegression': 'logistic_regression',
                'LinearRegression': 'linear_regression',
                'SVC': 'svm_classifier',
                'SVR': 'svm_regressor'
            }
            model_type = algorithm_mapping.get(algorithm, algorithm.lower())
            
            # Remove algorithm from parameters
            parameters = model_config_dict.get('parameters', {}).copy()
            parameters.pop('algorithm', None)
        else:
            model_type = model_config_dict.get('model_type', 'random_forest_regressor')
            parameters = model_config_dict.get('parameters', {})
        
        # Determine task type based on model type or explicitly set
        if 'task_type' in model_config_dict:
            task_type = model_config_dict['task_type']
        else:
            # Infer from model type
            task_type = 'regression' if 'regressor' in model_type or 'regression' in model_type else 'classification'
        
        model_config = ModelConfig(
            model_type=model_type,
            framework=framework,
            parameters=parameters,
            task_type=task_type,
            random_state=model_config_dict.get('random_state')
        )
        
        # Create model adapter
        adapter_class = self.adapter_registry[model_config.framework]
        self.model_adapter = adapter_class(model_config)
        
        self.logger.info(f"Setup {model_config.framework} {model_config.model_type} trainer")
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute model training."""
        try:
            # Setup the trainer if not already done
            if self.model_adapter is None:
                self.setup(context)
            
            start_time = datetime.now()
            
            # Load training data
            train_data, val_data, test_data = self._load_training_data(context)
            
            # Split features and targets
            X_train, y_train = self._split_features_target(train_data, context.config)
            X_val, y_val = self._split_features_target(val_data, context.config) if val_data is not None else (None, None)
            X_test, y_test = self._split_features_target(test_data, context.config) if test_data is not None else (None, None)
            
            self.logger.info(f"Training data shape: {X_train.shape}")
            if X_val is not None:
                self.logger.info(f"Validation data shape: {X_val.shape}")
            if X_test is not None:
                self.logger.info(f"Test data shape: {X_test.shape}")
            
            # Train the model
            self.model_adapter.fit(X_train, y_train, X_val, y_val)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            train_metrics = self._evaluate_model(X_train, y_train, "train")
            val_metrics = self._evaluate_model(X_val, y_val, "validation") if X_val is not None else {}
            test_metrics = self._evaluate_model(X_test, y_test, "test") if X_test is not None else None
            
            # Get feature importance
            feature_importance = self.model_adapter.get_feature_importance()
            
            # Get model size
            model_size = self.model_adapter.get_model_size()
            
            # Create training metrics
            self.training_metrics = TrainingMetrics(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                training_time=training_time,
                model_size_bytes=model_size,
                feature_importance=feature_importance
            )
            
            # Save model and artifacts
            artifacts = self._save_artifacts(context)
            
            # Prepare result metrics
            result_metrics = {
                'training_time_seconds': training_time,
                'model_size_bytes': model_size,
                'train_samples': len(X_train),
                'val_samples': len(X_val) if X_val is not None else 0,
                'test_samples': len(X_test) if X_test is not None else 0,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
            }
            
            if test_metrics:
                result_metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
            
            self.logger.info("Model training completed successfully")
            
            return ExecutionResult(
                success=True,
                artifacts=artifacts,
                metrics=result_metrics,
                metadata={
                    'model_type': self.model_adapter.config.model_type,
                    'framework': self.model_adapter.config.framework,
                    'task_type': self.model_adapter.config.task_type,
                    'feature_count': X_train.shape[1],
                    'has_feature_importance': feature_importance is not None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def _load_training_data(self, context: ExecutionContext) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load training data from artifacts."""
        artifacts_path = Path(context.artifacts_path)
        
        # Look for preprocessed data files
        train_file = artifacts_path / "train_preprocessed.parquet"
        val_file = artifacts_path / "val_preprocessed.parquet"
        test_file = artifacts_path / "test_preprocessed.parquet"
        
        if not train_file.exists():
            raise ModelError("Training data not found. Run preprocessing first.")
        
        train_data = pd.read_parquet(train_file)
        val_data = pd.read_parquet(val_file) if val_file.exists() else None
        test_data = pd.read_parquet(test_file) if test_file.exists() else None
        
        return train_data, val_data, test_data
    
    def _split_features_target(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Split data into features and target."""
        if data is None:
            return None, None
        
        # Look for target column in various places or use default
        target_column = 'target'  # Default
        
        # Check if there's a special target column in the data (like 'quality' for wine)
        # For now, we'll assume the last column is the target if 'target' doesn't exist
        if 'target' not in data.columns:
            # Use the last column as target
            target_column = data.columns[-1]
            self.logger.info(f"Using '{target_column}' as target column")
        
        if target_column not in data.columns:
            raise ModelError(f"Target column '{target_column}' not found in data. Available columns: {list(data.columns)}")
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        self.logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series, split_name: str) -> Dict[str, float]:
        """Evaluate model performance."""
        if X is None or y is None:
            return {}
        
        try:
            predictions = self.model_adapter.predict(X)
            probabilities = self.model_adapter.predict_proba(X)
            
            metrics = {}
            
            if self.model_adapter.config.task_type == 'classification':
                metrics['accuracy'] = float(accuracy_score(y, predictions))
                metrics['precision'] = float(precision_score(y, predictions, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y, predictions, average='weighted', zero_division=0))
                metrics['f1_score'] = float(f1_score(y, predictions, average='weighted', zero_division=0))
                
                # ROC AUC for binary classification
                if len(np.unique(y)) == 2 and probabilities is not None:
                    if probabilities.shape[1] == 2:
                        metrics['roc_auc'] = float(roc_auc_score(y, probabilities[:, 1]))
                    else:
                        metrics['roc_auc'] = float(roc_auc_score(y, probabilities))
            else:
                # Regression metrics
                metrics['mse'] = float(mean_squared_error(y, predictions))
                metrics['mae'] = float(mean_absolute_error(y, predictions))
                metrics['r2_score'] = float(r2_score(y, predictions))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
            
            self.logger.info(f"{split_name.capitalize()} metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate model on {split_name} data: {str(e)}")
            return {}
    
    def _save_artifacts(self, context: ExecutionContext) -> List[str]:
        """Save model and training artifacts."""
        artifacts = []
        artifacts_path = Path(context.artifacts_path)
        
        # Save trained model
        model_path = artifacts_path / "trained_model"
        if self.model_adapter.config.framework == 'pytorch':
            model_path = model_path.with_suffix('.pth')
        elif self.model_adapter.config.framework == 'xgboost':
            model_path = model_path.with_suffix('.json')
        else:
            model_path = model_path.with_suffix('.joblib')
        
        self.model_adapter.save_model(str(model_path))
        artifacts.append(str(model_path))
        
        # Save training metrics
        if self.training_metrics:
            metrics_path = artifacts_path / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(asdict(self.training_metrics), f, indent=2, default=str)
            artifacts.append(str(metrics_path))
        
        # Save model configuration
        config_path = artifacts_path / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.model_adapter.config), f, indent=2)
        artifacts.append(str(config_path))
        
        return artifacts
    
    def cleanup(self, context: ExecutionContext) -> None:
        """Cleanup training resources."""
        self.model_adapter = None
        self.training_metrics = None
        self.logger.info("Model training cleanup completed")