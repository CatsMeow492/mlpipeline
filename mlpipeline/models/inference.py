"""Model inference system with loading, validation, and prediction capabilities."""

import logging
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Iterator, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

from ..core.interfaces import PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
from ..core.errors import ModelError, DataError, ConfigurationError
from ..data.preprocessing import DataPreprocessor, PreprocessingMetadata


@dataclass
class ModelMetadata:
    """Metadata for trained models to ensure compatibility."""
    model_id: str
    model_type: str
    framework: str
    version: str
    created_at: str
    feature_columns: List[str]
    target_column: Optional[str]
    preprocessing_metadata_hash: str
    model_parameters: Dict[str, Any]
    training_metrics: Dict[str, float]
    data_schema: Dict[str, str]
    model_size_bytes: int
    python_version: str
    dependencies: Dict[str, str]


@dataclass
class InferenceResult:
    """Result of model inference."""
    predictions: np.ndarray
    prediction_probabilities: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    preprocessing_time: Optional[float] = None
    inference_time: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelLoader:
    """Handles loading and validation of trained models."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.supported_formats = {
            'pickle': self._load_pickle,
            'joblib': self._load_joblib,
            'sklearn': self._load_sklearn,
            'xgboost': self._load_xgboost,
            'pytorch': self._load_pytorch
        }
    
    def load_model(self, model_path: str, metadata_path: Optional[str] = None) -> Tuple[Any, Optional[ModelMetadata]]:
        """Load a model with optional metadata validation."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise ModelError(f"Model file not found: {model_path}")
        
        # Load metadata if provided
        metadata = None
        if metadata_path:
            metadata = self._load_metadata(metadata_path)
        
        # Determine model format from file extension or metadata
        model_format = self._detect_model_format(model_path, metadata)
        
        if model_format not in self.supported_formats:
            raise ModelError(f"Unsupported model format: {model_format}")
        
        # Load the model
        try:
            model = self.supported_formats[model_format](model_path)
            self.logger.info(f"Successfully loaded {model_format} model from {model_path}")
            return model, metadata
        except Exception as e:
            raise ModelError(f"Failed to load model: {str(e)}")
    
    def _load_metadata(self, metadata_path: str) -> ModelMetadata:
        """Load model metadata from JSON file."""
        metadata_path = Path(metadata_path)
        
        if not metadata_path.exists():
            raise ModelError(f"Metadata file not found: {metadata_path}")
        
        try:
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            return ModelMetadata(**metadata_dict)
        except Exception as e:
            raise ModelError(f"Failed to load metadata: {str(e)}")
    
    def _detect_model_format(self, model_path: Path, metadata: Optional[ModelMetadata]) -> str:
        """Detect model format from file extension or metadata."""
        if metadata and hasattr(metadata, 'framework'):
            framework_mapping = {
                'sklearn': 'sklearn',
                'scikit-learn': 'sklearn',
                'xgboost': 'xgboost',
                'pytorch': 'pytorch',
                'torch': 'pytorch'
            }
            return framework_mapping.get(metadata.framework.lower(), 'pickle')
        
        # Fallback to file extension
        extension = model_path.suffix.lower()
        extension_mapping = {
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.joblib': 'joblib',
            '.model': 'xgboost',
            '.pt': 'pytorch',
            '.pth': 'pytorch'
        }
        
        return extension_mapping.get(extension, 'pickle')
    
    def _load_pickle(self, model_path: Path) -> Any:
        """Load model using pickle."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_joblib(self, model_path: Path) -> Any:
        """Load model using joblib."""
        return joblib.load(model_path)
    
    def _load_sklearn(self, model_path: Path) -> Any:
        """Load scikit-learn model."""
        return joblib.load(model_path)
    
    def _load_xgboost(self, model_path: Path) -> Any:
        """Load XGBoost model."""
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(model_path))
            return model
        except ImportError:
            raise ModelError("XGBoost not installed")
    
    def _load_pytorch(self, model_path: Path) -> Any:
        """Load PyTorch model."""
        try:
            import torch
            return torch.load(model_path, map_location='cpu')
        except ImportError:
            raise ModelError("PyTorch not installed")


class ModelValidator:
    """Validates model compatibility with input data and preprocessing pipeline."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_model_compatibility(self, model: Any, model_metadata: Optional[ModelMetadata],
                                   preprocessing_metadata: Optional[PreprocessingMetadata],
                                   input_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate model compatibility with preprocessing pipeline and input data."""
        validation_results = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'checks_performed': []
        }
        
        # Check 1: Model metadata validation
        if model_metadata:
            self._validate_model_metadata(model_metadata, validation_results)
        else:
            validation_results['warnings'].append("No model metadata available for validation")
        
        # Check 2: Preprocessing pipeline compatibility
        if preprocessing_metadata and model_metadata:
            self._validate_preprocessing_compatibility(
                model_metadata, preprocessing_metadata, validation_results
            )
        else:
            validation_results['warnings'].append("Cannot validate preprocessing compatibility")
        
        # Check 3: Input data schema validation
        if model_metadata:
            self._validate_input_schema(model_metadata, input_data, validation_results)
        
        # Check 4: Feature count validation
        self._validate_feature_count(model, input_data, validation_results)
        
        # Check 5: Model type validation
        self._validate_model_type(model, validation_results)
        
        # Determine overall compatibility
        validation_results['compatible'] = len(validation_results['errors']) == 0
        
        return validation_results
    
    def _validate_model_metadata(self, metadata: ModelMetadata, results: Dict[str, Any]) -> None:
        """Validate model metadata completeness and consistency."""
        results['checks_performed'].append('model_metadata')
        
        required_fields = ['model_id', 'model_type', 'framework', 'feature_columns']
        missing_fields = [field for field in required_fields if not getattr(metadata, field, None)]
        
        if missing_fields:
            results['errors'].append(f"Missing required metadata fields: {missing_fields}")
        
        # Check if metadata is recent (within reasonable time)
        try:
            created_at = datetime.fromisoformat(metadata.created_at)
            age_days = (datetime.now() - created_at).days
            if age_days > 365:  # More than a year old
                results['warnings'].append(f"Model is {age_days} days old, consider retraining")
        except:
            results['warnings'].append("Invalid or missing creation timestamp")
    
    def _validate_preprocessing_compatibility(self, model_metadata: ModelMetadata,
                                           preprocessing_metadata: PreprocessingMetadata,
                                           results: Dict[str, Any]) -> None:
        """Validate preprocessing pipeline compatibility."""
        results['checks_performed'].append('preprocessing_compatibility')
        
        # Check preprocessing metadata hash
        current_hash = self._calculate_preprocessing_hash(preprocessing_metadata)
        if model_metadata.preprocessing_metadata_hash != current_hash:
            results['errors'].append(
                "Preprocessing pipeline has changed since model training. "
                "Model may produce incorrect predictions."
            )
        
        # Check feature columns match
        model_features = set(model_metadata.feature_columns)
        preprocessing_features = set(preprocessing_metadata.feature_columns)
        
        if model_features != preprocessing_features:
            missing_in_preprocessing = model_features - preprocessing_features
            extra_in_preprocessing = preprocessing_features - model_features
            
            if missing_in_preprocessing:
                results['errors'].append(
                    f"Features missing in preprocessing: {list(missing_in_preprocessing)}"
                )
            
            if extra_in_preprocessing:
                results['warnings'].append(
                    f"Extra features in preprocessing: {list(extra_in_preprocessing)}"
                )
    
    def _validate_input_schema(self, model_metadata: ModelMetadata,
                             input_data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Validate input data schema against model expectations."""
        results['checks_performed'].append('input_schema')
        
        # Check required columns are present
        required_columns = set(model_metadata.feature_columns)
        available_columns = set(input_data.columns)
        
        missing_columns = required_columns - available_columns
        if missing_columns:
            results['errors'].append(f"Missing required columns: {list(missing_columns)}")
        
        # Check data types if available in metadata
        if hasattr(model_metadata, 'data_schema') and model_metadata.data_schema:
            for column, expected_type in model_metadata.data_schema.items():
                if column in input_data.columns:
                    actual_type = str(input_data[column].dtype)
                    if not self._types_compatible(actual_type, expected_type):
                        results['warnings'].append(
                            f"Column '{column}' type mismatch: expected {expected_type}, got {actual_type}"
                        )
    
    def _validate_feature_count(self, model: Any, input_data: pd.DataFrame,
                              results: Dict[str, Any]) -> None:
        """Validate feature count compatibility."""
        results['checks_performed'].append('feature_count')
        
        expected_features = self._get_expected_feature_count(model)
        if expected_features is not None:
            actual_features = input_data.shape[1]
            if expected_features != actual_features:
                results['errors'].append(
                    f"Feature count mismatch: model expects {expected_features}, "
                    f"got {actual_features}"
                )
    
    def _validate_model_type(self, model: Any, results: Dict[str, Any]) -> None:
        """Validate model type and check if it supports required methods."""
        results['checks_performed'].append('model_type')
        
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            results['errors'].append("Model does not have a 'predict' method")
        
        # Check for probability prediction capability
        if hasattr(model, 'predict_proba'):
            results['metadata'] = results.get('metadata', {})
            results['metadata']['supports_probabilities'] = True
        
        # Check for feature importance
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            results['metadata'] = results.get('metadata', {})
            results['metadata']['supports_feature_importance'] = True
    
    def _calculate_preprocessing_hash(self, metadata: PreprocessingMetadata) -> str:
        """Calculate hash of preprocessing metadata for compatibility checking."""
        # Create a deterministic representation of preprocessing steps
        steps_str = json.dumps(metadata.steps, sort_keys=True)
        columns_str = json.dumps(sorted(metadata.feature_columns))
        
        combined = f"{steps_str}|{columns_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if data types are compatible."""
        # Define type compatibility groups
        numeric_types = ['int64', 'int32', 'float64', 'float32', 'number']
        string_types = ['object', 'string', 'category']
        
        actual_group = None
        expected_group = None
        
        for group in [numeric_types, string_types]:
            if any(t in actual_type.lower() for t in group):
                actual_group = group
            if any(t in expected_type.lower() for t in group):
                expected_group = group
        
        return actual_group == expected_group
    
    def _get_expected_feature_count(self, model: Any) -> Optional[int]:
        """Get expected feature count from model."""
        # Try different ways to get feature count
        if hasattr(model, 'n_features_in_'):
            return model.n_features_in_
        elif hasattr(model, 'n_features_'):
            return model.n_features_
        elif hasattr(model, 'coef_') and hasattr(model.coef_, 'shape'):
            return model.coef_.shape[-1]
        elif hasattr(model, 'feature_importances_'):
            return len(model.feature_importances_)
        
        return None


class ModelCache:
    """Caches loaded models to improve performance."""
    
    def __init__(self, max_cache_size: int = 5):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_times = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_model(self, model_path: str) -> Optional[Tuple[Any, Optional[ModelMetadata]]]:
        """Get model from cache if available."""
        cache_key = str(Path(model_path).resolve())
        
        if cache_key in self.cache:
            self.access_times[cache_key] = datetime.now()
            self.logger.debug(f"Model loaded from cache: {cache_key}")
            return self.cache[cache_key]
        
        return None
    
    def cache_model(self, model_path: str, model: Any, metadata: Optional[ModelMetadata]) -> None:
        """Cache a loaded model."""
        cache_key = str(Path(model_path).resolve())
        
        # Remove oldest model if cache is full
        if len(self.cache) >= self.max_cache_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            self.logger.debug(f"Removed oldest model from cache: {oldest_key}")
        
        self.cache[cache_key] = (model, metadata)
        self.access_times[cache_key] = datetime.now()
        self.logger.debug(f"Model cached: {cache_key}")
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        self.cache.clear()
        self.access_times.clear()
        self.logger.info("Model cache cleared")


class ModelInferenceEngine(PipelineComponent):
    """Main inference engine that coordinates model loading, validation, and prediction."""
    
    def __init__(self, enable_caching: bool = True, cache_size: int = 5):
        super().__init__(ComponentType.MODEL_INFERENCE)
        self.model_loader = ModelLoader()
        self.model_validator = ModelValidator()
        self.model_cache = ModelCache(cache_size) if enable_caching else None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.current_model = None
        self.current_metadata = None
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate inference configuration."""
        try:
            inference_config = config.get('inference', {})
            
            if 'model_path' not in inference_config:
                self.logger.error("Model path not specified in inference configuration")
                return False
            
            model_path = Path(inference_config['model_path'])
            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup inference engine from configuration."""
        config = context.config.get('inference', {})
        
        if not self.validate_config(context.config):
            raise ConfigurationError("Invalid inference configuration")
        
        self.logger.info("Setting up model inference engine")
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute model inference."""
        try:
            # Load configuration
            inference_config = context.config.get('inference', {})
            model_path = inference_config['model_path']
            metadata_path = inference_config.get('metadata_path')
            preprocessing_pipeline_path = inference_config.get('preprocessing_pipeline_path')
            preprocessing_metadata_path = inference_config.get('preprocessing_metadata_path')
            
            # Load input data
            input_data = self._load_input_data(context)
            
            # Load preprocessing pipeline if specified
            if preprocessing_pipeline_path and preprocessing_metadata_path:
                self.preprocessor = DataPreprocessor()
                self.preprocessor.load_preprocessing_pipeline(
                    preprocessing_pipeline_path, preprocessing_metadata_path
                )
            
            # Load model
            model, model_metadata = self._load_model_with_cache(model_path, metadata_path)
            
            # Validate model compatibility
            validation_results = self._validate_model(
                model, model_metadata, input_data
            )
            
            if not validation_results['compatible']:
                error_msg = f"Model validation failed: {validation_results['errors']}"
                self.logger.error(error_msg)
                return ExecutionResult(
                    success=False,
                    artifacts=[],
                    metrics={},
                    metadata=validation_results,
                    error_message=error_msg
                )
            
            # Perform inference
            inference_result = self._perform_inference(
                model, input_data, inference_config
            )
            
            # Save results
            artifacts = self._save_inference_results(
                inference_result, context.artifacts_path
            )
            
            # Calculate metrics
            metrics = self._calculate_inference_metrics(inference_result, input_data)
            
            self.logger.info("Model inference completed successfully")
            
            return ExecutionResult(
                success=True,
                artifacts=artifacts,
                metrics=metrics,
                metadata={
                    'validation_results': validation_results,
                    'model_metadata': asdict(model_metadata) if model_metadata and hasattr(model_metadata, '__dataclass_fields__') else None,
                    'inference_metadata': inference_result.metadata
                }
            )
            
        except Exception as e:
            self.logger.error(f"Model inference failed: {str(e)}")
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def _load_input_data(self, context: ExecutionContext) -> pd.DataFrame:
        """Load input data for inference."""
        # Look for preprocessed data first, then raw data
        artifacts_path = Path(context.artifacts_path)
        
        # Try preprocessed data
        for split in ['test', 'val', 'train']:
            preprocessed_path = artifacts_path / f"{split}_preprocessed.parquet"
            if preprocessed_path.exists():
                df = pd.read_parquet(preprocessed_path)
                # Remove target column if present
                if 'target' in df.columns:
                    df = df.drop(columns=['target'])
                self.logger.info(f"Loaded {split} data for inference: {df.shape}")
                return df
        
        # Try raw data
        for filename in ['ingested_data.parquet', 'data.parquet']:
            data_path = artifacts_path / filename
            if data_path.exists():
                df = pd.read_parquet(data_path)
                self.logger.info(f"Loaded raw data for inference: {df.shape}")
                return df
        
        raise DataError("No input data found for inference")
    
    def _load_model_with_cache(self, model_path: str, metadata_path: Optional[str]) -> Tuple[Any, Optional[ModelMetadata]]:
        """Load model with caching support."""
        # Try cache first
        if self.model_cache:
            cached_result = self.model_cache.get_model(model_path)
            if cached_result:
                return cached_result
        
        # Load model
        model, metadata = self.model_loader.load_model(model_path, metadata_path)
        
        # Cache the result
        if self.model_cache:
            self.model_cache.cache_model(model_path, model, metadata)
        
        self.current_model = model
        self.current_metadata = metadata
        
        return model, metadata
    
    def _validate_model(self, model: Any, model_metadata: Optional[ModelMetadata],
                       input_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate model compatibility."""
        preprocessing_metadata = None
        if self.preprocessor and self.preprocessor.metadata:
            preprocessing_metadata = self.preprocessor.metadata
        
        return self.model_validator.validate_model_compatibility(
            model, model_metadata, preprocessing_metadata, input_data
        )
    
    def _perform_inference(self, model: Any, input_data: pd.DataFrame,
                          config: Dict[str, Any]) -> InferenceResult:
        """Perform model inference on input data."""
        start_time = datetime.now()
        
        # Preprocess data if preprocessor is available
        preprocessing_time = None
        if self.preprocessor:
            preprocess_start = datetime.now()
            processed_data = self.preprocessor.transform_new_data(input_data)
            preprocessing_time = (datetime.now() - preprocess_start).total_seconds()
        else:
            processed_data = input_data
        
        # Perform prediction
        inference_start = datetime.now()
        
        # Handle different model types
        predictions = self._predict_with_model(model, processed_data)
        
        # Get prediction probabilities if available
        probabilities = None
        confidence_scores = None
        
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(processed_data)
                # Calculate confidence as max probability
                confidence_scores = np.max(probabilities, axis=1)
            except Exception as e:
                self.logger.warning(f"Could not get prediction probabilities: {str(e)}")
        
        # Get feature importance if available
        feature_importance = self._get_feature_importance(model, processed_data)
        
        inference_time = (datetime.now() - inference_start).total_seconds()
        
        return InferenceResult(
            predictions=predictions,
            prediction_probabilities=probabilities,
            confidence_scores=confidence_scores,
            feature_importance=feature_importance,
            preprocessing_time=preprocessing_time,
            inference_time=inference_time,
            metadata={
                'input_shape': input_data.shape,
                'processed_shape': processed_data.shape,
                'prediction_shape': predictions.shape,
                'model_type': type(model).__name__
            }
        )
    
    def _predict_with_model(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Make predictions with the model, handling different model types."""
        try:
            # Standard sklearn-like interface
            if hasattr(model, 'predict'):
                return model.predict(data)
            else:
                raise ModelError(f"Model {type(model)} does not support prediction")
                
        except Exception as e:
            # Try to handle specific model types
            model_type = type(model).__name__.lower()
            
            if 'xgb' in model_type or 'xgboost' in model_type:
                try:
                    import xgboost as xgb
                    if isinstance(model, xgb.Booster):
                        dmatrix = xgb.DMatrix(data)
                        return model.predict(dmatrix)
                except ImportError:
                    pass
            
            raise ModelError(f"Failed to make predictions: {str(e)}")
    
    def _get_feature_importance(self, model: Any, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available."""
        try:
            feature_names = data.columns.tolist()
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                coef = np.abs(model.coef_).flatten()
                return dict(zip(feature_names, coef))
            
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {str(e)}")
        
        return None
    
    def _save_inference_results(self, result: InferenceResult, artifacts_path: str) -> List[str]:
        """Save inference results to files."""
        artifacts_path = Path(artifacts_path)
        artifacts = []
        
        # Save predictions
        predictions_df = pd.DataFrame({'predictions': result.predictions})
        
        if result.prediction_probabilities is not None:
            # Add probability columns
            prob_cols = [f'prob_class_{i}' for i in range(result.prediction_probabilities.shape[1])]
            prob_df = pd.DataFrame(result.prediction_probabilities, columns=prob_cols)
            predictions_df = pd.concat([predictions_df, prob_df], axis=1)
        
        if result.confidence_scores is not None:
            predictions_df['confidence'] = result.confidence_scores
        
        predictions_path = artifacts_path / "predictions.parquet"
        predictions_df.to_parquet(predictions_path, index=False)
        artifacts.append(str(predictions_path))
        
        # Save feature importance if available
        if result.feature_importance:
            importance_df = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in result.feature_importance.items()
            ]).sort_values('importance', ascending=False)
            
            importance_path = artifacts_path / "feature_importance.parquet"
            importance_df.to_parquet(importance_path, index=False)
            artifacts.append(str(importance_path))
        
        # Save inference metadata
        metadata_dict = asdict(result)
        # Convert numpy arrays to lists for JSON serialization
        for key, value in metadata_dict.items():
            if isinstance(value, np.ndarray):
                metadata_dict[key] = value.tolist()
        
        metadata_path = artifacts_path / "inference_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        artifacts.append(str(metadata_path))
        
        return artifacts
    
    def _calculate_inference_metrics(self, result: InferenceResult, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate inference performance metrics."""
        metrics = {
            'num_predictions': len(result.predictions),
            'inference_time_seconds': result.inference_time,
            'predictions_per_second': len(result.predictions) / result.inference_time if result.inference_time else 0,
        }
        
        if result.preprocessing_time:
            metrics['preprocessing_time_seconds'] = result.preprocessing_time
            metrics['total_time_seconds'] = result.preprocessing_time + result.inference_time
        
        if result.confidence_scores is not None:
            metrics['mean_confidence'] = float(np.mean(result.confidence_scores))
            metrics['min_confidence'] = float(np.min(result.confidence_scores))
            metrics['max_confidence'] = float(np.max(result.confidence_scores))
        
        return metrics
    
    def predict_single(self, input_data: Union[pd.DataFrame, Dict[str, Any]]) -> InferenceResult:
        """Perform inference on a single sample or small batch."""
        if self.current_model is None:
            raise RuntimeError("No model loaded. Call execute() first or load a model.")
        
        # Convert dict to DataFrame if needed
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        return self._perform_inference(self.current_model, input_data, {})
    
    def clear_cache(self) -> None:
        """Clear model cache."""
        if self.model_cache:
            self.model_cache.clear_cache()


@dataclass
class BatchInferenceConfig:
    """Configuration for batch inference."""
    chunk_size: int = 1000
    max_workers: int = 4
    progress_callback: Optional[Callable[[int, int], None]] = None
    save_intermediate: bool = False
    output_format: str = 'parquet'  # 'parquet', 'csv', 'json'


@dataclass
class BatchInferenceResult:
    """Result of batch inference."""
    total_predictions: int
    processing_time: float
    chunks_processed: int
    failed_chunks: int
    output_files: List[str]
    summary_stats: Dict[str, Any]
    error_details: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.error_details is None:
            self.error_details = []


class BatchInferenceEngine:
    """Engine for processing large datasets in batches with progress tracking."""
    
    def __init__(self, model_engine: ModelInferenceEngine):
        self.model_engine = model_engine
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop_processing = False
    
    def predict_batch(self, input_data: pd.DataFrame, 
                     config: BatchInferenceConfig,
                     output_path: str) -> BatchInferenceResult:
        """Perform batch inference on large dataset with chunking and progress tracking."""
        start_time = time.time()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate chunks
        total_rows = len(input_data)
        chunk_size = min(config.chunk_size, total_rows)
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        self.logger.info(f"Starting batch inference: {total_rows} rows, {total_chunks} chunks")
        
        # Initialize results tracking
        processed_chunks = 0
        failed_chunks = 0
        output_files = []
        all_predictions = []
        all_probabilities = []
        all_confidence_scores = []
        error_details = []
        
        # Process chunks
        if config.max_workers > 1:
            # Parallel processing
            results = self._process_chunks_parallel(
                input_data, chunk_size, config, output_path
            )
        else:
            # Sequential processing
            results = self._process_chunks_sequential(
                input_data, chunk_size, config, output_path
            )
        
        # Collect results
        for result in results:
            if result['success']:
                processed_chunks += 1
                if result['output_file']:
                    output_files.append(result['output_file'])
                if result['predictions'] is not None:
                    all_predictions.extend(result['predictions'])
                if result['probabilities'] is not None:
                    all_probabilities.extend(result['probabilities'])
                if result['confidence_scores'] is not None:
                    all_confidence_scores.extend(result['confidence_scores'])
            else:
                failed_chunks += 1
                error_details.append({
                    'chunk_id': result['chunk_id'],
                    'error': result['error'],
                    'rows': result['rows']
                })
        
        # Save consolidated results if requested
        if not config.save_intermediate and all_predictions:
            consolidated_file = self._save_consolidated_results(
                all_predictions, all_probabilities, all_confidence_scores,
                output_path, config.output_format
            )
            output_files = [consolidated_file]
        
        # Calculate summary statistics
        summary_stats = self._calculate_batch_summary_stats(
            all_predictions, all_probabilities, all_confidence_scores
        )
        
        processing_time = time.time() - start_time
        
        self.logger.info(
            f"Batch inference completed: {processed_chunks}/{total_chunks} chunks successful, "
            f"{failed_chunks} failed, {processing_time:.2f}s"
        )
        
        return BatchInferenceResult(
            total_predictions=len(all_predictions),
            processing_time=processing_time,
            chunks_processed=processed_chunks,
            failed_chunks=failed_chunks,
            output_files=output_files,
            summary_stats=summary_stats,
            error_details=error_details
        )
    
    def _process_chunks_sequential(self, input_data: pd.DataFrame, chunk_size: int,
                                 config: BatchInferenceConfig, output_path: Path) -> List[Dict]:
        """Process chunks sequentially."""
        results = []
        total_chunks = (len(input_data) + chunk_size - 1) // chunk_size
        
        for chunk_id in range(total_chunks):
            if self._stop_processing:
                break
                
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, len(input_data))
            chunk_data = input_data.iloc[start_idx:end_idx]
            
            result = self._process_single_chunk(chunk_id, chunk_data, config, output_path)
            results.append(result)
            
            # Progress callback
            if config.progress_callback:
                config.progress_callback(chunk_id + 1, total_chunks)
        
        return results
    
    def _process_chunks_parallel(self, input_data: pd.DataFrame, chunk_size: int,
                               config: BatchInferenceConfig, output_path: Path) -> List[Dict]:
        """Process chunks in parallel using ThreadPoolExecutor."""
        results = []
        total_chunks = (len(input_data) + chunk_size - 1) // chunk_size
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {}
            for chunk_id in range(total_chunks):
                start_idx = chunk_id * chunk_size
                end_idx = min(start_idx + chunk_size, len(input_data))
                chunk_data = input_data.iloc[start_idx:end_idx]
                
                future = executor.submit(
                    self._process_single_chunk, chunk_id, chunk_data, config, output_path
                )
                future_to_chunk[future] = chunk_id
            
            # Collect results as they complete
            completed_chunks = 0
            for future in as_completed(future_to_chunk):
                if self._stop_processing:
                    # Cancel remaining futures
                    for f in future_to_chunk:
                        f.cancel()
                    break
                
                chunk_id = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Chunk {chunk_id} failed: {str(e)}")
                    results.append({
                        'chunk_id': chunk_id,
                        'success': False,
                        'error': str(e),
                        'predictions': None,
                        'probabilities': None,
                        'confidence_scores': None,
                        'output_file': None,
                        'rows': 0
                    })
                
                completed_chunks += 1
                
                # Progress callback
                if config.progress_callback:
                    config.progress_callback(completed_chunks, total_chunks)
        
        # Sort results by chunk_id to maintain order
        results.sort(key=lambda x: x['chunk_id'])
        return results
    
    def _process_single_chunk(self, chunk_id: int, chunk_data: pd.DataFrame,
                            config: BatchInferenceConfig, output_path: Path) -> Dict:
        """Process a single chunk of data."""
        try:
            # Perform inference on chunk
            inference_result = self.model_engine._perform_inference(
                self.model_engine.current_model, chunk_data, {}
            )
            
            output_file = None
            if config.save_intermediate:
                # Save chunk results
                output_file = self._save_chunk_results(
                    chunk_id, inference_result, output_path, config.output_format
                )
            
            return {
                'chunk_id': chunk_id,
                'success': True,
                'error': None,
                'predictions': inference_result.predictions.tolist() if inference_result.predictions is not None else None,
                'probabilities': inference_result.prediction_probabilities.tolist() if inference_result.prediction_probabilities is not None else None,
                'confidence_scores': inference_result.confidence_scores.tolist() if inference_result.confidence_scores is not None else None,
                'output_file': output_file,
                'rows': len(chunk_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            return {
                'chunk_id': chunk_id,
                'success': False,
                'error': str(e),
                'predictions': None,
                'probabilities': None,
                'confidence_scores': None,
                'output_file': None,
                'rows': len(chunk_data)
            }
    
    def _save_chunk_results(self, chunk_id: int, result: InferenceResult,
                          output_path: Path, output_format: str) -> str:
        """Save results for a single chunk."""
        # Create DataFrame with results
        df = pd.DataFrame({'predictions': result.predictions})
        
        if result.prediction_probabilities is not None:
            prob_cols = [f'prob_class_{i}' for i in range(result.prediction_probabilities.shape[1])]
            prob_df = pd.DataFrame(result.prediction_probabilities, columns=prob_cols)
            df = pd.concat([df, prob_df], axis=1)
        
        if result.confidence_scores is not None:
            df['confidence'] = result.confidence_scores
        
        # Save based on format
        if output_format == 'parquet':
            file_path = output_path / f"chunk_{chunk_id:06d}.parquet"
            df.to_parquet(file_path, index=False)
        elif output_format == 'csv':
            file_path = output_path / f"chunk_{chunk_id:06d}.csv"
            df.to_csv(file_path, index=False)
        elif output_format == 'json':
            file_path = output_path / f"chunk_{chunk_id:06d}.json"
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return str(file_path)
    
    def _save_consolidated_results(self, predictions: List, probabilities: List,
                                 confidence_scores: List, output_path: Path,
                                 output_format: str) -> str:
        """Save consolidated results from all chunks."""
        df = pd.DataFrame({'predictions': predictions})
        
        if probabilities and len(probabilities) > 0:
            # Assume all probability arrays have same shape
            prob_array = np.array(probabilities)
            prob_cols = [f'prob_class_{i}' for i in range(prob_array.shape[1])]
            prob_df = pd.DataFrame(prob_array, columns=prob_cols)
            df = pd.concat([df, prob_df], axis=1)
        
        if confidence_scores:
            df['confidence'] = confidence_scores
        
        # Save based on format
        if output_format == 'parquet':
            file_path = output_path / "batch_predictions.parquet"
            df.to_parquet(file_path, index=False)
        elif output_format == 'csv':
            file_path = output_path / "batch_predictions.csv"
            df.to_csv(file_path, index=False)
        elif output_format == 'json':
            file_path = output_path / "batch_predictions.json"
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return str(file_path)
    
    def _calculate_batch_summary_stats(self, predictions: List, probabilities: List,
                                     confidence_scores: List) -> Dict[str, Any]:
        """Calculate summary statistics for batch inference."""
        stats = {
            'total_predictions': len(predictions),
            'unique_predictions': len(set(predictions)) if predictions else 0
        }
        
        if predictions:
            pred_array = np.array(predictions)
            if pred_array.dtype in [np.int32, np.int64, np.float32, np.float64]:
                stats.update({
                    'prediction_mean': float(np.mean(pred_array)),
                    'prediction_std': float(np.std(pred_array)),
                    'prediction_min': float(np.min(pred_array)),
                    'prediction_max': float(np.max(pred_array))
                })
        
        if confidence_scores:
            conf_array = np.array(confidence_scores)
            stats.update({
                'confidence_mean': float(np.mean(conf_array)),
                'confidence_std': float(np.std(conf_array)),
                'confidence_min': float(np.min(conf_array)),
                'confidence_max': float(np.max(conf_array)),
                'low_confidence_count': int(np.sum(conf_array < 0.5)),
                'high_confidence_count': int(np.sum(conf_array > 0.8))
            })
        
        return stats
    
    def stop_processing(self):
        """Stop batch processing gracefully."""
        self._stop_processing = True
        self.logger.info("Batch processing stop requested")


@dataclass
class RealTimeInferenceConfig:
    """Configuration for real-time inference."""
    max_batch_size: int = 32
    timeout_seconds: float = 1.0
    confidence_threshold: float = 0.5
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    max_queue_size: int = 1000


class RealTimeInferenceEngine:
    """Engine for real-time inference with low latency and high throughput."""
    
    def __init__(self, model_engine: ModelInferenceEngine, config: RealTimeInferenceConfig):
        self.model_engine = model_engine
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Request queue for batching
        self.request_queue = Queue(maxsize=config.max_queue_size)
        self.response_queues = {}
        
        # Caching
        self.prediction_cache = {} if config.enable_caching else None
        self.cache_timestamps = {} if config.enable_caching else None
        
        # Processing thread
        self.processing_thread = None
        self.is_running = False
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_latency': 0.0,
            'throughput_per_second': 0.0
        }
        self.latency_history = []
        self.last_metrics_update = time.time()
    
    def start(self):
        """Start the real-time inference engine."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.processing_thread.start()
        self.logger.info("Real-time inference engine started")
    
    def stop(self):
        """Stop the real-time inference engine."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.logger.info("Real-time inference engine stopped")
    
    def predict_single(self, input_data: Union[Dict[str, Any], pd.DataFrame],
                      request_id: Optional[str] = None) -> Dict[str, Any]:
        """Make a single prediction with low latency."""
        if not self.is_running:
            raise RuntimeError("Real-time inference engine is not running. Call start() first.")
        
        request_id = request_id or f"req_{int(time.time() * 1000000)}"
        start_time = time.time()
        
        # Check cache first
        cache_key = None
        if self.prediction_cache is not None:
            cache_key = self._generate_cache_key(input_data)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.metrics['cache_hits'] += 1
                self.metrics['total_requests'] += 1
                return {
                    'request_id': request_id,
                    'predictions': cached_result['predictions'],
                    'probabilities': cached_result.get('probabilities'),
                    'confidence': cached_result.get('confidence'),
                    'latency_ms': (time.time() - start_time) * 1000,
                    'from_cache': True,
                    'success': True
                }
            else:
                self.metrics['cache_misses'] += 1
        
        # Create response queue for this request
        response_queue = Queue(maxsize=1)
        self.response_queues[request_id] = response_queue
        
        # Add request to processing queue
        try:
            request = {
                'id': request_id,
                'data': input_data,
                'timestamp': start_time
            }
            self.request_queue.put(request, timeout=self.config.timeout_seconds)
        except:
            del self.response_queues[request_id]
            raise RuntimeError("Request queue is full")
        
        # Wait for response
        try:
            response = response_queue.get(timeout=self.config.timeout_seconds)
            latency = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics['total_requests'] += 1
            if response.get('success', False):
                self.metrics['successful_predictions'] += 1
                
                # Cache result if enabled and we have a cache key
                if self.prediction_cache is not None and cache_key and response.get('predictions') is not None:
                    self._add_to_cache(cache_key, response)
            else:
                self.metrics['failed_predictions'] += 1
            
            self.latency_history.append(latency)
            if len(self.latency_history) > 1000:  # Keep last 1000 measurements
                self.latency_history = self.latency_history[-1000:]
            
            response['latency_ms'] = latency
            response['from_cache'] = False
            return response
            
        except Exception as e:
            self.metrics['total_requests'] += 1
            self.metrics['failed_predictions'] += 1
            raise RuntimeError(f"Prediction timeout: {str(e)}")
        finally:
            # Cleanup
            if request_id in self.response_queues:
                del self.response_queues[request_id]
    
    def predict_batch_realtime(self, input_batch: List[Union[Dict[str, Any], pd.DataFrame]],
                             request_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Make predictions on a batch of inputs for higher throughput."""
        if not request_ids:
            request_ids = [f"batch_req_{i}_{int(time.time() * 1000000)}" 
                          for i in range(len(input_batch))]
        
        if len(input_batch) != len(request_ids):
            raise ValueError("Number of inputs must match number of request IDs")
        
        # Process each request
        results = []
        for input_data, request_id in zip(input_batch, request_ids):
            try:
                result = self.predict_single(input_data, request_id)
                results.append(result)
            except Exception as e:
                results.append({
                    'request_id': request_id,
                    'success': False,
                    'error': str(e),
                    'predictions': None,
                    'probabilities': None,
                    'confidence': None,
                    'latency_ms': 0,
                    'from_cache': False
                })
        
        return results
    
    def _process_requests(self):
        """Main processing loop for handling requests."""
        batch_requests = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Collect requests for batching
                timeout = min(0.1, self.config.timeout_seconds)  # Use smaller timeout for responsiveness
                
                try:
                    request = self.request_queue.get(timeout=timeout)
                    batch_requests.append(request)
                except:
                    pass  # Timeout, process current batch if any
                
                # Process batch if conditions are met
                should_process = (
                    len(batch_requests) >= self.config.max_batch_size or
                    (batch_requests and time.time() - last_batch_time >= self.config.timeout_seconds) or
                    (batch_requests and len(batch_requests) > 0)  # Process any pending requests
                )
                
                if should_process and batch_requests:
                    self._process_batch(batch_requests)
                    batch_requests = []
                    last_batch_time = time.time()
                    
            except Exception as e:
                self.logger.error(f"Error in request processing loop: {str(e)}")
                # Send error responses for current batch
                for request in batch_requests:
                    self._send_error_response(request['id'], str(e))
                batch_requests = []
    
    def _process_batch(self, requests: List[Dict]):
        """Process a batch of requests."""
        if not requests:
            return
        
        try:
            # Check if model is available
            if not self.model_engine.current_model:
                for request in requests:
                    self._send_error_response(request['id'], "No model loaded")
                return
            
            # Prepare batch data
            batch_data = []
            for request in requests:
                if isinstance(request['data'], dict):
                    batch_data.append(pd.DataFrame([request['data']]))
                else:
                    batch_data.append(request['data'])
            
            # Combine into single DataFrame
            if len(batch_data) == 1:
                combined_data = batch_data[0]
            else:
                combined_data = pd.concat(batch_data, ignore_index=True)
            
            # Perform inference
            inference_result = self.model_engine._perform_inference(
                self.model_engine.current_model, combined_data, {}
            )
            
            # Split results back to individual requests
            for i, request in enumerate(requests):
                try:
                    # Handle single prediction case
                    if len(inference_result.predictions) == 1 and len(requests) == 1:
                        pred_idx = 0
                    else:
                        pred_idx = i
                    
                    response = {
                        'request_id': request['id'],
                        'success': True,
                        'predictions': inference_result.predictions[pred_idx] if inference_result.predictions is not None else None,
                        'probabilities': inference_result.prediction_probabilities[pred_idx].tolist() if inference_result.prediction_probabilities is not None else None,
                        'confidence': inference_result.confidence_scores[pred_idx] if inference_result.confidence_scores is not None else None
                    }
                    
                    # Check confidence threshold
                    if (response['confidence'] is not None and 
                        response['confidence'] < self.config.confidence_threshold):
                        response['low_confidence_warning'] = True
                    
                    self._send_response(request['id'], response)
                    
                except Exception as e:
                    self._send_error_response(request['id'], f"Error processing individual result: {str(e)}")
                    
        except Exception as e:
            # Send error to all requests in batch
            for request in requests:
                self._send_error_response(request['id'], f"Batch processing error: {str(e)}")
    
    def _send_response(self, request_id: str, response: Dict):
        """Send response to the appropriate queue."""
        if request_id in self.response_queues:
            try:
                self.response_queues[request_id].put(response, timeout=0.1)
            except:
                self.logger.warning(f"Failed to send response for request {request_id}")
    
    def _send_error_response(self, request_id: str, error_message: str):
        """Send error response to the appropriate queue."""
        response = {
            'request_id': request_id,
            'success': False,
            'error': error_message,
            'predictions': None,
            'probabilities': None,
            'confidence': None
        }
        self._send_response(request_id, response)
    
    def _generate_cache_key(self, input_data: Union[Dict, pd.DataFrame]) -> str:
        """Generate cache key for input data."""
        if isinstance(input_data, dict):
            # Sort keys for consistent hashing
            sorted_items = sorted(input_data.items())
            data_str = json.dumps(sorted_items, sort_keys=True, default=str)
        else:
            # For DataFrame, use hash of values
            data_str = str(input_data.values.tobytes())
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get prediction from cache if not expired."""
        if not self.prediction_cache or cache_key not in self.prediction_cache:
            return None
        
        # Check if cache entry is expired
        if cache_key in self.cache_timestamps:
            age = time.time() - self.cache_timestamps[cache_key]
            if age > self.config.cache_ttl_seconds:
                del self.prediction_cache[cache_key]
                del self.cache_timestamps[cache_key]
                return None
        
        return self.prediction_cache[cache_key]
    
    def _add_to_cache(self, cache_key: str, response: Dict):
        """Add prediction to cache."""
        if self.prediction_cache is None:
            return
        
        # Clean up expired entries periodically
        current_time = time.time()
        if len(self.prediction_cache) > 1000:  # Cleanup when cache gets large
            expired_keys = [
                key for key, timestamp in self.cache_timestamps.items()
                if current_time - timestamp > self.config.cache_ttl_seconds
            ]
            for key in expired_keys:
                self.prediction_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
        
        # Add new entry
        self.prediction_cache[cache_key] = {
            'predictions': response.get('predictions'),
            'probabilities': response.get('probabilities'),
            'confidence': response.get('confidence')
        }
        self.cache_timestamps[cache_key] = current_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        current_time = time.time()
        
        # Calculate average latency
        if self.latency_history:
            self.metrics['average_latency'] = np.mean(self.latency_history)
        
        # Calculate throughput
        time_elapsed = current_time - self.last_metrics_update
        if time_elapsed > 0:
            requests_in_period = self.metrics['total_requests']
            self.metrics['throughput_per_second'] = requests_in_period / time_elapsed
        
        # Add cache statistics
        if self.prediction_cache is not None:
            self.metrics['cache_size'] = len(self.prediction_cache)
            total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
            if total_cache_requests > 0:
                self.metrics['cache_hit_rate'] = self.metrics['cache_hits'] / total_cache_requests
            else:
                self.metrics['cache_hit_rate'] = 0.0
        else:
            self.metrics['cache_size'] = 0
            self.metrics['cache_hit_rate'] = 0.0
        
        return self.metrics.copy()
    
    def clear_cache(self):
        """Clear prediction cache."""
        if self.prediction_cache:
            self.prediction_cache.clear()
            self.cache_timestamps.clear()
            self.logger.info("Prediction cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the real-time inference engine."""
        return {
            'status': 'healthy' if self.is_running else 'stopped',
            'queue_size': self.request_queue.qsize(),
            'active_requests': len(self.response_queues),
            'cache_size': len(self.prediction_cache) if self.prediction_cache else 0,
            'metrics': self.get_metrics()
        }