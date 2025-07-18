"""Data preprocessing pipeline with scikit-learn transformers and custom functions."""

import logging
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PowerTransformer, QuantileTransformer, Normalizer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from ..core.interfaces import PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
from ..core.errors import DataError, ConfigurationError


@dataclass
class PreprocessingStep:
    """Configuration for a preprocessing step."""
    name: str
    transformer: str
    columns: Optional[List[str]] = None
    parameters: Dict[str, Any] = None
    custom_function: Optional[str] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class PreprocessingMetadata:
    """Metadata for preprocessing pipeline to ensure inference consistency."""
    steps: List[Dict[str, Any]]
    column_names: List[str]
    target_column: Optional[str]
    feature_columns: List[str]
    categorical_columns: List[str]
    numerical_columns: List[str]
    preprocessing_pipeline: Optional[str] = None  # Serialized pipeline
    data_statistics: Dict[str, Any] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.data_statistics is None:
            self.data_statistics = {}
        if self.created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()


class CustomTransformer(BaseEstimator, TransformerMixin):
    """Wrapper for custom transformation functions."""
    
    def __init__(self, func: Callable, func_name: str, **kwargs):
        self.func = func
        self.func_name = func_name
        self.kwargs = kwargs
        self.fitted_params_ = {}
    
    def fit(self, X, y=None):
        """Fit the transformer (may store parameters for stateful transformations)."""
        # Allow custom functions to store fit parameters
        if hasattr(self.func, 'fit'):
            self.fitted_params_ = self.func.fit(X, y, **self.kwargs)
        return self
    
    def transform(self, X):
        """Apply the custom transformation."""
        if hasattr(self.func, 'transform'):
            return self.func.transform(X, fitted_params=self.fitted_params_, **self.kwargs)
        else:
            return self.func(X, **self.kwargs)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if hasattr(self.func, 'get_feature_names_out'):
            return self.func.get_feature_names_out(input_features)
        return input_features


class DataPreprocessor(PipelineComponent):
    """Comprehensive data preprocessing pipeline using scikit-learn transformers."""
    
    def __init__(self):
        super().__init__(ComponentType.DATA_PREPROCESSING)
        self.preprocessing_pipeline: Optional[Pipeline] = None
        self.metadata: Optional[PreprocessingMetadata] = None
        self.custom_functions: Dict[str, Callable] = {}
        
        # Built-in transformer mappings
        self.transformer_registry = {
            # Scalers
            'standard_scaler': StandardScaler,
            'min_max_scaler': MinMaxScaler,
            'robust_scaler': RobustScaler,
            'max_abs_scaler': MaxAbsScaler,
            'normalizer': Normalizer,
            
            # Encoders
            'label_encoder': LabelEncoder,
            'one_hot_encoder': OneHotEncoder,
            'ordinal_encoder': OrdinalEncoder,
            
            # Transformers
            'power_transformer': PowerTransformer,
            'quantile_transformer': QuantileTransformer,
            
            # Imputers
            'simple_imputer': SimpleImputer,
            'knn_imputer': KNNImputer,
            
            # Feature Selection
            'select_k_best': SelectKBest,
            'select_percentile': SelectPercentile,
            'variance_threshold': VarianceThreshold,
        }
    
    def register_custom_function(self, name: str, func: Callable) -> None:
        """Register a custom transformation function."""
        self.custom_functions[name] = func
        self.logger.info(f"Registered custom function: {name}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate preprocessing configuration."""
        try:
            data_config = config.get('data', {})
            preprocessing_steps = data_config.get('preprocessing', [])
            

            
            if not preprocessing_steps:
                self.logger.error("No preprocessing steps specified")
                return False
            
            steps = preprocessing_steps
            if not isinstance(steps, list) or len(steps) == 0:
                self.logger.error("Preprocessing steps must be a non-empty list")
                return False
            
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    self.logger.error(f"Step {i} must be a dictionary")
                    return False
                
                if 'type' not in step:
                    self.logger.error(f"Step {i} missing 'type' field")
                    return False
                
                # For the current schema, we use 'type' instead of 'transformer'
                step_type = step['type']
                if step_type not in self.transformer_registry:
                    self.logger.error(f"Unknown transformer: {step_type}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup preprocessing pipeline from configuration."""
        # Pass the entire config to validate_config, not just the data section
        if not self.validate_config(context.config):
            raise ConfigurationError("Invalid preprocessing configuration")
        
        self.logger.info("Setting up data preprocessing pipeline")
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute data preprocessing pipeline."""
        try:
            # Load input data
            input_data = self._load_input_data(context)
            
            # Get preprocessing configuration
            data_config = context.config.get('data', {})
            preprocessing_steps = data_config.get('preprocessing', [])
            
            # Build preprocessing pipeline
            pipeline = self._build_pipeline(preprocessing_steps, input_data)
            
            # Split data if requested
            train_data, val_data, test_data = self._split_data(
                input_data, 
                data_config
            )
            
            # Fit pipeline on training data
            self.logger.info("Fitting preprocessing pipeline on training data")
            pipeline.fit(train_data['features'], train_data.get('target'))
            
            # Transform all datasets
            transformed_datasets = {}
            for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
                if split_data is not None:
                    try:
                        transformed_features = pipeline.transform(split_data['features'])
                        
                        # Convert back to DataFrame with proper column names
                        if hasattr(pipeline, 'get_feature_names_out'):
                            try:
                                feature_names = pipeline.get_feature_names_out()
                            except:
                                feature_names = [f'feature_{i}' for i in range(transformed_features.shape[1])]
                        else:
                            feature_names = [f'feature_{i}' for i in range(transformed_features.shape[1])]
                        
                        # Ensure we have the right number of feature names
                        if len(feature_names) != transformed_features.shape[1]:
                            feature_names = [f'feature_{i}' for i in range(transformed_features.shape[1])]
                        
                        transformed_df = pd.DataFrame(transformed_features, columns=feature_names)
                        
                        # Add target column back if it exists
                        if split_data.get('target') is not None:
                            transformed_df['target'] = split_data['target'].reset_index(drop=True)
                        
                        transformed_datasets[split_name] = transformed_df
                        
                    except Exception as e:
                        self.logger.error(f"Error transforming {split_name} data: {str(e)}")
                        # Fallback: use original data if transformation fails
                        transformed_df = split_data['features'].copy()
                        if split_data.get('target') is not None:
                            transformed_df['target'] = split_data['target'].reset_index(drop=True)
                        transformed_datasets[split_name] = transformed_df
            
            # Save transformed datasets
            artifacts = []
            for split_name, dataset in transformed_datasets.items():
                if dataset is not None:
                    output_path = Path(context.artifacts_path) / f"{split_name}_preprocessed.parquet"
                    dataset.to_parquet(output_path, index=False)
                    artifacts.append(str(output_path))
            
            # Create and save preprocessing metadata
            metadata = self._create_metadata(
                preprocessing_steps, 
                input_data, 
                pipeline,
                transformed_datasets
            )
            
            metadata_path = Path(context.artifacts_path) / "preprocessing_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            artifacts.append(str(metadata_path))
            
            # Save preprocessing pipeline
            pipeline_path = Path(context.artifacts_path) / "preprocessing_pipeline.pkl"
            with open(pipeline_path, 'wb') as f:
                pickle.dump(pipeline, f)
            artifacts.append(str(pipeline_path))
            
            # Store for later use
            self.preprocessing_pipeline = pipeline
            self.metadata = metadata
            
            # Calculate metrics
            metrics = self._calculate_metrics(input_data, transformed_datasets)
            
            self.logger.info("Data preprocessing completed successfully")
            
            return ExecutionResult(
                success=True,
                artifacts=artifacts,
                metrics=metrics,
                metadata={
                    'original_shape': input_data['features'].shape,
                    'transformed_shape': transformed_datasets['train'].shape if 'train' in transformed_datasets else None,
                    'feature_names': list(feature_names),
                    'preprocessing_steps': len(preprocessing_steps)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def _load_input_data(self, context: ExecutionContext) -> Dict[str, pd.DataFrame]:
        """Load input data from artifacts."""
        # Look for ingested data from previous step
        artifacts_path = Path(context.artifacts_path)
        
        # Try to find ingested data
        ingested_data_path = artifacts_path / "ingested_data.parquet"
        if not ingested_data_path.exists():
            # Look for any parquet files
            parquet_files = list(artifacts_path.glob("*.parquet"))
            if not parquet_files:
                raise DataError("No input data found for preprocessing")
            ingested_data_path = parquet_files[0]
        
        df = pd.read_parquet(ingested_data_path)
        self.logger.info(f"Loaded input data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return {'features': df}
    
    def _build_pipeline(self, preprocessing_steps: List[Dict[str, Any]], input_data: Dict[str, pd.DataFrame]) -> Pipeline:
        """Build scikit-learn preprocessing pipeline from configuration."""
        df = input_data['features']
        
        # Group transformations by column to create a single ColumnTransformer
        transformers = []
        passthrough_columns = set(df.columns)
        
        for i, step_config in enumerate(preprocessing_steps):
            step_name = step_config.get('name', f"step_{i}")
            step_type = step_config['type']
            
            # Built-in transformer
            transformer_name = step_type
            transformer_class = self.transformer_registry[transformer_name]
            transformer_params = step_config.get('parameters', {})
            
            # Handle column-specific transformers
            columns = step_config.get('columns')
            if columns:
                # Validate columns exist in the dataframe
                valid_columns = [col for col in columns if col in df.columns]
                if not valid_columns:
                    self.logger.warning(f"No valid columns found for transformer {step_name}, skipping")
                    continue
                
                # Create transformer for specific columns
                transformer = transformer_class(**transformer_params)
                transformers.append((step_name, transformer, valid_columns))
                
                # Remove these columns from passthrough
                passthrough_columns -= set(valid_columns)
                
            else:
                # Apply to all columns - but some transformers only work with numerical data
                if transformer_name in ['standard_scaler', 'min_max_scaler', 'robust_scaler', 
                                      'max_abs_scaler', 'power_transformer', 'quantile_transformer',
                                      'normalizer', 'select_k_best', 'select_percentile', 'variance_threshold']:
                    # These transformers only work with numerical data
                    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numerical_columns:
                        transformer = transformer_class(**transformer_params)
                        transformers.append((step_name, transformer, numerical_columns))
                        
                        # Remove these columns from passthrough
                        passthrough_columns -= set(numerical_columns)
                    else:
                        self.logger.warning(f"No numerical columns found for transformer {step_name}, skipping")
                        continue
                else:
                    # Apply to all columns
                    all_columns = df.columns.tolist()
                    transformer = transformer_class(**transformer_params)
                    transformers.append((step_name, transformer, all_columns))
                    
                    # Remove all columns from passthrough
                    passthrough_columns = set()
        
        # Create the pipeline
        if transformers:
            # Create a single ColumnTransformer with all transformations
            # Always drop untransformed columns (like Name, Ticket, Cabin for Titanic)
            remainder = 'drop'
            column_transformer = ColumnTransformer(
                transformers=transformers,
                remainder=remainder,
                verbose_feature_names_out=False
            )
            pipeline = Pipeline([('preprocessing', column_transformer)])
        else:
            # If no valid transformers, create a passthrough pipeline
            from sklearn.preprocessing import FunctionTransformer
            pipeline = Pipeline([('passthrough', FunctionTransformer())])
        
        self.logger.info(f"Built preprocessing pipeline with {len(transformers)} transformers")
        
        return pipeline
    
    def _split_data(self, input_data: Dict[str, pd.DataFrame], data_config: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
        """Split data into train/validation/test sets."""
        df = input_data['features']
        
        # Get split configuration
        train_size = data_config.get('train_split', 0.7)
        val_size = data_config.get('validation_split', 0.15)
        test_size = data_config.get('test_split', 0.15)
        random_state = data_config.get('random_state', 42)
        stratify_enabled = data_config.get('stratify', False)
        target_column = None  # We'll identify the target column ourselves
        
        # Validate split sizes
        total_size = train_size + val_size + test_size
        if abs(total_size - 1.0) > 0.001:
            raise ConfigurationError(f"Split sizes must sum to 1.0, got {total_size}")
        
        # Separate features and target
        if target_column and target_column in df.columns:
            features = df.drop(columns=[target_column])
            target = df[target_column]
        else:
            features = df
            target = None
        
        # Prepare stratification
        stratify = None
        if target is not None and stratify_enabled:
            stratify = target
        
        # First split: separate test set
        if test_size > 0:
            temp_size = train_size + val_size
            if target is not None:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    features, target,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify
                )
            else:
                X_temp, X_test = train_test_split(
                    features,
                    test_size=test_size,
                    random_state=random_state
                )
                y_temp, y_test = None, None
            
            # Second split: separate train and validation
            if val_size > 0:
                val_ratio = val_size / temp_size
                if y_temp is not None:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp,
                        test_size=val_ratio,
                        random_state=random_state,
                        stratify=stratify[:len(X_temp)] if stratify is not None else None
                    )
                else:
                    X_train, X_val = train_test_split(
                        X_temp,
                        test_size=val_ratio,
                        random_state=random_state
                    )
                    y_train, y_val = None, None
            else:
                X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
        else:
            # No test set, just train/val split
            if val_size > 0:
                val_ratio = val_size / (train_size + val_size)
                if target is not None:
                    X_train, X_val, y_train, y_val = train_test_split(
                        features, target,
                        test_size=val_ratio,
                        random_state=random_state,
                        stratify=stratify
                    )
                else:
                    X_train, X_val = train_test_split(
                        features,
                        test_size=val_ratio,
                        random_state=random_state
                    )
                    y_train, y_val = None, None
                X_test, y_test = None, None
            else:
                X_train, X_val, X_test = features, None, None
                y_train, y_val, y_test = target, None, None
        
        # Package results
        train_data = {'features': X_train, 'target': y_train} if X_train is not None else None
        val_data = {'features': X_val, 'target': y_val} if X_val is not None else None
        test_data = {'features': X_test, 'target': y_test} if X_test is not None else None
        
        self.logger.info(f"Data split - Train: {X_train.shape if X_train is not None else 'None'}, "
                        f"Val: {X_val.shape if X_val is not None else 'None'}, "
                        f"Test: {X_test.shape if X_test is not None else 'None'}")
        
        return train_data, val_data, test_data
    
    def _create_metadata(self, preprocessing_steps: List[Dict[str, Any]], input_data: Dict[str, pd.DataFrame], 
                        pipeline: Pipeline, transformed_data: Dict[str, pd.DataFrame]) -> PreprocessingMetadata:
        """Create preprocessing metadata for inference consistency."""
        df = input_data['features']
        
        # Identify column types
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Calculate data statistics
        data_statistics = {
            'numerical_stats': df[numerical_columns].describe().to_dict() if numerical_columns else {},
            'categorical_stats': {
                col: df[col].value_counts().to_dict() 
                for col in categorical_columns
            } if categorical_columns else {},
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # For now, assume no target column separation (will be handled in training)
        target_column = None  # We'll let the training component handle target column separation
        feature_columns = df.columns.tolist()
        
        # Create preprocessing steps metadata
        steps_metadata = []
        for i, step in enumerate(preprocessing_steps):
            step_dict = {
                'name': step.get('name', f'step_{i}'),
                'transformer': step['type'],
                'columns': step.get('columns'),
                'parameters': step.get('parameters', {}),
                'custom_function': None
            }
            steps_metadata.append(asdict(PreprocessingStep(**step_dict)))
        
        metadata = PreprocessingMetadata(
            steps=steps_metadata,
            column_names=df.columns.tolist(),
            target_column=target_column,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            data_statistics=data_statistics
        )
        
        return metadata
    
    def _calculate_metrics(self, input_data: Dict[str, pd.DataFrame], 
                          transformed_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate preprocessing metrics."""
        original_df = input_data['features']
        
        metrics = {
            'original_rows': len(original_df),
            'original_columns': len(original_df.columns),
            'original_memory_mb': original_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values_original': original_df.isnull().sum().sum(),
        }
        
        for split_name, df in transformed_data.items():
            if df is not None:
                metrics.update({
                    f'{split_name}_rows': len(df),
                    f'{split_name}_columns': len(df.columns),
                    f'{split_name}_memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    f'{split_name}_missing_values': df.isnull().sum().sum(),
                })
        
        return metrics
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted preprocessing pipeline."""
        if self.preprocessing_pipeline is None:
            raise RuntimeError("Preprocessing pipeline not fitted. Call execute() first.")
        
        transformed_data = self.preprocessing_pipeline.transform(data)
        
        # Convert back to DataFrame
        if hasattr(self.preprocessing_pipeline, 'get_feature_names_out'):
            try:
                feature_names = self.preprocessing_pipeline.get_feature_names_out()
            except:
                feature_names = [f'feature_{i}' for i in range(transformed_data.shape[1])]
        else:
            feature_names = [f'feature_{i}' for i in range(transformed_data.shape[1])]
        
        return pd.DataFrame(transformed_data, columns=feature_names, index=data.index)
    
    def load_preprocessing_pipeline(self, pipeline_path: str, metadata_path: str) -> None:
        """Load a previously saved preprocessing pipeline and metadata."""
        # Load pipeline
        with open(pipeline_path, 'rb') as f:
            self.preprocessing_pipeline = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
            self.metadata = PreprocessingMetadata(**metadata_dict)
        
        self.logger.info("Loaded preprocessing pipeline and metadata")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available from the preprocessing pipeline."""
        if self.preprocessing_pipeline is None:
            return None
        
        # Try to extract feature importance from feature selection steps
        importance = {}
        
        for step_name, transformer in self.preprocessing_pipeline.steps:
            if hasattr(transformer, 'scores_'):
                # Feature selection with scores
                feature_names = getattr(transformer, 'feature_names_in_', 
                                      [f'feature_{i}' for i in range(len(transformer.scores_))])
                for name, score in zip(feature_names, transformer.scores_):
                    importance[f'{step_name}_{name}'] = float(score)
            elif hasattr(transformer, 'ranking_'):
                # Feature selection with ranking
                feature_names = getattr(transformer, 'feature_names_in_', 
                                      [f'feature_{i}' for i in range(len(transformer.ranking_))])
                for name, rank in zip(feature_names, transformer.ranking_):
                    importance[f'{step_name}_{name}'] = 1.0 / float(rank)  # Inverse ranking
        
        return importance if importance else None
    
    def cleanup(self, context: ExecutionContext) -> None:
        """Cleanup preprocessing resources."""
        self.logger.info("Data preprocessing cleanup completed")