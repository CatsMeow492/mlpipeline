"""Tests for data preprocessing pipeline."""

import pytest
import tempfile
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from mlpipeline.data.preprocessing import (
    DataPreprocessor, PreprocessingStep, PreprocessingMetadata, CustomTransformer
)
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
from mlpipeline.core.errors import DataError, ConfigurationError


# Global function for testing custom transformations (can be pickled)
def global_log_transform(X):
    """Global log transform function for testing."""
    return np.log1p(X)


class TestPreprocessingStep:
    """Test preprocessing step configuration."""
    
    def test_basic_step(self):
        """Test basic preprocessing step creation."""
        step = PreprocessingStep(
            name="scaler",
            transformer="standard_scaler",
            columns=["feature1", "feature2"]
        )
        
        assert step.name == "scaler"
        assert step.transformer == "standard_scaler"
        assert step.columns == ["feature1", "feature2"]
        assert step.parameters == {}
    
    def test_step_with_parameters(self):
        """Test preprocessing step with parameters."""
        step = PreprocessingStep(
            name="imputer",
            transformer="simple_imputer",
            parameters={"strategy": "mean", "fill_value": 0}
        )
        
        assert step.transformer == "simple_imputer"
        assert step.parameters["strategy"] == "mean"
        assert step.parameters["fill_value"] == 0
    
    def test_custom_function_step(self):
        """Test preprocessing step with custom function."""
        step = PreprocessingStep(
            name="custom_transform",
            transformer="",
            custom_function="my_custom_func",
            parameters={"param1": "value1"}
        )
        
        assert step.custom_function == "my_custom_func"
        assert step.parameters["param1"] == "value1"


class TestPreprocessingMetadata:
    """Test preprocessing metadata."""
    
    def test_metadata_creation(self):
        """Test metadata creation with all fields."""
        metadata = PreprocessingMetadata(
            steps=[{"name": "scaler", "transformer": "standard_scaler"}],
            column_names=["feature1", "feature2", "target"],
            target_column="target",
            feature_columns=["feature1", "feature2"],
            categorical_columns=[],
            numerical_columns=["feature1", "feature2"],
            data_statistics={"mean": 0.5, "std": 1.0}
        )
        
        assert len(metadata.steps) == 1
        assert metadata.target_column == "target"
        assert len(metadata.feature_columns) == 2
        assert metadata.data_statistics["mean"] == 0.5
        assert metadata.created_at is not None


class TestCustomTransformer:
    """Test custom transformer wrapper."""
    
    def test_simple_custom_function(self):
        """Test custom transformer with simple function."""
        def double_values(X):
            return X * 2
        
        transformer = CustomTransformer(double_values, "double_values")
        
        # Test data
        X = np.array([[1, 2], [3, 4]])
        
        # Fit and transform
        transformer.fit(X)
        result = transformer.transform(X)
        
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(result, expected)
    
    def test_stateful_custom_function(self):
        """Test custom transformer with stateful function."""
        class StatefulFunction:
            def fit(self, X, y=None, **kwargs):
                return {"mean": np.mean(X, axis=0)}
            
            def transform(self, X, fitted_params=None, **kwargs):
                if fitted_params:
                    return X - fitted_params["mean"]
                return X
        
        func = StatefulFunction()
        transformer = CustomTransformer(func, "center_data")
        
        # Test data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Fit and transform
        transformer.fit(X)
        result = transformer.transform(X)
        
        # Should center the data around zero
        expected_mean = np.array([3, 4])  # Original mean
        expected = X - expected_mean
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_custom_function_with_parameters(self):
        """Test custom transformer with parameters."""
        def scale_by_factor(X, factor=1.0):
            return X * factor
        
        transformer = CustomTransformer(scale_by_factor, "scale_by_factor", factor=3.0)
        
        X = np.array([[1, 2], [3, 4]])
        
        transformer.fit(X)
        result = transformer.transform(X)
        
        expected = np.array([[3, 6], [9, 12]])
        np.testing.assert_array_equal(result, expected)


class TestDataPreprocessor:
    """Test data preprocessing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'numerical_feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numerical_feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 1]
        })
        
        # Save sample data
        self.data_path = Path(self.temp_dir) / "ingested_data.parquet"
        self.sample_data.to_parquet(self.data_path, index=False)
        
        # Create execution context
        self.context = ExecutionContext(
            experiment_id="test_exp",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={
                'data': {
                    'preprocessing': {
                        'steps': [
                            {
                                'name': 'scaler',
                                'transformer': 'standard_scaler',
                                'columns': ['numerical_feature1', 'numerical_feature2']
                            }
                        ],
                        'data_split': {
                            'train_size': 0.6,
                            'val_size': 0.2,
                            'test_size': 0.2,
                            'target_column': 'target',
                            'random_state': 42
                        }
                    }
                }
            },
            artifacts_path=str(self.temp_dir),
            logger=Mock(),
            metadata={}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        assert self.preprocessor.component_type == ComponentType.DATA_PREPROCESSING
        assert self.preprocessor.preprocessing_pipeline is None
        assert self.preprocessor.metadata is None
        assert len(self.preprocessor.transformer_registry) > 0
        assert 'standard_scaler' in self.preprocessor.transformer_registry
        assert 'label_encoder' in self.preprocessor.transformer_registry
    
    def test_register_custom_function(self):
        """Test registering custom transformation functions."""
        def my_custom_func(X):
            return X * 2
        
        self.preprocessor.register_custom_function("double", my_custom_func)
        
        assert "double" in self.preprocessor.custom_functions
        assert self.preprocessor.custom_functions["double"] == my_custom_func
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = {
            'preprocessing': {
                'steps': [
                    {
                        'name': 'scaler',
                        'transformer': 'standard_scaler',
                        'columns': ['feature1']
                    }
                ]
            }
        }
        
        assert self.preprocessor.validate_config(config) is True
    
    def test_validate_config_missing_steps(self):
        """Test configuration validation with missing steps."""
        config = {'preprocessing': {}}
        assert self.preprocessor.validate_config(config) is False
        
        config = {'preprocessing': {'steps': []}}
        assert self.preprocessor.validate_config(config) is False
    
    def test_validate_config_invalid_transformer(self):
        """Test configuration validation with invalid transformer."""
        config = {
            'preprocessing': {
                'steps': [
                    {
                        'name': 'invalid',
                        'transformer': 'nonexistent_transformer'
                    }
                ]
            }
        }
        
        assert self.preprocessor.validate_config(config) is False
    
    def test_validate_config_missing_transformer_and_function(self):
        """Test configuration validation with missing transformer and function."""
        config = {
            'preprocessing': {
                'steps': [
                    {
                        'name': 'invalid'
                        # Missing both transformer and custom_function
                    }
                ]
            }
        }
        
        assert self.preprocessor.validate_config(config) is False
    
    def test_load_input_data(self):
        """Test loading input data from artifacts."""
        input_data = self.preprocessor._load_input_data(self.context)
        
        assert 'features' in input_data
        df = input_data['features']
        assert len(df) == 5
        assert len(df.columns) == 4
        assert 'numerical_feature1' in df.columns
    
    def test_load_input_data_no_file(self):
        """Test loading input data when no file exists."""
        # Remove the data file
        self.data_path.unlink()
        
        with pytest.raises(DataError, match="No input data found"):
            self.preprocessor._load_input_data(self.context)
    
    def test_split_data_basic(self):
        """Test basic data splitting."""
        input_data = {'features': self.sample_data}
        split_config = {
            'train_size': 0.6,
            'val_size': 0.2,
            'test_size': 0.2,
            'random_state': 42
        }
        
        train_data, val_data, test_data = self.preprocessor._split_data(input_data, split_config)
        
        assert train_data is not None
        assert val_data is not None
        assert test_data is not None
        
        # Check sizes (approximately)
        total_rows = len(self.sample_data)
        assert len(train_data['features']) == int(total_rows * 0.6)
        assert len(val_data['features']) == 1  # Small dataset, so 1 row
        assert len(test_data['features']) == 1
    
    def test_split_data_with_target(self):
        """Test data splitting with target column."""
        input_data = {'features': self.sample_data}
        split_config = {
            'train_size': 0.6,
            'val_size': 0.2,
            'test_size': 0.2,
            'target_column': 'target',
            'random_state': 42
        }
        
        train_data, val_data, test_data = self.preprocessor._split_data(input_data, split_config)
        
        assert train_data['target'] is not None
        assert val_data['target'] is not None
        assert test_data['target'] is not None
        
        # Check that target is separated from features
        assert 'target' not in train_data['features'].columns
        assert 'target' not in val_data['features'].columns
        assert 'target' not in test_data['features'].columns
    
    def test_split_data_invalid_sizes(self):
        """Test data splitting with invalid split sizes."""
        input_data = {'features': self.sample_data}
        split_config = {
            'train_size': 0.5,
            'val_size': 0.3,
            'test_size': 0.3  # Sum > 1.0
        }
        
        with pytest.raises(ConfigurationError, match="Split sizes must sum to 1.0"):
            self.preprocessor._split_data(input_data, split_config)
    
    def test_build_pipeline_single_transformer(self):
        """Test building pipeline with single transformer."""
        config = {
            'steps': [
                {
                    'name': 'scaler',
                    'transformer': 'standard_scaler',
                    'columns': ['numerical_feature1', 'numerical_feature2']
                }
            ]
        }
        
        input_data = {'features': self.sample_data}
        pipeline = self.preprocessor._build_pipeline(config, input_data)
        
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == 'preprocessing'  # Updated to match new structure
        
        # Check that the ColumnTransformer contains our transformer
        column_transformer = pipeline.steps[0][1]
        assert len(column_transformer.transformers) == 1
        assert column_transformer.transformers[0][0] == 'scaler'
    
    def test_build_pipeline_multiple_transformers(self):
        """Test building pipeline with multiple transformers."""
        config = {
            'steps': [
                {
                    'name': 'imputer',
                    'transformer': 'simple_imputer',
                    'parameters': {'strategy': 'mean'}
                },
                {
                    'name': 'scaler',
                    'transformer': 'standard_scaler'
                }
            ]
        }
        
        input_data = {'features': self.sample_data}
        pipeline = self.preprocessor._build_pipeline(config, input_data)
        
        assert len(pipeline.steps) == 1  # Single ColumnTransformer step
        assert pipeline.steps[0][0] == 'preprocessing'
        
        # Check that the ColumnTransformer contains both transformers
        column_transformer = pipeline.steps[0][1]
        assert len(column_transformer.transformers) == 2
        transformer_names = [t[0] for t in column_transformer.transformers]
        assert 'imputer' in transformer_names
        assert 'scaler' in transformer_names
    
    def test_build_pipeline_custom_function(self):
        """Test building pipeline with custom function."""
        # Register custom function
        def double_values(X):
            return X * 2
        
        self.preprocessor.register_custom_function("double", double_values)
        
        config = {
            'steps': [
                {
                    'name': 'doubler',
                    'custom_function': 'double',
                    'columns': ['numerical_feature1']
                }
            ]
        }
        
        input_data = {'features': self.sample_data}
        pipeline = self.preprocessor._build_pipeline(config, input_data)
        
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == 'preprocessing'  # Updated to match new structure
        
        # Check that the ColumnTransformer contains our custom transformer
        column_transformer = pipeline.steps[0][1]
        assert len(column_transformer.transformers) == 1
        assert column_transformer.transformers[0][0] == 'doubler'
    
    def test_execute_basic_preprocessing(self):
        """Test basic preprocessing execution."""
        result = self.preprocessor.execute(self.context)
        
        assert result.success is True
        assert len(result.artifacts) >= 3  # train, val, test data + metadata + pipeline
        assert 'original_shape' in result.metadata
        assert 'transformed_shape' in result.metadata
        
        # Check that files were created
        train_path = Path(self.temp_dir) / "train_preprocessed.parquet"
        assert train_path.exists()
        
        metadata_path = Path(self.temp_dir) / "preprocessing_metadata.json"
        assert metadata_path.exists()
        
        pipeline_path = Path(self.temp_dir) / "preprocessing_pipeline.pkl"
        assert pipeline_path.exists()
    
    def test_execute_with_custom_function(self):
        """Test preprocessing execution with custom function."""
        # Register the global custom function
        self.preprocessor.register_custom_function("log_transform", global_log_transform)
        
        # Update config to use custom function
        self.context.config['data']['preprocessing']['steps'] = [
            {
                'name': 'log_transform',
                'custom_function': 'log_transform',
                'columns': ['numerical_feature1', 'numerical_feature2']
            }
        ]
        
        result = self.preprocessor.execute(self.context)
        
        assert result.success is True
        assert len(result.artifacts) >= 3
    
    def test_execute_no_input_data(self):
        """Test preprocessing execution with no input data."""
        # Remove input data file
        self.data_path.unlink()
        
        result = self.preprocessor.execute(self.context)
        
        assert result.success is False
        assert "No input data found" in result.error_message
    
    def test_execute_invalid_config(self):
        """Test preprocessing execution with invalid configuration."""
        # Set invalid config
        self.context.config['data']['preprocessing']['steps'] = []
        
        with pytest.raises(ConfigurationError):
            self.preprocessor.setup(self.context)
    
    def test_transform_new_data(self):
        """Test transforming new data with fitted pipeline."""
        # First execute preprocessing to fit pipeline
        result = self.preprocessor.execute(self.context)
        assert result.success is True
        
        # Create new data to transform
        new_data = pd.DataFrame({
            'numerical_feature1': [6.0, 7.0],
            'numerical_feature2': [60.0, 70.0],
            'categorical_feature': ['A', 'B']
        })
        
        # Transform new data
        transformed = self.preprocessor.transform_new_data(new_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == 2
        # Should have transformed features
        assert transformed.shape[1] > 0
    
    def test_transform_new_data_not_fitted(self):
        """Test transforming new data without fitted pipeline."""
        new_data = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(RuntimeError, match="Preprocessing pipeline not fitted"):
            self.preprocessor.transform_new_data(new_data)
    
    def test_load_preprocessing_pipeline(self):
        """Test loading saved preprocessing pipeline."""
        # First execute preprocessing to create files
        result = self.preprocessor.execute(self.context)
        assert result.success is True
        
        # Create new preprocessor instance
        new_preprocessor = DataPreprocessor()
        
        # Load the saved pipeline and metadata
        pipeline_path = Path(self.temp_dir) / "preprocessing_pipeline.pkl"
        metadata_path = Path(self.temp_dir) / "preprocessing_metadata.json"
        
        new_preprocessor.load_preprocessing_pipeline(str(pipeline_path), str(metadata_path))
        
        assert new_preprocessor.preprocessing_pipeline is not None
        assert new_preprocessor.metadata is not None
        assert new_preprocessor.metadata.target_column == 'target'
    
    def test_create_metadata(self):
        """Test creating preprocessing metadata."""
        config = {
            'steps': [
                {
                    'name': 'scaler',
                    'transformer': 'standard_scaler'
                }
            ],
            'data_split': {
                'target_column': 'target'
            }
        }
        
        input_data = {'features': self.sample_data}
        
        # Create a dummy pipeline for testing
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = Pipeline([('scaler', StandardScaler())])
        
        transformed_data = {'train': self.sample_data}
        
        metadata = self.preprocessor._create_metadata(config, input_data, pipeline, transformed_data)
        
        assert isinstance(metadata, PreprocessingMetadata)
        assert len(metadata.steps) == 1
        assert metadata.target_column == 'target'
        assert len(metadata.numerical_columns) == 2
        assert len(metadata.categorical_columns) == 1
        assert 'numerical_stats' in metadata.data_statistics
        assert 'categorical_stats' in metadata.data_statistics
    
    def test_calculate_metrics(self):
        """Test calculating preprocessing metrics."""
        input_data = {'features': self.sample_data}
        transformed_data = {
            'train': self.sample_data.iloc[:3],
            'val': self.sample_data.iloc[3:4],
            'test': self.sample_data.iloc[4:5]
        }
        
        metrics = self.preprocessor._calculate_metrics(input_data, transformed_data)
        
        assert 'original_rows' in metrics
        assert 'original_columns' in metrics
        assert 'train_rows' in metrics
        assert 'val_rows' in metrics
        assert 'test_rows' in metrics
        assert metrics['original_rows'] == 5
        assert metrics['train_rows'] == 3
        assert metrics['val_rows'] == 1
        assert metrics['test_rows'] == 1
    
    def test_get_feature_importance_no_pipeline(self):
        """Test getting feature importance without fitted pipeline."""
        importance = self.preprocessor.get_feature_importance()
        assert importance is None
    
    def test_get_feature_importance_with_feature_selection(self):
        """Test getting feature importance with feature selection."""
        # Create config with feature selection - only apply to numerical columns
        self.context.config['data']['preprocessing']['steps'] = [
            {
                'name': 'selector',
                'transformer': 'select_k_best',
                'parameters': {'k': 2},
                'columns': ['numerical_feature1', 'numerical_feature2']
            }
        ]
        
        # Execute preprocessing
        result = self.preprocessor.execute(self.context)
        assert result.success is True
        
        # Get feature importance
        importance = self.preprocessor.get_feature_importance()
        
        # Should have some importance scores (may be None if no feature selection with scores)
        # This depends on the specific transformer used
        assert importance is None or isinstance(importance, dict)
    
    def test_preprocessing_with_missing_values(self):
        """Test preprocessing with missing values in data."""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'numerical_feature1'] = np.nan
        data_with_missing.loc[1, 'categorical_feature'] = np.nan
        
        # Save data with missing values
        missing_data_path = Path(self.temp_dir) / "data_with_missing.parquet"
        data_with_missing.to_parquet(missing_data_path, index=False)
        
        # Update data path
        self.data_path.unlink()
        missing_data_path.rename(self.data_path)
        
        # Add imputation step to config
        self.context.config['data']['preprocessing']['steps'] = [
            {
                'name': 'imputer',
                'transformer': 'simple_imputer',
                'parameters': {'strategy': 'mean'},
                'columns': ['numerical_feature1', 'numerical_feature2']
            },
            {
                'name': 'scaler',
                'transformer': 'standard_scaler',
                'columns': ['numerical_feature1', 'numerical_feature2']
            }
        ]
        
        result = self.preprocessor.execute(self.context)
        
        assert result.success is True
        
        # Check that missing values were handled
        train_path = Path(self.temp_dir) / "train_preprocessed.parquet"
        train_data = pd.read_parquet(train_path)
        
        # Should have no missing values in numerical features after imputation
        numerical_cols = [col for col in train_data.columns if 'numerical_feature' in col]
        assert train_data[numerical_cols].isnull().sum().sum() == 0
    
    def test_preprocessing_categorical_encoding(self):
        """Test preprocessing with categorical encoding."""
        # Update config to include categorical encoding
        self.context.config['data']['preprocessing']['steps'] = [
            {
                'name': 'encoder',
                'transformer': 'one_hot_encoder',
                'columns': ['categorical_feature'],
                'parameters': {'sparse_output': False, 'handle_unknown': 'ignore'}
            }
        ]
        
        result = self.preprocessor.execute(self.context)
        
        assert result.success is True
        
        # Check that categorical features were encoded
        train_path = Path(self.temp_dir) / "train_preprocessed.parquet"
        train_data = pd.read_parquet(train_path)
        
        # Should have more columns due to one-hot encoding
        assert train_data.shape[1] > self.sample_data.shape[1] - 1  # -1 for target column
    
    def test_preprocessing_with_all_transformers(self):
        """Test preprocessing pipeline with multiple transformer types."""
        # Create more complex data
        complex_data = pd.DataFrame({
            'numerical1': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            'numerical2': [10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0, 80.0],
            'categorical1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
            'categorical2': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y'],
            'target': [0, 1, 0, 1, 1, 0, 1, 0]
        })
        
        # Save complex data
        complex_data_path = Path(self.temp_dir) / "complex_data.parquet"
        complex_data.to_parquet(complex_data_path, index=False)
        
        # Update data path
        self.data_path.unlink()
        complex_data_path.rename(self.data_path)
        
        # Configure comprehensive preprocessing pipeline
        self.context.config['data']['preprocessing'] = {
            'steps': [
                {
                    'name': 'imputer',
                    'transformer': 'simple_imputer',
                    'parameters': {'strategy': 'mean'},
                    'columns': ['numerical1', 'numerical2']
                },
                {
                    'name': 'scaler',
                    'transformer': 'standard_scaler',
                    'columns': ['numerical1', 'numerical2']
                },
                {
                    'name': 'encoder',
                    'transformer': 'one_hot_encoder',
                    'columns': ['categorical1', 'categorical2'],
                    'parameters': {'sparse_output': False, 'handle_unknown': 'ignore'}
                }
            ],
            'data_split': {
                'train_size': 0.6,
                'val_size': 0.2,
                'test_size': 0.2,
                'target_column': 'target',
                'random_state': 42
            }
        }
        
        result = self.preprocessor.execute(self.context)
        
        assert result.success is True
        assert len(result.artifacts) >= 4  # train, val, test, metadata, pipeline
        
        # Verify all datasets were created
        for split in ['train', 'val', 'test']:
            split_path = Path(self.temp_dir) / f"{split}_preprocessed.parquet"
            assert split_path.exists()
            
            split_data = pd.read_parquet(split_path)
            assert len(split_data) > 0
            assert 'target' in split_data.columns
    
    def test_preprocessing_edge_cases(self):
        """Test preprocessing with edge cases."""
        # Create edge case data
        edge_data = pd.DataFrame({
            'constant_feature': [1.0, 1.0, 1.0, 1.0, 1.0],  # No variance
            'single_category': ['A', 'A', 'A', 'A', 'A'],    # Single category
            'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan],  # All missing
            'normal_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [0, 1, 0, 1, 1]
        })
        
        # Save edge case data
        edge_data_path = Path(self.temp_dir) / "edge_data.parquet"
        edge_data.to_parquet(edge_data_path, index=False)
        
        # Update data path
        self.data_path.unlink()
        edge_data_path.rename(self.data_path)
        
        # Configure preprocessing to handle edge cases
        self.context.config['data']['preprocessing'] = {
            'steps': [
                {
                    'name': 'imputer',
                    'transformer': 'simple_imputer',
                    'parameters': {'strategy': 'constant', 'fill_value': 0},
                    'columns': ['all_missing']
                },
                {
                    'name': 'variance_filter',
                    'transformer': 'variance_threshold',
                    'parameters': {'threshold': 0.0},  # Remove zero variance features
                    'columns': ['constant_feature', 'normal_feature', 'all_missing']  # Only numerical columns
                }
            ],
            'data_split': {
                'train_size': 0.6,
                'val_size': 0.2,
                'test_size': 0.2,
                'target_column': 'target',
                'random_state': 42
            }
        }
        
        result = self.preprocessor.execute(self.context)
        
        # Should handle edge cases gracefully
        assert result.success is True
    
    def test_preprocessing_no_data_split(self):
        """Test preprocessing without data splitting."""
        # Configure preprocessing without data split
        self.context.config['data']['preprocessing'] = {
            'steps': [
                {
                    'name': 'scaler',
                    'transformer': 'standard_scaler',
                    'columns': ['numerical_feature1', 'numerical_feature2']
                }
            ],
            'data_split': {
                'train_size': 1.0,
                'val_size': 0.0,
                'test_size': 0.0
            }
        }
        
        result = self.preprocessor.execute(self.context)
        
        assert result.success is True
        
        # Should only create train data
        train_path = Path(self.temp_dir) / "train_preprocessed.parquet"
        assert train_path.exists()
        
        val_path = Path(self.temp_dir) / "val_preprocessed.parquet"
        test_path = Path(self.temp_dir) / "test_preprocessed.parquet"
        assert not val_path.exists()
        assert not test_path.exists()
    
    def test_preprocessing_pipeline_serialization(self):
        """Test that preprocessing pipeline can be properly serialized and deserialized."""
        # Execute preprocessing
        result = self.preprocessor.execute(self.context)
        assert result.success is True
        
        # Load pipeline from file
        pipeline_path = Path(self.temp_dir) / "preprocessing_pipeline.pkl"
        with open(pipeline_path, 'rb') as f:
            loaded_pipeline = pickle.load(f)
        
        # Test that loaded pipeline works
        test_data = self.sample_data[['numerical_feature1', 'numerical_feature2', 'categorical_feature']]
        transformed = loaded_pipeline.transform(test_data)
        
        assert transformed is not None
        assert transformed.shape[0] == len(test_data)
    
    def test_cleanup(self):
        """Test preprocessing cleanup."""
        # Execute preprocessing first
        result = self.preprocessor.execute(self.context)
        assert result.success is True
        
        # Cleanup
        self.preprocessor.cleanup(self.context)
        
        # Should not raise any errors
        assert True