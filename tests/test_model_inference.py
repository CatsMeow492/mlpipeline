"""Tests for model inference system."""

import pytest
import pandas as pd
import numpy as np
import json
import pickle
import joblib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from mlpipeline.models.inference import (
    ModelMetadata,
    InferenceResult,
    ModelLoader,
    ModelValidator,
    ModelCache,
    ModelInferenceEngine
)
from mlpipeline.data.preprocessing import PreprocessingMetadata
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
from mlpipeline.core.errors import ModelError, DataError, ConfigurationError


class TestModelMetadata:
    """Test ModelMetadata dataclass."""
    
    def test_model_metadata_creation(self):
        """Test creating ModelMetadata instance."""
        metadata = ModelMetadata(
            model_id="test_model_123",
            model_type="classifier",
            framework="sklearn",
            version="1.0.0",
            created_at="2024-01-01T00:00:00",
            feature_columns=["feature1", "feature2"],
            target_column="target",
            preprocessing_metadata_hash="abc123",
            model_parameters={"n_estimators": 100},
            training_metrics={"accuracy": 0.95},
            data_schema={"feature1": "float64", "feature2": "int64"},
            model_size_bytes=1024,
            python_version="3.8.0",
            dependencies={"sklearn": "1.0.0"}
        )
        
        assert metadata.model_id == "test_model_123"
        assert metadata.framework == "sklearn"
        assert len(metadata.feature_columns) == 2
        assert metadata.training_metrics["accuracy"] == 0.95


class TestInferenceResult:
    """Test InferenceResult dataclass."""
    
    def test_inference_result_creation(self):
        """Test creating InferenceResult instance."""
        predictions = np.array([1, 0, 1])
        probabilities = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        
        result = InferenceResult(
            predictions=predictions,
            prediction_probabilities=probabilities,
            confidence_scores=np.array([0.9, 0.8, 0.7]),
            preprocessing_time=0.1,
            inference_time=0.05
        )
        
        assert len(result.predictions) == 3
        assert result.prediction_probabilities.shape == (3, 2)
        assert result.preprocessing_time == 0.1
        assert result.metadata == {}  # Default empty dict


class TestModelLoader:
    """Test ModelLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ModelLoader()
        self.temp_dir = Path("test_temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_load_pickle_model(self):
        """Test loading pickle model."""
        # Create a simple sklearn model
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        model = LogisticRegression()
        # Fit with dummy data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model.fit(X, y)
        
        # Save as pickle
        model_path = self.temp_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Load model
        loaded_model, metadata = self.loader.load_model(str(model_path))
        
        assert loaded_model is not None
        assert metadata is None
        assert hasattr(loaded_model, 'predict')
    
    def test_load_joblib_model(self):
        """Test loading joblib model."""
        # Create a simple sklearn model
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        model = LogisticRegression()
        # Fit with dummy data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model.fit(X, y)
        
        # Save as joblib
        model_path = self.temp_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Load model
        loaded_model, metadata = self.loader.load_model(str(model_path))
        
        assert loaded_model is not None
        assert metadata is None
    
    def test_load_model_with_metadata(self):
        """Test loading model with metadata."""
        # Create a simple sklearn model
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        model = LogisticRegression()
        # Fit with dummy data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model.fit(X, y)
        
        model_path = self.temp_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Create metadata
        metadata_dict = {
            "model_id": "test_123",
            "model_type": "classifier",
            "framework": "sklearn",
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00",
            "feature_columns": ["feature1", "feature2"],
            "target_column": "target",
            "preprocessing_metadata_hash": "abc123",
            "model_parameters": {},
            "training_metrics": {},
            "data_schema": {},
            "model_size_bytes": 1024,
            "python_version": "3.8.0",
            "dependencies": {}
        }
        
        metadata_path = self.temp_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f)
        
        # Load model with metadata
        loaded_model, metadata = self.loader.load_model(str(model_path), str(metadata_path))
        
        assert loaded_model is not None
        assert metadata is not None
        assert metadata.model_id == "test_123"
        assert metadata.framework == "sklearn"
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model raises error."""
        with pytest.raises(ModelError, match="Model file not found"):
            self.loader.load_model("nonexistent.pkl")
    
    def test_detect_model_format(self):
        """Test model format detection."""
        # Test with metadata
        metadata = Mock()
        metadata.framework = "sklearn"
        
        format_detected = self.loader._detect_model_format(Path("test.pkl"), metadata)
        assert format_detected == "sklearn"
        
        # Test with file extension
        format_detected = self.loader._detect_model_format(Path("test.joblib"), None)
        assert format_detected == "joblib"
        
        format_detected = self.loader._detect_model_format(Path("test.pkl"), None)
        assert format_detected == "pickle"


class TestModelValidator:
    """Test ModelValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ModelValidator()
    
    def test_validate_model_compatibility_success(self):
        """Test successful model validation."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock()
        mock_model.n_features_in_ = 2
        
        # Create model metadata
        model_metadata = ModelMetadata(
            model_id="test_123",
            model_type="classifier",
            framework="sklearn",
            version="1.0.0",
            created_at=datetime.now().isoformat(),
            feature_columns=["feature1", "feature2"],
            target_column="target",
            preprocessing_metadata_hash="abc123",
            model_parameters={},
            training_metrics={},
            data_schema={"feature1": "float64", "feature2": "int64"},
            model_size_bytes=1024,
            python_version="3.8.0",
            dependencies={}
        )
        
        # Create preprocessing metadata
        preprocessing_metadata = PreprocessingMetadata(
            steps=[],
            column_names=["feature1", "feature2", "target"],
            target_column="target",
            feature_columns=["feature1", "feature2"],
            categorical_columns=[],
            numerical_columns=["feature1", "feature2"]
        )
        
        # Create input data
        input_data = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [10, 20, 30]
        })
        
        # Mock hash calculation to match
        with patch.object(self.validator, '_calculate_preprocessing_hash', return_value="abc123"):
            results = self.validator.validate_model_compatibility(
                mock_model, model_metadata, preprocessing_metadata, input_data
            )
        
        assert results['compatible'] is True
        assert len(results['errors']) == 0
        assert 'model_metadata' in results['checks_performed']
    
    def test_validate_model_compatibility_feature_mismatch(self):
        """Test validation with feature mismatch."""
        mock_model = Mock()
        mock_model.predict = Mock()
        mock_model.n_features_in_ = 3  # Model expects 3 features
        
        model_metadata = ModelMetadata(
            model_id="test_123",
            model_type="classifier",
            framework="sklearn",
            version="1.0.0",
            created_at=datetime.now().isoformat(),
            feature_columns=["feature1", "feature2", "feature3"],
            target_column="target",
            preprocessing_metadata_hash="abc123",
            model_parameters={},
            training_metrics={},
            data_schema={},
            model_size_bytes=1024,
            python_version="3.8.0",
            dependencies={}
        )
        
        # Input data has only 2 features
        input_data = pd.DataFrame({
            "feature1": [1.0, 2.0],
            "feature2": [10, 20]
        })
        
        results = self.validator.validate_model_compatibility(
            mock_model, model_metadata, None, input_data
        )
        
        assert results['compatible'] is False
        assert any("Feature count mismatch" in error for error in results['errors'])
        assert any("Missing required columns" in error for error in results['errors'])
    
    def test_validate_preprocessing_compatibility_hash_mismatch(self):
        """Test preprocessing compatibility with hash mismatch."""
        mock_model = Mock()
        
        model_metadata = ModelMetadata(
            model_id="test_123",
            model_type="classifier",
            framework="sklearn",
            version="1.0.0",
            created_at=datetime.now().isoformat(),
            feature_columns=["feature1", "feature2"],
            target_column="target",
            preprocessing_metadata_hash="old_hash",
            model_parameters={},
            training_metrics={},
            data_schema={},
            model_size_bytes=1024,
            python_version="3.8.0",
            dependencies={}
        )
        
        preprocessing_metadata = PreprocessingMetadata(
            steps=[],
            column_names=["feature1", "feature2"],
            target_column="target",
            feature_columns=["feature1", "feature2"],
            categorical_columns=[],
            numerical_columns=["feature1", "feature2"]
        )
        
        input_data = pd.DataFrame({"feature1": [1], "feature2": [2]})
        
        # Mock hash to return different value
        with patch.object(self.validator, '_calculate_preprocessing_hash', return_value="new_hash"):
            results = self.validator.validate_model_compatibility(
                mock_model, model_metadata, preprocessing_metadata, input_data
            )
        
        assert results['compatible'] is False
        assert any("Preprocessing pipeline has changed" in error for error in results['errors'])
    
    def test_calculate_preprocessing_hash(self):
        """Test preprocessing hash calculation."""
        metadata = PreprocessingMetadata(
            steps=[{"name": "scaler", "transformer": "standard_scaler"}],
            column_names=["feature1", "feature2"],
            target_column="target",
            feature_columns=["feature1", "feature2"],
            categorical_columns=[],
            numerical_columns=["feature1", "feature2"]
        )
        
        hash1 = self.validator._calculate_preprocessing_hash(metadata)
        hash2 = self.validator._calculate_preprocessing_hash(metadata)
        
        # Same metadata should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        
        # Different metadata should produce different hash
        metadata.feature_columns = ["feature1", "feature3"]
        hash3 = self.validator._calculate_preprocessing_hash(metadata)
        assert hash1 != hash3


class TestModelCache:
    """Test ModelCache class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = ModelCache(max_cache_size=2)
    
    def test_cache_and_retrieve_model(self):
        """Test caching and retrieving models."""
        mock_model = Mock()
        metadata = Mock()
        
        # Cache model
        self.cache.cache_model("model1.pkl", mock_model, metadata)
        
        # Retrieve model
        result = self.cache.get_model("model1.pkl")
        
        assert result is not None
        assert result[0] is mock_model
        assert result[1] is metadata
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        # Add models up to limit
        for i in range(3):  # Cache size is 2, so this should evict oldest
            mock_model = Mock()
            self.cache.cache_model(f"model{i}.pkl", mock_model, None)
        
        # First model should be evicted
        result = self.cache.get_model("model0.pkl")
        assert result is None
        
        # Last two models should still be cached
        result = self.cache.get_model("model1.pkl")
        assert result is not None
        
        result = self.cache.get_model("model2.pkl")
        assert result is not None
    
    def test_clear_cache(self):
        """Test clearing cache."""
        mock_model = Mock()
        self.cache.cache_model("model.pkl", mock_model, None)
        
        # Verify model is cached
        assert self.cache.get_model("model.pkl") is not None
        
        # Clear cache
        self.cache.clear_cache()
        
        # Verify model is no longer cached
        assert self.cache.get_model("model.pkl") is None


class TestModelInferenceEngine:
    """Test ModelInferenceEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ModelInferenceEngine(enable_caching=False)
        self.temp_dir = Path("test_temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        # Create a simple sklearn model
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        model = LogisticRegression()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model.fit(X, y)
        
        model_path = self.temp_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        config = {
            "inference": {
                "model_path": str(model_path)
            }
        }
        
        assert self.engine.validate_config(config) is True
    
    def test_validate_config_missing_model_path(self):
        """Test configuration validation with missing model path."""
        config = {"inference": {}}
        
        assert self.engine.validate_config(config) is False
    
    def test_validate_config_nonexistent_model(self):
        """Test configuration validation with non-existent model."""
        config = {
            "inference": {
                "model_path": "nonexistent.pkl"
            }
        }
        
        assert self.engine.validate_config(config) is False
    
    @patch('mlpipeline.models.inference.ModelInferenceEngine._load_input_data')
    @patch('mlpipeline.models.inference.ModelInferenceEngine._load_model_with_cache')
    @patch('mlpipeline.models.inference.ModelInferenceEngine._validate_model')
    @patch('mlpipeline.models.inference.ModelInferenceEngine._perform_inference')
    @patch('mlpipeline.models.inference.ModelInferenceEngine._save_inference_results')
    def test_execute_success(self, mock_save, mock_inference, mock_validate, 
                           mock_load_model, mock_load_data):
        """Test successful execution."""
        # Setup mocks
        mock_data = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        mock_load_data.return_value = mock_data
        
        mock_model = Mock()
        mock_metadata = Mock()
        mock_load_model.return_value = (mock_model, mock_metadata)
        
        mock_validate.return_value = {"compatible": True, "errors": [], "warnings": []}
        
        mock_result = InferenceResult(
            predictions=np.array([1, 0]),
            inference_time=0.1
        )
        mock_inference.return_value = mock_result
        
        mock_save.return_value = ["predictions.parquet"]
        
        # Create execution context
        context = ExecutionContext(
            experiment_id="test_exp",
            stage_name="inference",
            component_type=ComponentType.MODEL_INFERENCE,
            config={
                "inference": {
                    "model_path": "model.pkl"
                }
            },
            artifacts_path=str(self.temp_dir),
            logger=Mock()
        )
        
        # Execute
        result = self.engine.execute(context)
        
        assert result.success is True
        assert len(result.artifacts) > 0
        assert "num_predictions" in result.metrics
    
    def test_predict_with_model_sklearn(self):
        """Test prediction with sklearn-like model."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, 1])
        
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        
        predictions = self.engine._predict_with_model(mock_model, data)
        
        assert len(predictions) == 3
        mock_model.predict.assert_called_once()
    
    def test_predict_with_model_no_predict_method(self):
        """Test prediction with model that has no predict method."""
        mock_model = Mock()
        del mock_model.predict  # Remove predict method
        
        data = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        
        with pytest.raises(ModelError, match="does not support prediction"):
            self.engine._predict_with_model(mock_model, data)
    
    def test_get_feature_importance_with_feature_importances(self):
        """Test feature importance extraction from model with feature_importances_."""
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.6, 0.4])
        
        data = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        
        importance = self.engine._get_feature_importance(mock_model, data)
        
        assert importance is not None
        assert "feature1" in importance
        assert "feature2" in importance
        assert importance["feature1"] == 0.6
        assert importance["feature2"] == 0.4
    
    def test_get_feature_importance_with_coef(self):
        """Test feature importance extraction from linear model with coef_."""
        mock_model = Mock()
        mock_model.coef_ = np.array([0.8, -0.3])
        del mock_model.feature_importances_  # Remove feature_importances_
        
        data = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        
        importance = self.engine._get_feature_importance(mock_model, data)
        
        assert importance is not None
        assert importance["feature1"] == 0.8  # abs(0.8)
        assert importance["feature2"] == 0.3  # abs(-0.3)
    
    def test_get_feature_importance_none(self):
        """Test feature importance when model doesn't support it."""
        mock_model = Mock()
        del mock_model.feature_importances_
        del mock_model.coef_
        
        data = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        
        importance = self.engine._get_feature_importance(mock_model, data)
        
        assert importance is None
    
    def test_calculate_inference_metrics(self):
        """Test inference metrics calculation."""
        result = InferenceResult(
            predictions=np.array([1, 0, 1]),
            confidence_scores=np.array([0.9, 0.8, 0.7]),
            preprocessing_time=0.1,
            inference_time=0.05
        )
        
        input_data = pd.DataFrame({"feature1": [1, 2, 3]})
        
        metrics = self.engine._calculate_inference_metrics(result, input_data)
        
        assert metrics["num_predictions"] == 3
        assert metrics["inference_time_seconds"] == 0.05
        assert metrics["preprocessing_time_seconds"] == 0.1
        assert abs(metrics["total_time_seconds"] - 0.15) < 0.001
        assert abs(metrics["mean_confidence"] - 0.8) < 0.001
        assert metrics["predictions_per_second"] == 60.0  # 3 / 0.05
    
    def test_predict_single_dict_input(self):
        """Test single prediction with dictionary input."""
        # Setup mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        self.engine.current_model = mock_model
        
        # Test with dictionary input
        input_dict = {"feature1": 1.0, "feature2": 2.0}
        
        with patch.object(self.engine, '_perform_inference') as mock_inference:
            mock_inference.return_value = InferenceResult(predictions=np.array([1]))
            
            result = self.engine.predict_single(input_dict)
            
            # Verify DataFrame was created from dict
            call_args = mock_inference.call_args[0]
            input_df = call_args[1]
            assert isinstance(input_df, pd.DataFrame)
            assert len(input_df) == 1
            assert "feature1" in input_df.columns
            assert "feature2" in input_df.columns
    
    def test_predict_single_no_model_loaded(self):
        """Test single prediction when no model is loaded."""
        input_data = pd.DataFrame({"feature1": [1.0]})
        
        with pytest.raises(RuntimeError, match="No model loaded"):
            self.engine.predict_single(input_data)


if __name__ == "__main__":
    pytest.main([__file__])