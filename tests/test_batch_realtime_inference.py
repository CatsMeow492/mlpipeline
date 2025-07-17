"""Tests for batch and real-time inference engines."""

import pytest
import pandas as pd
import numpy as np
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from mlpipeline.models.inference import (
    BatchInferenceConfig,
    BatchInferenceResult,
    BatchInferenceEngine,
    RealTimeInferenceConfig,
    RealTimeInferenceEngine,
    ModelInferenceEngine,
    InferenceResult
)


class TestBatchInferenceConfig:
    """Test BatchInferenceConfig dataclass."""
    
    def test_batch_config_defaults(self):
        """Test default configuration values."""
        config = BatchInferenceConfig()
        
        assert config.chunk_size == 1000
        assert config.max_workers == 4
        assert config.progress_callback is None
        assert config.save_intermediate is False
        assert config.output_format == 'parquet'
    
    def test_batch_config_custom_values(self):
        """Test custom configuration values."""
        def dummy_callback(current, total):
            pass
        
        config = BatchInferenceConfig(
            chunk_size=500,
            max_workers=2,
            progress_callback=dummy_callback,
            save_intermediate=True,
            output_format='csv'
        )
        
        assert config.chunk_size == 500
        assert config.max_workers == 2
        assert config.progress_callback is dummy_callback
        assert config.save_intermediate is True
        assert config.output_format == 'csv'


class TestBatchInferenceResult:
    """Test BatchInferenceResult dataclass."""
    
    def test_batch_result_creation(self):
        """Test creating BatchInferenceResult instance."""
        result = BatchInferenceResult(
            total_predictions=1000,
            processing_time=45.5,
            chunks_processed=10,
            failed_chunks=1,
            output_files=['file1.parquet', 'file2.parquet'],
            summary_stats={'mean': 0.5}
        )
        
        assert result.total_predictions == 1000
        assert result.processing_time == 45.5
        assert result.chunks_processed == 10
        assert result.failed_chunks == 1
        assert len(result.output_files) == 2
        assert result.error_details == []  # Default empty list


class TestBatchInferenceEngine:
    """Test BatchInferenceEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock model engine
        self.mock_model_engine = Mock(spec=ModelInferenceEngine)
        self.mock_model_engine.current_model = Mock()
        
        # Create batch engine
        self.batch_engine = BatchInferenceEngine(self.mock_model_engine)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        self.temp_dir = Path("test_batch_output")
        self.temp_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_predict_batch_sequential(self):
        """Test sequential batch processing."""
        # Mock inference results
        mock_inference_result = InferenceResult(
            predictions=np.array([1, 0, 1]),
            confidence_scores=np.array([0.9, 0.8, 0.7]),
            inference_time=0.1
        )
        self.mock_model_engine._perform_inference.return_value = mock_inference_result
        
        # Configure for sequential processing
        config = BatchInferenceConfig(
            chunk_size=30,
            max_workers=1,  # Sequential
            save_intermediate=False
        )
        
        # Run batch inference
        result = self.batch_engine.predict_batch(
            self.test_data, config, str(self.temp_dir)
        )
        
        assert isinstance(result, BatchInferenceResult)
        assert result.chunks_processed > 0
        assert result.failed_chunks == 0
        assert result.total_predictions > 0
        assert result.processing_time > 0
        assert len(result.output_files) > 0
    
    def test_predict_batch_parallel(self):
        """Test parallel batch processing."""
        # Mock inference results
        mock_inference_result = InferenceResult(
            predictions=np.array([1, 0]),
            confidence_scores=np.array([0.9, 0.8]),
            inference_time=0.1
        )
        self.mock_model_engine._perform_inference.return_value = mock_inference_result
        
        # Configure for parallel processing
        config = BatchInferenceConfig(
            chunk_size=25,
            max_workers=2,  # Parallel
            save_intermediate=False
        )
        
        # Run batch inference
        result = self.batch_engine.predict_batch(
            self.test_data, config, str(self.temp_dir)
        )
        
        assert isinstance(result, BatchInferenceResult)
        assert result.chunks_processed > 0
        assert result.failed_chunks == 0
    
    def test_predict_batch_with_progress_callback(self):
        """Test batch processing with progress callback."""
        # Mock inference results
        mock_inference_result = InferenceResult(
            predictions=np.array([1]),
            inference_time=0.1
        )
        self.mock_model_engine._perform_inference.return_value = mock_inference_result
        
        # Track progress calls
        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        config = BatchInferenceConfig(
            chunk_size=20,
            max_workers=1,
            progress_callback=progress_callback
        )
        
        # Run batch inference
        result = self.batch_engine.predict_batch(
            self.test_data, config, str(self.temp_dir)
        )
        
        assert len(progress_calls) > 0
        # Check that progress was reported
        final_call = progress_calls[-1]
        assert final_call[0] == final_call[1]  # current == total at the end
    
    def test_predict_batch_save_intermediate(self):
        """Test batch processing with intermediate file saving."""
        # Mock inference results
        mock_inference_result = InferenceResult(
            predictions=np.array([1, 0]),
            inference_time=0.1
        )
        self.mock_model_engine._perform_inference.return_value = mock_inference_result
        
        config = BatchInferenceConfig(
            chunk_size=50,
            max_workers=1,
            save_intermediate=True,
            output_format='parquet'
        )
        
        # Run batch inference
        result = self.batch_engine.predict_batch(
            self.test_data, config, str(self.temp_dir)
        )
        
        # Check that intermediate files were created
        assert len(result.output_files) > 0
        for file_path in result.output_files:
            assert Path(file_path).exists()
            assert file_path.endswith('.parquet')
    
    def test_predict_batch_different_output_formats(self):
        """Test batch processing with different output formats."""
        mock_inference_result = InferenceResult(
            predictions=np.array([1, 0]),
            inference_time=0.1
        )
        self.mock_model_engine._perform_inference.return_value = mock_inference_result
        
        # Test CSV format
        config = BatchInferenceConfig(
            chunk_size=50,
            max_workers=1,
            save_intermediate=False,
            output_format='csv'
        )
        
        result = self.batch_engine.predict_batch(
            self.test_data, config, str(self.temp_dir)
        )
        
        assert len(result.output_files) > 0
        assert result.output_files[0].endswith('.csv')
    
    def test_predict_batch_with_errors(self):
        """Test batch processing with some chunks failing."""
        # Mock inference to fail sometimes
        def mock_inference_side_effect(*args, **kwargs):
            # Fail every other call
            if not hasattr(mock_inference_side_effect, 'call_count'):
                mock_inference_side_effect.call_count = 0
            mock_inference_side_effect.call_count += 1
            
            if mock_inference_side_effect.call_count % 2 == 0:
                raise Exception("Mock inference error")
            
            return InferenceResult(
                predictions=np.array([1, 0]),
                inference_time=0.1
            )
        
        self.mock_model_engine._perform_inference.side_effect = mock_inference_side_effect
        
        config = BatchInferenceConfig(
            chunk_size=25,
            max_workers=1
        )
        
        # Run batch inference
        result = self.batch_engine.predict_batch(
            self.test_data, config, str(self.temp_dir)
        )
        
        # Should have some failed chunks
        assert result.failed_chunks > 0
        assert len(result.error_details) > 0
        assert result.chunks_processed + result.failed_chunks > 0
    
    def test_stop_processing(self):
        """Test stopping batch processing."""
        # This is a simple test since stopping is mainly for long-running processes
        self.batch_engine.stop_processing()
        assert self.batch_engine._stop_processing is True
    
    def test_calculate_batch_summary_stats(self):
        """Test summary statistics calculation."""
        predictions = [1, 0, 1, 1, 0]
        probabilities = [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.9, 0.1]]
        confidence_scores = [0.9, 0.8, 0.7, 0.8, 0.9]
        
        stats = self.batch_engine._calculate_batch_summary_stats(
            predictions, probabilities, confidence_scores
        )
        
        assert stats['total_predictions'] == 5
        assert stats['unique_predictions'] == 2  # 0 and 1
        assert 'confidence_mean' in stats
        assert 'confidence_std' in stats
        assert 'low_confidence_count' in stats
        assert 'high_confidence_count' in stats


class TestRealTimeInferenceConfig:
    """Test RealTimeInferenceConfig dataclass."""
    
    def test_realtime_config_defaults(self):
        """Test default configuration values."""
        config = RealTimeInferenceConfig()
        
        assert config.max_batch_size == 32
        assert config.timeout_seconds == 1.0
        assert config.confidence_threshold == 0.5
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 300
        assert config.max_queue_size == 1000
    
    def test_realtime_config_custom_values(self):
        """Test custom configuration values."""
        config = RealTimeInferenceConfig(
            max_batch_size=16,
            timeout_seconds=0.5,
            confidence_threshold=0.8,
            enable_caching=False,
            cache_ttl_seconds=600,
            max_queue_size=500
        )
        
        assert config.max_batch_size == 16
        assert config.timeout_seconds == 0.5
        assert config.confidence_threshold == 0.8
        assert config.enable_caching is False
        assert config.cache_ttl_seconds == 600
        assert config.max_queue_size == 500


class TestRealTimeInferenceEngine:
    """Test RealTimeInferenceEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock model engine
        self.mock_model_engine = Mock(spec=ModelInferenceEngine)
        self.mock_model_engine.current_model = Mock()
        
        # Mock inference results
        self.mock_inference_result = InferenceResult(
            predictions=np.array([1]),
            prediction_probabilities=np.array([[0.2, 0.8]]),
            confidence_scores=np.array([0.8]),
            inference_time=0.01
        )
        self.mock_model_engine._perform_inference.return_value = self.mock_inference_result
        
        # Create real-time engine
        self.config = RealTimeInferenceConfig(
            max_batch_size=4,
            timeout_seconds=0.1,
            enable_caching=True
        )
        self.realtime_engine = RealTimeInferenceEngine(self.mock_model_engine, self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.realtime_engine.is_running:
            self.realtime_engine.stop()
    
    def test_start_stop_engine(self):
        """Test starting and stopping the real-time engine."""
        assert not self.realtime_engine.is_running
        
        # Start engine
        self.realtime_engine.start()
        assert self.realtime_engine.is_running
        assert self.realtime_engine.processing_thread is not None
        
        # Stop engine
        self.realtime_engine.stop()
        assert not self.realtime_engine.is_running
    
    def test_predict_single_dict_input(self):
        """Test single prediction with dictionary input."""
        self.realtime_engine.start()
        
        try:
            input_data = {"feature1": 1.0, "feature2": 2.0}
            
            result = self.realtime_engine.predict_single(input_data)
            
            assert 'request_id' in result
            assert 'predictions' in result
            assert 'latency_ms' in result
            assert result['from_cache'] in [True, False]
            
        finally:
            self.realtime_engine.stop()
    
    def test_predict_single_dataframe_input(self):
        """Test single prediction with DataFrame input."""
        self.realtime_engine.start()
        
        try:
            input_data = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
            
            result = self.realtime_engine.predict_single(input_data)
            
            assert 'request_id' in result
            assert 'predictions' in result
            assert 'latency_ms' in result
            
        finally:
            self.realtime_engine.stop()
    
    def test_predict_single_with_caching(self):
        """Test single prediction with caching enabled."""
        self.realtime_engine.start()
        
        try:
            input_data = {"feature1": 1.0, "feature2": 2.0}
            
            # First prediction - should miss cache
            result1 = self.realtime_engine.predict_single(input_data)
            assert result1['from_cache'] is False
            
            # Second prediction with same input - should hit cache
            result2 = self.realtime_engine.predict_single(input_data)
            assert result2['from_cache'] is True
            
            # Results should be the same
            assert result1['predictions'] == result2['predictions']
            
        finally:
            self.realtime_engine.stop()
    
    def test_predict_single_not_running(self):
        """Test prediction when engine is not running."""
        input_data = {"feature1": 1.0, "feature2": 2.0}
        
        with pytest.raises(RuntimeError, match="not running"):
            self.realtime_engine.predict_single(input_data)
    
    def test_predict_batch_realtime(self):
        """Test batch prediction in real-time mode."""
        self.realtime_engine.start()
        
        try:
            input_batch = [
                {"feature1": 1.0, "feature2": 2.0},
                {"feature1": 3.0, "feature2": 4.0},
                {"feature1": 5.0, "feature2": 6.0}
            ]
            
            results = self.realtime_engine.predict_batch_realtime(input_batch)
            
            assert len(results) == 3
            for result in results:
                assert 'request_id' in result
                assert 'predictions' in result
                assert 'latency_ms' in result
                
        finally:
            self.realtime_engine.stop()
    
    def test_predict_batch_realtime_with_request_ids(self):
        """Test batch prediction with custom request IDs."""
        self.realtime_engine.start()
        
        try:
            input_batch = [
                {"feature1": 1.0, "feature2": 2.0},
                {"feature1": 3.0, "feature2": 4.0}
            ]
            request_ids = ["req1", "req2"]
            
            results = self.realtime_engine.predict_batch_realtime(input_batch, request_ids)
            
            assert len(results) == 2
            assert results[0]['request_id'] == "req1"
            assert results[1]['request_id'] == "req2"
            
        finally:
            self.realtime_engine.stop()
    
    def test_predict_batch_realtime_mismatched_ids(self):
        """Test batch prediction with mismatched input and ID counts."""
        input_batch = [{"feature1": 1.0}]
        request_ids = ["req1", "req2"]  # More IDs than inputs
        
        with pytest.raises(ValueError, match="Number of inputs must match"):
            self.realtime_engine.predict_batch_realtime(input_batch, request_ids)
    
    def test_get_metrics(self):
        """Test metrics collection."""
        self.realtime_engine.start()
        
        try:
            # Make some predictions to generate metrics
            input_data = {"feature1": 1.0, "feature2": 2.0}
            self.realtime_engine.predict_single(input_data)
            self.realtime_engine.predict_single(input_data)  # Should hit cache
            
            metrics = self.realtime_engine.get_metrics()
            
            assert 'total_requests' in metrics
            assert 'successful_predictions' in metrics
            assert 'failed_predictions' in metrics
            assert 'cache_hits' in metrics
            assert 'cache_misses' in metrics
            assert 'average_latency' in metrics
            assert 'cache_size' in metrics
            assert 'cache_hit_rate' in metrics
            
            assert metrics['total_requests'] >= 2
            assert metrics['cache_hits'] >= 1
            
        finally:
            self.realtime_engine.stop()
    
    def test_clear_cache(self):
        """Test cache clearing."""
        self.realtime_engine.start()
        
        try:
            # Add something to cache
            input_data = {"feature1": 1.0, "feature2": 2.0}
            self.realtime_engine.predict_single(input_data)
            
            # Verify cache has content
            assert len(self.realtime_engine.prediction_cache) > 0
            
            # Clear cache
            self.realtime_engine.clear_cache()
            
            # Verify cache is empty
            assert len(self.realtime_engine.prediction_cache) == 0
            
        finally:
            self.realtime_engine.stop()
    
    def test_health_check(self):
        """Test health check functionality."""
        # Health check when stopped
        health = self.realtime_engine.health_check()
        assert health['status'] == 'stopped'
        assert 'queue_size' in health
        assert 'active_requests' in health
        assert 'cache_size' in health
        assert 'metrics' in health
        
        # Health check when running
        self.realtime_engine.start()
        try:
            health = self.realtime_engine.health_check()
            assert health['status'] == 'healthy'
        finally:
            self.realtime_engine.stop()
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        # Create engine with very short cache TTL
        config = RealTimeInferenceConfig(
            cache_ttl_seconds=0.1,  # 100ms
            enable_caching=True
        )
        engine = RealTimeInferenceEngine(self.mock_model_engine, config)
        engine.start()
        
        try:
            input_data = {"feature1": 1.0, "feature2": 2.0}
            
            # First prediction
            result1 = engine.predict_single(input_data)
            assert result1['from_cache'] is False
            
            # Immediate second prediction - should hit cache
            result2 = engine.predict_single(input_data)
            assert result2['from_cache'] is True
            
            # Wait for cache to expire
            time.sleep(0.15)
            
            # Third prediction - should miss cache due to expiration
            result3 = engine.predict_single(input_data)
            assert result3['from_cache'] is False
            
        finally:
            engine.stop()
    
    def test_confidence_threshold_warning(self):
        """Test low confidence warning."""
        # Mock low confidence result
        low_conf_result = InferenceResult(
            predictions=np.array([1]),
            confidence_scores=np.array([0.3]),  # Below threshold
            inference_time=0.01
        )
        self.mock_model_engine._perform_inference.return_value = low_conf_result
        
        # Create engine with higher confidence threshold
        config = RealTimeInferenceConfig(confidence_threshold=0.5)
        engine = RealTimeInferenceEngine(self.mock_model_engine, config)
        engine.start()
        
        try:
            input_data = {"feature1": 1.0, "feature2": 2.0}
            result = engine.predict_single(input_data)
            
            assert 'low_confidence_warning' in result
            assert result['low_confidence_warning'] is True
            
        finally:
            engine.stop()
    
    def test_request_timeout(self):
        """Test request timeout handling."""
        # Create engine with short timeout
        config = RealTimeInferenceConfig(timeout_seconds=0.1)  # 100ms timeout
        engine = RealTimeInferenceEngine(self.mock_model_engine, config)
        
        # Mock inference to take longer than timeout
        def slow_inference(*args, **kwargs):
            time.sleep(0.5)  # 500ms - longer than timeout
            return self.mock_inference_result
        
        self.mock_model_engine._perform_inference.side_effect = slow_inference
        engine.start()
        
        try:
            input_data = {"feature1": 1.0, "feature2": 2.0}
            
            # This should timeout
            with pytest.raises(RuntimeError, match="timeout"):
                engine.predict_single(input_data)
                
        finally:
            engine.stop()


if __name__ == "__main__":
    pytest.main([__file__])