"""Performance and load testing for ML pipeline components."""

import pytest
import tempfile
import pandas as pd
import numpy as np
import time
import threading
import multiprocessing
import psutil
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock
import shutil

from mlpipeline.data.preprocessing import DataPreprocessor
from mlpipeline.models.training import ModelTrainer
from mlpipeline.models.inference import ModelInferenceEngine, BatchInferenceEngine
from mlpipeline.core.orchestrator import PipelineOrchestrator
from mlpipeline.core.interfaces import ExecutionContext, ComponentType, PipelineStage


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"{self.name} took {self.duration:.4f} seconds")


class MemoryProfiler:
    """Memory usage profiler."""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
    
    def start(self):
        """Start memory profiling."""
        gc.collect()  # Force garbage collection
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
    
    def update_peak(self):
        """Update peak memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def stop(self):
        """Stop memory profiling and return results."""
        gc.collect()
        self.final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return {
            'initial_mb': self.initial_memory,
            'peak_mb': self.peak_memory,
            'final_mb': self.final_memory,
            'memory_increase_mb': self.final_memory - self.initial_memory
        }


class TestDataPreprocessingPerformance:
    """Test data preprocessing performance with various data sizes."""    

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.preprocessor = DataPreprocessor()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_synthetic_data(self, n_rows, n_features=10, n_categorical=3):
        """Create synthetic dataset for performance testing."""
        np.random.seed(42)
        
        # Numerical features
        numerical_data = {}
        for i in range(n_features - n_categorical):
            numerical_data[f'num_feature_{i}'] = np.random.randn(n_rows)
        
        # Categorical features
        categorical_data = {}
        for i in range(n_categorical):
            categories = [f'cat_{j}' for j in range(5)]  # 5 categories each
            categorical_data[f'cat_feature_{i}'] = np.random.choice(categories, n_rows)
        
        # Target
        target_data = {'target': np.random.randint(0, 2, n_rows)}
        
        # Combine all data
        all_data = {**numerical_data, **categorical_data, **target_data}
        return pd.DataFrame(all_data)
    
    def create_preprocessing_context(self, data_path):
        """Create execution context for preprocessing."""
        return ExecutionContext(
            experiment_id="perf_test",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={
                'data': {
                    'preprocessing': {
                        'steps': [
                            {
                                'name': 'scaler',
                                'transformer': 'standard_scaler',
                                'columns': [col for col in data_path if col.startswith('num_')]
                            },
                            {
                                'name': 'encoder',
                                'transformer': 'one_hot_encoder',
                                'columns': [col for col in data_path if col.startswith('cat_')],
                                'parameters': {'sparse_output': False, 'handle_unknown': 'ignore'}
                            }
                        ],
                        'data_split': {
                            'train_size': 0.7,
                            'val_size': 0.15,
                            'test_size': 0.15,
                            'target_column': 'target',
                            'random_state': 42
                        }
                    }
                }
            },
            artifacts_path=self.temp_dir,
            logger=Mock(),
            metadata={}
        ) 
   
    @pytest.mark.parametrize("n_rows", [1000, 10000, 50000, 100000])
    def test_preprocessing_scalability(self, n_rows):
        """Test preprocessing performance with different data sizes."""
        # Create synthetic data
        data = self.create_synthetic_data(n_rows)
        data_path = Path(self.temp_dir) / f"data_{n_rows}.parquet"
        data.to_parquet(data_path, index=False)
        
        # Create ingested data file
        ingested_path = Path(self.temp_dir) / "ingested_data.parquet"
        data.to_parquet(ingested_path, index=False)
        
        # Create context
        context = self.create_preprocessing_context(data.columns)
        
        # Profile memory and time
        profiler = MemoryProfiler()
        profiler.start()
        
        with PerformanceTimer(f"Preprocessing {n_rows} rows") as timer:
            result = self.preprocessor.execute(context)
            profiler.update_peak()
        
        memory_stats = profiler.stop()
        
        # Assertions
        assert result.success is True
        assert timer.duration < (n_rows / 1000) * 2  # Max 2 seconds per 1000 rows
        assert memory_stats['memory_increase_mb'] < (n_rows / 1000) * 50  # Max 50MB per 1000 rows
        
        # Performance metrics
        rows_per_second = n_rows / timer.duration
        print(f"Processed {rows_per_second:.0f} rows/second")
        print(f"Memory usage: {memory_stats}")
        
        # Verify output quality
        assert result.metrics['original_rows'] == n_rows
        assert len(result.artifacts) >= 3  # train, val, test files
    
    def test_preprocessing_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Create large dataset
        large_data = self.create_synthetic_data(200000, n_features=50)
        data_path = Path(self.temp_dir) / "large_data.parquet"
        large_data.to_parquet(data_path, index=False)
        
        # Create ingested data file
        ingested_path = Path(self.temp_dir) / "ingested_data.parquet"
        large_data.to_parquet(ingested_path, index=False)
        
        # Create context
        context = self.create_preprocessing_context(large_data.columns)
        
        # Monitor memory throughout execution
        profiler = MemoryProfiler()
        profiler.start()
        
        result = self.preprocessor.execute(context)
        
        memory_stats = profiler.stop()
        
        assert result.success is True
        # Memory increase should be reasonable (less than 2x original data size)
        original_size_mb = large_data.memory_usage(deep=True).sum() / 1024 / 1024
        assert memory_stats['memory_increase_mb'] < original_size_mb * 2
        
        print(f"Original data size: {original_size_mb:.2f} MB")
        print(f"Memory increase: {memory_stats['memory_increase_mb']:.2f} MB")
        print(f"Memory efficiency ratio: {memory_stats['memory_increase_mb'] / original_size_mb:.2f}")
    
    def test_concurrent_preprocessing(self):
        """Test concurrent preprocessing operations."""
        n_concurrent = 4
        n_rows_each = 5000
        
        def run_preprocessing(worker_id):
            """Run preprocessing in a separate thread."""
            temp_dir = tempfile.mkdtemp()
            try:
                # Create data for this worker
                data = self.create_synthetic_data(n_rows_each)
                data_path = Path(temp_dir) / f"data_worker_{worker_id}.parquet"
                data.to_parquet(data_path, index=False)
                
                # Create ingested data file
                ingested_path = Path(temp_dir) / "ingested_data.parquet"
                data.to_parquet(ingested_path, index=False)
                
                # Create preprocessor and context
                preprocessor = DataPreprocessor()
                context = ExecutionContext(
                    experiment_id=f"concurrent_test_{worker_id}",
                    stage_name="preprocessing",
                    component_type=ComponentType.DATA_PREPROCESSING,
                    config={
                        'data': {
                            'preprocessing': {
                                'steps': [
                                    {
                                        'name': 'scaler',
                                        'transformer': 'standard_scaler'
                                    }
                                ],
                                'data_split': {
                                    'train_size': 0.8,
                                    'val_size': 0.1,
                                    'test_size': 0.1,
                                    'target_column': 'target',
                                    'random_state': 42
                                }
                            }
                        }
                    },
                    artifacts_path=temp_dir,
                    logger=Mock(),
                    metadata={}
                )
                
                start_time = time.time()
                result = preprocessor.execute(context)
                duration = time.time() - start_time
                
                return {
                    'worker_id': worker_id,
                    'success': result.success,
                    'duration': duration,
                    'rows_processed': n_rows_each
                }
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Run concurrent preprocessing
        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            start_time = time.time()
            futures = [executor.submit(run_preprocessing, i) for i in range(n_concurrent)]
            results = [future.result() for future in futures]
            total_duration = time.time() - start_time
        
        # Verify all succeeded
        assert all(r['success'] for r in results)
        
        # Performance analysis
        total_rows = sum(r['rows_processed'] for r in results)
        throughput = total_rows / total_duration
        
        print(f"Concurrent processing: {n_concurrent} workers")
        print(f"Total rows processed: {total_rows}")
        print(f"Total time: {total_duration:.2f} seconds")
        print(f"Throughput: {throughput:.0f} rows/second")
        
        # Should be faster than sequential processing
        sequential_estimate = sum(r['duration'] for r in results)
        speedup = sequential_estimate / total_duration
        print(f"Estimated speedup: {speedup:.2f}x")
        
        assert speedup > 1.5  # Should have some speedup from concurrency


class TestModelTrainingPerformance:
    """Test model training performance with various configurations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = ModelTrainer()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_training_data(self, n_samples, n_features=20):
        """Create training data for performance testing."""
        np.random.seed(42)
        
        # Create features
        X = np.random.randn(n_samples, n_features)
        
        # Create target with some signal
        weights = np.random.randn(n_features)
        y = (X @ weights + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        # Create train/val/test splits
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        train_data = pd.DataFrame(X[:train_size], columns=[f'feature_{i}' for i in range(n_features)])
        train_data['target'] = y[:train_size]
        
        val_data = pd.DataFrame(X[train_size:train_size+val_size], columns=[f'feature_{i}' for i in range(n_features)])
        val_data['target'] = y[train_size:train_size+val_size]
        
        test_data = pd.DataFrame(X[train_size+val_size:], columns=[f'feature_{i}' for i in range(n_features)])
        test_data['target'] = y[train_size+val_size:]
        
        return train_data, val_data, test_data
    
    @pytest.mark.parametrize("n_samples,model_type", [
        (1000, "logistic_regression"),
        (5000, "logistic_regression"),
        (10000, "logistic_regression"),
        (1000, "random_forest_classifier"),
        (5000, "random_forest_classifier"),
    ])
    def test_training_scalability(self, n_samples, model_type):
        """Test training performance with different data sizes and models."""
        # Create training data
        train_data, val_data, test_data = self.create_training_data(n_samples)
        
        # Save data files
        train_data.to_parquet(Path(self.temp_dir) / "train_preprocessed.parquet", index=False)
        val_data.to_parquet(Path(self.temp_dir) / "val_preprocessed.parquet", index=False)
        test_data.to_parquet(Path(self.temp_dir) / "test_preprocessed.parquet", index=False)
        
        # Create context
        context = ExecutionContext(
            experiment_id="training_perf_test",
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config={
                'training': {
                    'model': {
                        'framework': 'sklearn',
                        'model_type': model_type,
                        'task_type': 'classification',
                        'parameters': {
                            'random_state': 42,
                            'n_estimators': 10 if 'forest' in model_type else None
                        }
                    },
                    'target_column': 'target'
                }
            },
            artifacts_path=self.temp_dir,
            logger=Mock(),
            metadata={}
        )
        
        # Profile performance
        profiler = MemoryProfiler()
        profiler.start()
        
        with PerformanceTimer(f"Training {model_type} on {n_samples} samples") as timer:
            result = self.trainer.execute(context)
            profiler.update_peak()
        
        memory_stats = profiler.stop()
        
        # Assertions
        assert result.success is True
        assert timer.duration < 60  # Should complete within 60 seconds
        assert result.metrics['train_accuracy'] > 0.5  # Should learn something
        
        # Performance metrics
        samples_per_second = n_samples / timer.duration
        print(f"Training speed: {samples_per_second:.0f} samples/second")
        print(f"Memory usage: {memory_stats}")
        print(f"Model accuracy: {result.metrics.get('train_accuracy', 'N/A')}")
        
        # Model-specific performance expectations
        if model_type == "logistic_regression":
            assert timer.duration < n_samples / 100  # Should be very fast
        elif "forest" in model_type:
            assert timer.duration < n_samples / 10   # Tree models are slower
    
    def test_hyperparameter_optimization_performance(self):
        """Test performance of hyperparameter optimization."""
        # Create moderate-sized dataset
        train_data, val_data, test_data = self.create_training_data(5000)
        
        # Save data files
        train_data.to_parquet(Path(self.temp_dir) / "train_preprocessed.parquet", index=False)
        val_data.to_parquet(Path(self.temp_dir) / "val_preprocessed.parquet", index=False)
        test_data.to_parquet(Path(self.temp_dir) / "test_preprocessed.parquet", index=False)
        
        # Create context with hyperparameter optimization
        context = ExecutionContext(
            experiment_id="hyperparam_perf_test",
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config={
                'training': {
                    'model': {
                        'framework': 'sklearn',
                        'model_type': 'random_forest_classifier',
                        'task_type': 'classification',
                        'parameters': {'random_state': 42},
                        'hyperparameter_tuning': {
                            'enabled': True,
                            'method': 'random_search',
                            'n_trials': 10,
                            'parameters': {
                                'n_estimators': [5, 10, 20],
                                'max_depth': [3, 5, 7]
                            }
                        }
                    },
                    'target_column': 'target'
                }
            },
            artifacts_path=self.temp_dir,
            logger=Mock(),
            metadata={}
        )
        
        # Profile performance
        with PerformanceTimer("Hyperparameter optimization") as timer:
            result = self.trainer.execute(context)
        
        # Assertions
        assert result.success is True
        assert timer.duration < 300  # Should complete within 5 minutes
        assert 'best_params' in result.metadata
        
        print(f"Hyperparameter optimization took {timer.duration:.2f} seconds")
        print(f"Best parameters: {result.metadata.get('best_params', 'N/A')}")
        print(f"Best score: {result.metadata.get('best_score', 'N/A')}")
    
    def test_training_memory_leak(self):
        """Test for memory leaks during repeated training."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run multiple training iterations
        for i in range(5):
            # Create fresh data each time
            train_data, val_data, test_data = self.create_training_data(1000)
            
            # Save to temporary files
            temp_dir = tempfile.mkdtemp()
            try:
                train_data.to_parquet(Path(temp_dir) / "train_preprocessed.parquet", index=False)
                val_data.to_parquet(Path(temp_dir) / "val_preprocessed.parquet", index=False)
                test_data.to_parquet(Path(temp_dir) / "test_preprocessed.parquet", index=False)
                
                context = ExecutionContext(
                    experiment_id=f"memory_test_{i}",
                    stage_name="training",
                    component_type=ComponentType.MODEL_TRAINING,
                    config={
                        'training': {
                            'model': {
                                'framework': 'sklearn',
                                'model_type': 'logistic_regression',
                                'task_type': 'classification',
                                'parameters': {'random_state': 42}
                            },
                            'target_column': 'target'
                        }
                    },
                    artifacts_path=temp_dir,
                    logger=Mock(),
                    metadata={}
                )
                
                # Create fresh trainer each time
                trainer = ModelTrainer()
                result = trainer.execute(context)
                assert result.success is True
                
                # Force cleanup
                del trainer
                gc.collect()
                
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Memory increase should be minimal (less than 100MB)
        assert memory_increase < 100, f"Potential memory leak: {memory_increase:.2f} MB increase"

class TestInferencePerformance:
    """Test model inference performance and throughput."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_model_and_data(self, n_samples=1000, n_features=10):
        """Create mock trained model and inference data."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create training data
        np.random.seed(42)
        X_train = np.random.randn(1000, n_features)
        y_train = np.random.randint(0, 2, 1000)
        
        # Train a simple model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model
        model_path = Path(self.temp_dir) / "trained_model.joblib"
        joblib.dump(model, model_path)
        
        # Create inference data
        X_inference = np.random.randn(n_samples, n_features)
        inference_data = pd.DataFrame(X_inference, columns=[f'feature_{i}' for i in range(n_features)])
        
        # Save inference data
        inference_path = Path(self.temp_dir) / "test_preprocessed.parquet"
        inference_data.to_parquet(inference_path, index=False)
        
        return model_path, inference_path, inference_data
    
    @pytest.mark.parametrize("n_samples", [100, 1000, 10000, 50000])
    def test_batch_inference_throughput(self, n_samples):
        """Test batch inference throughput with different data sizes."""
        # Create model and data
        model_path, inference_path, inference_data = self.create_mock_model_and_data(n_samples)
        
        # Create inference engine
        inference_engine = ModelInferenceEngine()
        
        # Create context
        context = ExecutionContext(
            experiment_id="inference_perf_test",
            stage_name="inference",
            component_type=ComponentType.MODEL_INFERENCE,
            config={
                'inference': {
                    'model_path': str(model_path),
                    'batch_size': min(1000, n_samples // 10)
                }
            },
            artifacts_path=self.temp_dir,
            logger=Mock(),
            metadata={}
        )
        
        # Profile performance
        profiler = MemoryProfiler()
        profiler.start()
        
        with PerformanceTimer(f"Batch inference on {n_samples} samples") as timer:
            result = inference_engine.execute(context)
            profiler.update_peak()
        
        memory_stats = profiler.stop()
        
        # Assertions
        assert result.success is True
        assert timer.duration < n_samples / 100  # Should process at least 100 samples/second
        
        # Performance metrics
        throughput = n_samples / timer.duration
        print(f"Inference throughput: {throughput:.0f} samples/second")
        print(f"Memory usage: {memory_stats}")
        
        # Throughput should scale reasonably
        if n_samples >= 1000:
            assert throughput > 500  # Should achieve at least 500 samples/second
    
    def test_concurrent_inference_requests(self):
        """Test handling concurrent inference requests."""
        n_concurrent = 8
        n_samples_each = 1000
        
        # Create shared model and data
        model_path, _, _ = self.create_mock_model_and_data(n_samples_each)
        
        def run_inference(worker_id):
            """Run inference in a separate thread."""
            temp_dir = tempfile.mkdtemp()
            try:
                # Create data for this worker
                np.random.seed(worker_id)  # Different seed for each worker
                inference_data = pd.DataFrame(
                    np.random.randn(n_samples_each, 10),
                    columns=[f'feature_{i}' for i in range(10)]
                )
                inference_path = Path(temp_dir) / "test_preprocessed.parquet"
                inference_data.to_parquet(inference_path, index=False)
                
                # Create inference engine
                inference_engine = ModelInferenceEngine()
                
                context = ExecutionContext(
                    experiment_id=f"concurrent_inference_{worker_id}",
                    stage_name="inference",
                    component_type=ComponentType.MODEL_INFERENCE,
                    config={
                        'inference': {
                            'model_path': str(model_path),
                            'batch_size': 100
                        }
                    },
                    artifacts_path=temp_dir,
                    logger=Mock(),
                    metadata={}
                )
                
                start_time = time.time()
                result = inference_engine.execute(context)
                duration = time.time() - start_time
                
                return {
                    'worker_id': worker_id,
                    'success': result.success,
                    'duration': duration,
                    'samples_processed': n_samples_each
                }
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Run concurrent inference
        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            start_time = time.time()
            futures = [executor.submit(run_inference, i) for i in range(n_concurrent)]
            results = [future.result() for future in futures]
            total_duration = time.time() - start_time
        
        # Verify all succeeded
        assert all(r['success'] for r in results)
        
        # Performance analysis
        total_samples = sum(r['samples_processed'] for r in results)
        throughput = total_samples / total_duration
        
        print(f"Concurrent inference: {n_concurrent} workers")
        print(f"Total samples processed: {total_samples}")
        print(f"Total time: {total_duration:.2f} seconds")
        print(f"Aggregate throughput: {throughput:.0f} samples/second")
        
        # Should handle concurrent requests efficiently
        assert throughput > 1000  # Should achieve good aggregate throughput
    
    def test_inference_latency_distribution(self):
        """Test inference latency distribution for real-time scenarios."""
        # Create small batches to simulate real-time requests
        model_path, _, _ = self.create_mock_model_and_data(100)
        
        latencies = []
        n_requests = 100
        
        for i in range(n_requests):
            # Create small inference batch
            inference_data = pd.DataFrame(
                np.random.randn(1, 10),  # Single sample
                columns=[f'feature_{i}' for i in range(10)]
            )
            
            temp_dir = tempfile.mkdtemp()
            try:
                inference_path = Path(temp_dir) / "test_preprocessed.parquet"
                inference_data.to_parquet(inference_path, index=False)
                
                inference_engine = ModelInferenceEngine()
                context = ExecutionContext(
                    experiment_id=f"latency_test_{i}",
                    stage_name="inference",
                    component_type=ComponentType.MODEL_INFERENCE,
                    config={
                        'inference': {
                            'model_path': str(model_path),
                            'batch_size': 1
                        }
                    },
                    artifacts_path=temp_dir,
                    logger=Mock(),
                    metadata={}
                )
                
                start_time = time.time()
                result = inference_engine.execute(context)
                latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                assert result.success is True
                latencies.append(latency)
                
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Analyze latency distribution
        latencies = np.array(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        mean_latency = np.mean(latencies)
        
        print(f"Latency statistics (ms):")
        print(f"  Mean: {mean_latency:.2f}")
        print(f"  P50: {p50:.2f}")
        print(f"  P95: {p95:.2f}")
        print(f"  P99: {p99:.2f}")
        
        # Real-time inference requirements
        assert p95 < 100, f"P95 latency too high: {p95:.2f}ms"  # 95% under 100ms
        assert p99 < 200, f"P99 latency too high: {p99:.2f}ms"  # 99% under 200ms
        assert mean_latency < 50, f"Mean latency too high: {mean_latency:.2f}ms"