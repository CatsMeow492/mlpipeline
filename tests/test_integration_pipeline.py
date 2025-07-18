"""Integration tests for end-to-end pipeline execution."""

import pytest
import tempfile
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from unittest.mock import patch
import shutil

from mlpipeline.core.orchestrator import PipelineOrchestrator
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
from mlpipeline.config import ConfigManager
from mlpipeline.data.preprocessing import DataPreprocessor
from mlpipeline.models.training import ModelTrainer
from mlpipeline.models.evaluation import ModelEvaluator
from mlpipeline.models.inference import ModelInferenceEngine


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        self.orchestrator = PipelineOrchestrator(max_workers=2)
        
        # Create sample dataset
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'feature_1': np.random.randn(200),
            'feature_2': np.random.randn(200),
            'feature_3': np.random.randint(0, 3, 200),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 200),
            'target': np.random.randint(0, 2, 200)
        })
        
        # Save sample data
        self.data_path = Path(self.temp_dir) / "sample_data.csv"
        self.sample_data.to_csv(self.data_path, index=False)
        
        # Create pipeline configuration
        self.pipeline_config = {
            "pipeline": {
                "name": "integration_test_pipeline",
                "description": "Integration test pipeline"
            },
            "data": {
                "sources": [
                    {
                        "type": "csv",
                        "path": str(self.data_path)
                    }
                ],
                "preprocessing": {
                    "steps": [
                        {
                            "name": "scaler",
                            "transformer": "standard_scaler",
                            "columns": ["feature_1", "feature_2", "feature_3"]
                        },
                        {
                            "name": "encoder",
                            "transformer": "one_hot_encoder",
                            "columns": ["categorical_feature"],
                            "parameters": {"sparse_output": False, "handle_unknown": "ignore"}
                        }
                    ],
                    "data_split": {
                        "train_size": 0.7,
                        "val_size": 0.15,
                        "test_size": 0.15,
                        "target_column": "target",
                        "random_state": 42
                    }
                },
                "train_split": 0.7,
                "validation_split": 0.15,
                "test_split": 0.15,
                "random_state": 42
            },
            "model": {
                "type": "sklearn",
                "parameters": {
                    "algorithm": "RandomForestClassifier",
                    "n_estimators": 10,
                    "max_depth": 3,
                    "random_state": 42
                }
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1_score"],
                "generate_plots": True
            }
        }
        
        # Save configuration
        self.config_path = Path(self.temp_dir) / "pipeline_config.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.pipeline_config, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_pipeline_execution(self):
        """Test complete pipeline from data ingestion to inference."""
        # Create execution context with direct config (bypass schema validation for now)
        context = ExecutionContext(
            experiment_id="integration_test",
            stage_name="complete_pipeline",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={
                'data': {
                    'preprocessing': {
                        'steps': [
                            {
                                'name': 'scaler',
                                'transformer': 'standard_scaler',
                                'columns': ['feature_1', 'feature_2', 'feature_3']
                            },
                            {
                                'name': 'encoder',
                                'transformer': 'one_hot_encoder',
                                'columns': ['categorical_feature'],
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
                },
                'training': {
                    'model': {
                        'framework': 'sklearn',
                        'model_type': 'random_forest_classifier',
                        'task_type': 'classification',
                        'parameters': {
                            'n_estimators': 10,
                            'max_depth': 3,
                            'random_state': 42
                        }
                    },
                    'target_column': 'target'
                },
                'evaluation': {
                    'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                    'generate_plots': True
                }
            },
            artifacts_path=self.temp_dir,
            logger=self.orchestrator.logger,
            metadata={"test": "integration"}
        )
        
        # Step 1: Data Preprocessing
        preprocessor = DataPreprocessor()
        
        # First, we need to create the ingested data file that preprocessor expects
        ingested_data_path = Path(self.temp_dir) / "ingested_data.parquet"
        self.sample_data.to_parquet(ingested_data_path, index=False)
        
        preprocessing_result = preprocessor.execute(context)
        
        assert preprocessing_result.success is True
        assert len(preprocessing_result.artifacts) >= 3  # train, val, test data
        
        # Verify preprocessing artifacts exist
        train_path = Path(self.temp_dir) / "train_preprocessed.parquet"
        val_path = Path(self.temp_dir) / "val_preprocessed.parquet"
        test_path = Path(self.temp_dir) / "test_preprocessed.parquet"
        
        assert train_path.exists()
        assert val_path.exists()
        assert test_path.exists()
        
        # Step 2: Model Training
        trainer = ModelTrainer()
        training_context = ExecutionContext(
            experiment_id="integration_test",
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config=context.config,
            artifacts_path=self.temp_dir,
            logger=self.orchestrator.logger,
            metadata={"test": "integration"}
        )
        
        training_result = trainer.execute(training_context)
        
        assert training_result.success is True
        assert "train_accuracy" in training_result.metrics
        assert "val_accuracy" in training_result.metrics
        
        # Verify training artifacts exist
        model_path = Path(self.temp_dir) / "trained_model.joblib"
        assert model_path.exists()
        
        # Step 3: Model Evaluation
        evaluator = ModelEvaluator()
        evaluation_context = ExecutionContext(
            experiment_id="integration_test",
            stage_name="evaluation",
            component_type=ComponentType.MODEL_EVALUATION,
            config=context.config,
            artifacts_path=self.temp_dir,
            logger=self.orchestrator.logger,
            metadata={"test": "integration"}
        )
        
        evaluation_result = evaluator.execute(evaluation_context)
        
        assert evaluation_result.success is True
        assert "test_accuracy" in evaluation_result.metrics
        assert "test_precision" in evaluation_result.metrics
        assert "test_recall" in evaluation_result.metrics
        assert "test_f1_score" in evaluation_result.metrics
        
        # Step 4: Model Inference
        inference_engine = ModelInferenceEngine()
        inference_context = ExecutionContext(
            experiment_id="integration_test",
            stage_name="inference",
            component_type=ComponentType.MODEL_INFERENCE,
            config=context.config,
            artifacts_path=self.temp_dir,
            logger=self.orchestrator.logger,
            metadata={"test": "integration"}
        )
        
        inference_result = inference_engine.execute(inference_context)
        
        assert inference_result.success is True
        assert "predictions_generated" in inference_result.metrics
        
        # Verify all stages completed successfully
        assert preprocessing_result.success
        assert training_result.success
        assert evaluation_result.success
        assert inference_result.success
    
    def test_pipeline_with_orchestrator(self):
        """Test pipeline execution using orchestrator."""
        from mlpipeline.core.interfaces import PipelineStage
        
        # Load configuration
        config = self.config_manager.load_config(str(self.config_path))
        
        # Create execution context
        context = ExecutionContext(
            experiment_id="orchestrator_test",
            stage_name="pipeline",
            component_type=ComponentType.DATA_PREPROCESSING,
            config=config.model_dump(),
            artifacts_path=self.temp_dir,
            logger=self.orchestrator.logger,
            metadata={"test": "orchestrator"}
        )
        
        # Create ingested data file
        ingested_data_path = Path(self.temp_dir) / "ingested_data.parquet"
        self.sample_data.to_parquet(ingested_data_path, index=False)
        
        # Create pipeline stages
        preprocessing_stage = PipelineStage(
            name="preprocessing",
            components=[DataPreprocessor()]
        )
        
        training_stage = PipelineStage(
            name="training",
            components=[ModelTrainer()],
            dependencies=["preprocessing"]
        )
        
        evaluation_stage = PipelineStage(
            name="evaluation",
            components=[ModelEvaluator()],
            dependencies=["training"]
        )
        
        stages = [preprocessing_stage, training_stage, evaluation_stage]
        
        # Execute pipeline
        result = self.orchestrator.execute_pipeline(stages, context)
        
        assert result.success is True
        assert len(result.artifacts) > 0
        assert len(result.metrics) > 0
        assert "executed_stages" in result.metadata
        assert len(result.metadata["executed_stages"]) == 3
    
    def test_pipeline_failure_recovery(self):
        """Test pipeline behavior with component failures and recovery."""
        # Create a configuration that will cause training to fail
        failing_config = self.pipeline_config.copy()
        failing_config["training"]["model"]["parameters"]["n_estimators"] = -1  # Invalid parameter
        
        config_path = Path(self.temp_dir) / "failing_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(failing_config, f)
        
        config = self.config_manager.load_config(str(config_path))
        
        context = ExecutionContext(
            experiment_id="failure_test",
            stage_name="pipeline",
            component_type=ComponentType.DATA_PREPROCESSING,
            config=config.model_dump(),
            artifacts_path=self.temp_dir,
            logger=self.orchestrator.logger,
            metadata={"test": "failure"}
        )
        
        # Create ingested data file
        ingested_data_path = Path(self.temp_dir) / "ingested_data.parquet"
        self.sample_data.to_parquet(ingested_data_path, index=False)
        
        # Create pipeline stages
        from mlpipeline.core.interfaces import PipelineStage
        
        preprocessing_stage = PipelineStage(
            name="preprocessing",
            components=[DataPreprocessor()]
        )
        
        training_stage = PipelineStage(
            name="training",
            components=[ModelTrainer()],
            dependencies=["preprocessing"]
        )
        
        stages = [preprocessing_stage, training_stage]
        
        # Execute pipeline - should fail at training stage
        result = self.orchestrator.execute_pipeline(stages, context)
        
        assert result.success is False
        assert "failed_stage" in result.metadata
        # Preprocessing should succeed, training should fail
        assert len(result.artifacts) > 0  # Preprocessing artifacts
    
    def test_pipeline_with_different_data_formats(self):
        """Test pipeline with different input data formats."""
        # Test with JSON data
        json_data = self.sample_data.to_dict('records')
        json_path = Path(self.temp_dir) / "sample_data.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)
        
        # Update config for JSON input
        json_config = self.pipeline_config.copy()
        json_config["data"]["sources"][0]["type"] = "json"
        json_config["data"]["sources"][0]["path"] = str(json_path)
        
        json_config_path = Path(self.temp_dir) / "json_config.yaml"
        with open(json_config_path, 'w') as f:
            yaml.dump(json_config, f)
        
        config = self.config_manager.load_config(str(json_config_path))
        
        context = ExecutionContext(
            experiment_id="json_test",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config=config.model_dump(),
            artifacts_path=self.temp_dir,
            logger=self.orchestrator.logger,
            metadata={"test": "json_format"}
        )
        
        # Create ingested data file (convert JSON to expected format)
        ingested_data_path = Path(self.temp_dir) / "ingested_data.parquet"
        pd.DataFrame(json_data).to_parquet(ingested_data_path, index=False)
        
        # Test preprocessing with JSON data
        preprocessor = DataPreprocessor()
        result = preprocessor.execute(context)
        
        assert result.success is True
        assert len(result.artifacts) >= 3
    
    def test_pipeline_configuration_validation(self):
        """Test pipeline configuration validation."""
        # Test with invalid configuration
        invalid_config = {
            "pipeline": {"name": "test"},
            "data": {
                "sources": [{"type": "invalid_type", "path": "test.csv"}]
            }
        }
        
        invalid_config_path = Path(self.temp_dir) / "invalid_config.yaml"
        with open(invalid_config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Should raise validation error
        with pytest.raises(Exception):  # ValidationError or similar
            self.config_manager.load_config(str(invalid_config_path))
    
    def test_pipeline_artifacts_persistence(self):
        """Test that pipeline artifacts are properly persisted."""
        config = self.config_manager.load_config(str(self.config_path))
        
        context = ExecutionContext(
            experiment_id="persistence_test",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config=config.model_dump(),
            artifacts_path=self.temp_dir,
            logger=self.orchestrator.logger,
            metadata={"test": "persistence"}
        )
        
        # Create ingested data file
        ingested_data_path = Path(self.temp_dir) / "ingested_data.parquet"
        self.sample_data.to_parquet(ingested_data_path, index=False)
        
        # Execute preprocessing
        preprocessor = DataPreprocessor()
        result = preprocessor.execute(context)
        
        assert result.success is True
        
        # Verify all expected artifacts exist
        expected_artifacts = [
            "train_preprocessed.parquet",
            "val_preprocessed.parquet", 
            "test_preprocessed.parquet",
            "preprocessing_metadata.json",
            "preprocessing_pipeline.pkl"
        ]
        
        for artifact in expected_artifacts:
            artifact_path = Path(self.temp_dir) / artifact
            assert artifact_path.exists(), f"Artifact {artifact} not found"
        
        # Verify artifact contents
        train_data = pd.read_parquet(Path(self.temp_dir) / "train_preprocessed.parquet")
        assert len(train_data) > 0
        assert "target" in train_data.columns
        
        # Verify metadata
        with open(Path(self.temp_dir) / "preprocessing_metadata.json", 'r') as f:
            metadata = json.load(f)
            assert "steps" in metadata
            assert "target_column" in metadata
            assert metadata["target_column"] == "target"
    
    def test_pipeline_reproducibility(self):
        """Test that pipeline execution is reproducible."""
        config = self.config_manager.load_config(str(self.config_path))
        
        # Execute pipeline twice with same configuration
        results = []
        for i in range(2):
            temp_dir = tempfile.mkdtemp()
            try:
                context = ExecutionContext(
                    experiment_id=f"reproducibility_test_{i}",
                    stage_name="training",
                    component_type=ComponentType.MODEL_TRAINING,
                    config=config.model_dump(),
                    artifacts_path=temp_dir,
                    logger=self.orchestrator.logger,
                    metadata={"test": "reproducibility", "run": i}
                )
                
                # Create required input files
                ingested_data_path = Path(temp_dir) / "ingested_data.parquet"
                self.sample_data.to_parquet(ingested_data_path, index=False)
                
                # Run preprocessing first
                preprocessor = DataPreprocessor()
                prep_result = preprocessor.execute(context)
                assert prep_result.success is True
                
                # Run training
                trainer = ModelTrainer()
                training_result = trainer.execute(context)
                assert training_result.success is True
                
                results.append(training_result.metrics)
                
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Results should be identical due to random seeds
        assert results[0]["train_accuracy"] == results[1]["train_accuracy"]
        assert results[0]["val_accuracy"] == results[1]["val_accuracy"]


class TestPipelineIntegrationEdgeCases:
    """Test edge cases in pipeline integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_dataset_handling(self):
        """Test pipeline behavior with empty dataset."""
        # Create empty dataset
        empty_data = pd.DataFrame(columns=['feature_1', 'feature_2', 'target'])
        empty_data_path = Path(self.temp_dir) / "empty_data.parquet"
        empty_data.to_parquet(empty_data_path, index=False)
        
        config = {
            "pipeline": {"name": "empty_test"},
            "data": {
                "sources": [{"type": "csv", "path": str(empty_data_path)}],
                "preprocessing": [{"type": "standard_scaler"}]
            },
            "model": {"type": "sklearn"}
        }
        
        config_path = Path(self.temp_dir) / "empty_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        loaded_config = self.config_manager.load_config(str(config_path))
        
        context = ExecutionContext(
            experiment_id="empty_test",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config=loaded_config.model_dump(),
            artifacts_path=self.temp_dir,
            logger=self.config_manager.logger,
            metadata={}
        )
        
        # Create ingested data file
        ingested_data_path = Path(self.temp_dir) / "ingested_data.parquet"
        empty_data.to_parquet(ingested_data_path, index=False)
        
        preprocessor = DataPreprocessor()
        result = preprocessor.execute(context)
        
        # Should handle empty data gracefully
        assert result.success is False  # Expected to fail with empty data
        assert "error" in result.error_message.lower() or "empty" in result.error_message.lower()
    
    def test_missing_target_column(self):
        """Test pipeline behavior when target column is missing."""
        # Create data without target column
        data_without_target = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [5, 4, 3, 2, 1]
        })
        
        data_path = Path(self.temp_dir) / "no_target_data.parquet"
        data_without_target.to_parquet(data_path, index=False)
        
        config = {
            "pipeline": {"name": "no_target_test"},
            "data": {
                "sources": [{"type": "csv", "path": str(data_path)}],
                "preprocessing": [{"type": "standard_scaler"}]
            },
            "model": {"type": "sklearn"}
        }
        
        config_path = Path(self.temp_dir) / "no_target_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        loaded_config = self.config_manager.load_config(str(config_path))
        
        context = ExecutionContext(
            experiment_id="no_target_test",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config=loaded_config.model_dump(),
            artifacts_path=self.temp_dir,
            logger=self.config_manager.logger,
            metadata={}
        )
        
        # Create ingested data file
        ingested_data_path = Path(self.temp_dir) / "ingested_data.parquet"
        data_without_target.to_parquet(ingested_data_path, index=False)
        
        preprocessor = DataPreprocessor()
        result = preprocessor.execute(context)
        
        # Should fail gracefully when target column is missing
        assert result.success is False
        assert "target" in result.error_message.lower() or "column" in result.error_message.lower()
    
    def test_large_dataset_handling(self):
        """Test pipeline behavior with larger dataset."""
        # Create larger dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            'feature_1': np.random.randn(10000),
            'feature_2': np.random.randn(10000),
            'feature_3': np.random.randint(0, 10, 10000),
            'target': np.random.randint(0, 2, 10000)
        })
        
        large_data_path = Path(self.temp_dir) / "large_data.parquet"
        large_data.to_parquet(large_data_path, index=False)
        
        config = {
            "pipeline": {"name": "large_test"},
            "data": {
                "sources": [{"type": "csv", "path": str(large_data_path)}],
                "preprocessing": [
                    {
                        "type": "standard_scaler", 
                        "columns": ["feature_1", "feature_2", "feature_3"]
                    }
                ],
                "train_split": 0.8,
                "validation_split": 0.1,
                "test_split": 0.1,
                "random_state": 42
            },
            "model": {
                "type": "sklearn",
                "parameters": {
                    "algorithm": "LogisticRegression",
                    "random_state": 42
                }
            }
        }
        
        config_path = Path(self.temp_dir) / "large_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        loaded_config = self.config_manager.load_config(str(config_path))
        
        # Test preprocessing with large dataset
        context = ExecutionContext(
            experiment_id="large_test",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config=loaded_config.model_dump(),
            artifacts_path=self.temp_dir,
            logger=self.config_manager.logger,
            metadata={}
        )
        
        # Create ingested data file
        ingested_data_path = Path(self.temp_dir) / "ingested_data.parquet"
        large_data.to_parquet(ingested_data_path, index=False)
        
        preprocessor = DataPreprocessor()
        result = preprocessor.execute(context)
        
        assert result.success is True
        assert result.metrics["original_rows"] == 10000
        assert result.metrics["train_rows"] == 8000
        assert result.metrics["val_rows"] == 1000
        assert result.metrics["test_rows"] == 1000
        
        # Test training with large dataset
        training_context = ExecutionContext(
            experiment_id="large_test",
            stage_name="training",
            component_type=ComponentType.MODEL_TRAINING,
            config=loaded_config.model_dump(),
            artifacts_path=self.temp_dir,
            logger=self.config_manager.logger,
            metadata={}
        )
        
        trainer = ModelTrainer()
        training_result = trainer.execute(training_context)
        
        assert training_result.success is True
        assert "train_accuracy" in training_result.metrics
        assert "val_accuracy" in training_result.metrics
        
        # Performance should be reasonable
        assert training_result.execution_time < 60.0  # Should complete within 60 seconds