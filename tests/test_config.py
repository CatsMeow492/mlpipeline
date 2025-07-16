"""Tests for configuration management and validation."""

import pytest
import tempfile
import os
import yaml
import json
from pathlib import Path
from unittest.mock import patch

from mlpipeline.config import ConfigManager, PipelineConfig, ValidationError
from mlpipeline.config.schema import (
    DataSourceType, ModelType, HyperparameterMethod, DriftMethod
)


class TestConfigSchema:
    """Test configuration schema validation."""
    
    def test_valid_minimal_config(self):
        """Test minimal valid configuration."""
        config_data = {
            "pipeline": {
                "name": "test_experiment"
            },
            "data": {
                "sources": [
                    {
                        "type": "csv",
                        "path": "data/test.csv"
                    }
                ]
            },
            "model": {
                "type": "sklearn",
                "parameters": {
                    "algorithm": "RandomForestClassifier"
                }
            }
        }
        
        config = PipelineConfig(**config_data)
        assert config.pipeline.name == "test_experiment"
        assert len(config.data.sources) == 1
        assert config.data.sources[0].type == DataSourceType.CSV
        assert config.model.type == ModelType.SKLEARN
    
    def test_data_splits_validation(self):
        """Test data split validation."""
        config_data = {
            "pipeline": {"name": "test"},
            "data": {
                "sources": [{"type": "csv", "path": "test.csv"}],
                "train_split": 0.6,
                "validation_split": 0.2,
                "test_split": 0.3  # This should fail (sum > 1.0)
            },
            "model": {"type": "sklearn"}
        }
        
        with pytest.raises(ValueError, match="Data splits must sum to 1.0"):
            PipelineConfig(**config_data)
    
    def test_sql_source_validation(self):
        """Test SQL data source validation."""
        config_data = {
            "pipeline": {"name": "test"},
            "data": {
                "sources": [
                    {
                        "type": "sql",
                        "path": "dummy",
                        # Missing connection_string - should fail
                    }
                ]
            },
            "model": {"type": "sklearn"}
        }
        
        with pytest.raises(ValueError, match="connection_string is required"):
            PipelineConfig(**config_data)
    
    def test_hyperparameter_config(self):
        """Test hyperparameter optimization configuration."""
        config_data = {
            "pipeline": {"name": "test"},
            "data": {"sources": [{"type": "csv", "path": "test.csv"}]},
            "model": {
                "type": "xgboost",
                "hyperparameter_tuning": {
                    "method": "optuna",
                    "n_trials": 100,
                    "parameters": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 6, 9]
                    }
                }
            }
        }
        
        config = PipelineConfig(**config_data)
        assert config.model.hyperparameter_tuning.method == HyperparameterMethod.OPTUNA
        assert config.model.hyperparameter_tuning.n_trials == 100
    
    def test_drift_detection_config(self):
        """Test drift detection configuration."""
        config_data = {
            "pipeline": {"name": "test"},
            "data": {"sources": [{"type": "csv", "path": "test.csv"}]},
            "model": {"type": "sklearn"},
            "drift_detection": {
                "enabled": True,
                "baseline_data": "data/baseline.csv",
                "methods": ["evidently", "kl_divergence"],
                "thresholds": {
                    "data_drift": 0.1,
                    "prediction_drift": 0.05
                }
            }
        }
        
        config = PipelineConfig(**config_data)
        assert config.drift_detection.enabled is True
        assert DriftMethod.EVIDENTLY in config.drift_detection.methods
        assert config.drift_detection.thresholds.data_drift == 0.1
    
    def test_few_shot_config_validation(self):
        """Test few-shot learning configuration validation."""
        config_data = {
            "pipeline": {"name": "test"},
            "data": {"sources": [{"type": "csv", "path": "test.csv"}]},
            "model": {"type": "sklearn"},
            "few_shot": {
                "enabled": True
                # Missing prompt_template - should fail
            }
        }
        
        with pytest.raises(ValueError, match="prompt_template is required"):
            PipelineConfig(**config_data)
    
    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        config_data = {
            "pipeline": {"name": "test"},
            "data": {"sources": [{"type": "csv", "path": "test.csv"}]},
            "model": {"type": "sklearn"},
            "unknown_field": "should_fail"  # Extra field
        }
        
        with pytest.raises(Exception, match="Extra inputs are not permitted"):
            PipelineConfig(**config_data)


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration."""
        config_data = {
            "pipeline": {"name": "yaml_test"},
            "data": {"sources": [{"type": "csv", "path": "test.csv"}]},
            "model": {"type": "sklearn"}
        }
        
        config_path = Path(self.temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = self.config_manager.load_config(str(config_path))
        assert config.pipeline.name == "yaml_test"
    
    def test_load_json_config(self):
        """Test loading JSON configuration."""
        config_data = {
            "pipeline": {"name": "json_test"},
            "data": {"sources": [{"type": "csv", "path": "test.csv"}]},
            "model": {"type": "sklearn"}
        }
        
        config_path = Path(self.temp_dir) / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = self.config_manager.load_config(str(config_path))
        assert config.pipeline.name == "json_test"
    
    def test_environment_variable_substitution(self):
        """Test environment variable substitution."""
        config_data = {
            "pipeline": {"name": "${EXPERIMENT_NAME:default_experiment}"},
            "data": {"sources": [{"type": "csv", "path": "$DATA_PATH"}]},
            "model": {"type": "sklearn"}
        }
        
        config_path = Path(self.temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with patch.dict(os.environ, {"EXPERIMENT_NAME": "env_test", "DATA_PATH": "/data/test.csv"}):
            config = self.config_manager.load_config(str(config_path))
            assert config.pipeline.name == "env_test"
            assert config.data.sources[0].path == "/data/test.csv"
    
    def test_environment_variable_defaults(self):
        """Test environment variable default values."""
        config_data = {
            "pipeline": {"name": "${MISSING_VAR:default_name}"},
            "data": {"sources": [{"type": "csv", "path": "test.csv"}]},
            "model": {"type": "sklearn"}
        }
        
        config_path = Path(self.temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = self.config_manager.load_config(str(config_path))
        assert config.pipeline.name == "default_name"
    
    def test_validation_errors(self):
        """Test configuration validation error handling."""
        config_data = {
            "pipeline": {"name": "test"},
            "data": {
                "sources": [{"type": "csv", "path": "test.csv"}],
                "train_split": 1.5  # Invalid value
            },
            "model": {"type": "sklearn"}
        }
        
        config_path = Path(self.temp_dir) / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValidationError):
            self.config_manager.load_config(str(config_path))
    
    def test_file_not_found(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            self.config_manager.load_config("nonexistent_config.yaml")
    
    def test_invalid_yaml_syntax(self):
        """Test handling of invalid YAML syntax."""
        config_path = Path(self.temp_dir) / "invalid.yaml"
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: syntax: [")
        
        with pytest.raises(ValidationError, match="Invalid YAML syntax"):
            self.config_manager.load_config(str(config_path))
    
    def test_config_caching(self):
        """Test configuration caching."""
        config_data = {
            "pipeline": {"name": "cache_test"},
            "data": {"sources": [{"type": "csv", "path": "test.csv"}]},
            "model": {"type": "sklearn"}
        }
        
        config_path = Path(self.temp_dir) / "cache_test.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config twice
        config1 = self.config_manager.load_config(str(config_path))
        config2 = self.config_manager.load_config(str(config_path))
        
        # Should be the same object (cached)
        assert config1 is config2
    
    def test_save_config_yaml(self):
        """Test saving configuration to YAML."""
        config_data = {
            "pipeline": {"name": "save_test"},
            "data": {"sources": [{"type": "csv", "path": "test.csv"}]},
            "model": {"type": "sklearn"}
        }
        
        config = PipelineConfig(**config_data)
        output_path = Path(self.temp_dir) / "saved_config.yaml"
        
        self.config_manager.save_config(config, str(output_path), "yaml")
        
        # Verify file was created and can be loaded
        assert output_path.exists()
        loaded_config = self.config_manager.load_config(str(output_path))
        assert loaded_config.pipeline.name == "save_test"
    
    def test_get_default_config(self):
        """Test getting default configuration template."""
        default_config = self.config_manager.get_default_config()
        
        assert "pipeline" in default_config
        assert "data" in default_config
        assert "model" in default_config
        assert default_config["pipeline"]["name"] == "example_experiment"
    
    def test_custom_validations_warnings(self):
        """Test custom validation warnings."""
        config_data = {
            "pipeline": {"name": "warning_test"},
            "data": {"sources": [{"type": "csv", "path": "test.csv"}]},
            "model": {"type": "pytorch"},  # Should trigger warning about device
            "drift_detection": {"enabled": True, "baseline_data": "baseline.csv"}  # No alert channels
        }
        
        validation_result = self.config_manager.validate_schema(config_data)
        
        assert validation_result.valid is True
        assert len(validation_result.warnings) > 0
        assert any("device" in warning for warning in validation_result.warnings)
        assert any("alert channels" in warning for warning in validation_result.warnings)