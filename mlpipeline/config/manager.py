"""Configuration manager with validation and loading capabilities."""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import re

from pydantic import ValidationError as PydanticValidationError
from .schema import PipelineConfig, ValidationResult, ValidationError


class ConfigManager:
    """Manages pipeline configuration loading, validation, and variable substitution."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config_cache: Dict[str, PipelineConfig] = {}
    
    def load_config(self, config_path: str, validate: bool = True) -> PipelineConfig:
        """
        Load and validate configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            validate: Whether to validate the configuration
            
        Returns:
            PipelineConfig: Validated configuration object
            
        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Check cache first
        cache_key = str(config_path.absolute())
        if cache_key in self._config_cache:
            self.logger.debug(f"Using cached configuration for {config_path}")
            return self._config_cache[cache_key]
        
        try:
            # Load raw configuration
            raw_config = self._load_raw_config(config_path)
            
            # Resolve environment variables
            resolved_config = self.resolve_variables(raw_config)
            
            if validate:
                # Validate configuration
                validation_result = self.validate_schema(resolved_config)
                if not validation_result.valid:
                    raise ValidationError(
                        f"Configuration validation failed: {'; '.join(validation_result.errors)}",
                        validation_result.errors
                    )
                config = validation_result.config
            else:
                # Create config without validation
                config = PipelineConfig(**resolved_config)
            
            # Cache the configuration
            self._config_cache[cache_key] = config
            
            self.logger.info(f"Successfully loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise
    
    def _load_raw_config(self, config_path: Path) -> Dict[str, Any]:
        """Load raw configuration from file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML syntax: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON syntax: {str(e)}")
    
    def validate_schema(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration against schema.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            ValidationResult: Validation result with errors and warnings
        """
        errors = []
        warnings = []
        
        try:
            # Attempt to create PipelineConfig
            pipeline_config = PipelineConfig(**config)
            
            # Additional custom validations
            custom_warnings = self._perform_custom_validations(pipeline_config)
            warnings.extend(custom_warnings)
            
            return ValidationResult(
                valid=True,
                errors=errors,
                warnings=warnings,
                config=pipeline_config
            )
            
        except PydanticValidationError as e:
            # Extract detailed error messages
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error['loc'])
                error_msg = f"{field_path}: {error['msg']}"
                errors.append(error_msg)
            
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                config=None
            )
        except Exception as e:
            errors.append(f"Unexpected validation error: {str(e)}")
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                config=None
            )
    
    def _perform_custom_validations(self, config: PipelineConfig) -> List[str]:
        """Perform additional custom validations and return warnings."""
        warnings = []
        
        # Check for common configuration issues
        if config.model.type.value == "pytorch" and not config.model.parameters.get("device"):
            warnings.append("PyTorch model specified but no device configured. Will use CPU by default.")
        
        if config.drift_detection.enabled and len(config.drift_detection.alert_channels) == 0:
            warnings.append("Drift detection enabled but no alert channels configured.")
        
        if config.few_shot.enabled and config.model.type.value not in ["huggingface"]:
            warnings.append("Few-shot learning works best with Hugging Face models.")
        
        # Check for performance considerations
        if config.model.hyperparameter_tuning and config.model.hyperparameter_tuning.n_trials > 100:
            warnings.append("Large number of hyperparameter trials may take significant time.")
        
        return warnings
    
    def resolve_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve environment variables and other substitutions in configuration.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Dict[str, Any]: Configuration with resolved variables
        """
        def resolve_value(value):
            if isinstance(value, str):
                return self._substitute_variables(value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value
        
        return resolve_value(config)
    
    def _substitute_variables(self, value: str) -> str:
        """
        Substitute environment variables and other placeholders in string values.
        
        Supports:
        - ${VAR_NAME} or ${VAR_NAME:default_value}
        - $VAR_NAME
        """
        # Pattern for ${VAR_NAME} or ${VAR_NAME:default}
        pattern1 = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')
        
        def replace_match(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_value)
        
        # Replace ${VAR} patterns
        result = pattern1.sub(replace_match, value)
        
        # Pattern for $VAR_NAME (simple form)
        pattern2 = re.compile(r'\$([A-Za-z_][A-Za-z0-9_]*)')
        
        def replace_simple(match):
            var_name = match.group(1)
            return os.environ.get(var_name, f"${var_name}")  # Keep original if not found
        
        # Replace $VAR patterns
        result = pattern2.sub(replace_simple, result)
        
        return result
    
    def save_config(self, config: PipelineConfig, output_path: str, format: str = "yaml") -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            output_path: Output file path
            format: Output format ('yaml' or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.model_dump(mode='json')  # Use JSON mode for proper serialization
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {output_path}: {str(e)}")
            raise
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._config_cache.clear()
        self.logger.debug("Configuration cache cleared")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get a default configuration template."""
        return {
            "pipeline": {
                "name": "example_experiment",
                "description": "Example ML pipeline experiment",
                "tags": ["example", "demo"]
            },
            "data": {
                "sources": [
                    {
                        "type": "csv",
                        "path": "data/train.csv"
                    }
                ],
                "preprocessing": [
                    {
                        "type": "standard_scaler",
                        "columns": ["feature1", "feature2"]
                    }
                ]
            },
            "model": {
                "type": "sklearn",
                "parameters": {
                    "algorithm": "RandomForestClassifier",
                    "n_estimators": 100,
                    "random_state": 42
                }
            },
            "evaluation": {
                "metrics": ["accuracy", "f1_score", "precision", "recall"]
            }
        }