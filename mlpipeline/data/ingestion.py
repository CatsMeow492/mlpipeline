"""Data ingestion engine with support for multiple data sources."""

import logging
import json
from typing import Dict, Any, List, Optional, Union, Protocol
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
from dataclasses import dataclass

from ..core.interfaces import PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
from ..core.errors import DataError, ConfigurationError


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    source_type: str
    path: str
    schema_path: Optional[str] = None
    connection_string: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None
    options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.options is None:
            self.options = {}


@dataclass
class DataSchema:
    """Data schema definition."""
    columns: Dict[str, str]  # column_name -> data_type
    required_columns: List[str]
    nullable_columns: List[str] = None
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.nullable_columns is None:
            self.nullable_columns = []
        if self.constraints is None:
            self.constraints = {}


class DataSourceConnector(ABC):
    """Abstract base class for data source connectors."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from source."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection to data source."""
        pass
    
    def disconnect(self) -> None:
        """Disconnect from data source (optional override)."""
        pass
    
    def load_schema(self) -> Optional[DataSchema]:
        """Load schema definition if available."""
        if not self.config.schema_path:
            return None
        
        try:
            schema_path = Path(self.config.schema_path)
            if not schema_path.exists():
                self.logger.warning(f"Schema file not found: {schema_path}")
                return None
            
            with open(schema_path, 'r') as f:
                schema_data = json.load(f)
            
            return DataSchema(
                columns=schema_data.get('columns', {}),
                required_columns=schema_data.get('required_columns', []),
                nullable_columns=schema_data.get('nullable_columns', []),
                constraints=schema_data.get('constraints', {})
            )
        except Exception as e:
            self.logger.error(f"Failed to load schema: {str(e)}")
            return None


class CSVConnector(DataSourceConnector):
    """Connector for CSV files."""
    
    def connect(self) -> None:
        """Validate CSV file exists."""
        file_path = Path(self.config.path)
        if not file_path.exists():
            raise DataError(f"CSV file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.csv':
            raise DataError(f"File is not a CSV: {file_path}")
    
    def validate_connection(self) -> bool:
        """Validate CSV file accessibility."""
        try:
            file_path = Path(self.config.path)
            return file_path.exists() and file_path.is_file()
        except Exception:
            return False
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            # Extract pandas-specific options
            pandas_options = {
                'sep': self.config.options.get('separator', ','),
                'header': self.config.options.get('header', 0),
                'encoding': self.config.options.get('encoding', 'utf-8'),
                'na_values': self.config.options.get('na_values', None),
                'dtype': self.config.options.get('dtype', None),
                'parse_dates': self.config.options.get('parse_dates', None),
                'nrows': self.config.options.get('nrows', None),
                'skiprows': self.config.options.get('skiprows', None),
                'usecols': self.config.options.get('usecols', None)
            }
            
            # Remove None values
            pandas_options = {k: v for k, v in pandas_options.items() if v is not None}
            
            df = pd.read_csv(self.config.path, **pandas_options)
            self.logger.info(f"Loaded CSV data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            raise DataError(f"Failed to load CSV data: {str(e)}")


class JSONConnector(DataSourceConnector):
    """Connector for JSON files."""
    
    def connect(self) -> None:
        """Validate JSON file exists."""
        file_path = Path(self.config.path)
        if not file_path.exists():
            raise DataError(f"JSON file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.json':
            raise DataError(f"File is not a JSON: {file_path}")
    
    def validate_connection(self) -> bool:
        """Validate JSON file accessibility."""
        try:
            file_path = Path(self.config.path)
            return file_path.exists() and file_path.is_file()
        except Exception:
            return False
    
    def load_data(self) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            # Extract pandas-specific options
            pandas_options = {
                'orient': self.config.options.get('orient', 'records'),
                'lines': self.config.options.get('lines', False),
                'encoding': self.config.options.get('encoding', 'utf-8'),
                'dtype': self.config.options.get('dtype', None),
                'convert_dates': self.config.options.get('convert_dates', True),
                'keep_default_dates': self.config.options.get('keep_default_dates', True)
            }
            
            # Remove None values
            pandas_options = {k: v for k, v in pandas_options.items() if v is not None}
            
            df = pd.read_json(self.config.path, **pandas_options)
            self.logger.info(f"Loaded JSON data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            raise DataError(f"Failed to load JSON data: {str(e)}")


class ParquetConnector(DataSourceConnector):
    """Connector for Parquet files."""
    
    def connect(self) -> None:
        """Validate Parquet file exists."""
        file_path = Path(self.config.path)
        if not file_path.exists():
            raise DataError(f"Parquet file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.parquet':
            raise DataError(f"File is not a Parquet: {file_path}")
    
    def validate_connection(self) -> bool:
        """Validate Parquet file accessibility."""
        try:
            file_path = Path(self.config.path)
            return file_path.exists() and file_path.is_file()
        except Exception:
            return False
    
    def load_data(self) -> pd.DataFrame:
        """Load data from Parquet file."""
        try:
            # Extract pandas-specific options
            pandas_options = {
                'columns': self.config.options.get('columns', None),
                'filters': self.config.options.get('filters', None),
                'use_nullable_dtypes': self.config.options.get('use_nullable_dtypes', False)
            }
            
            # Remove None values
            pandas_options = {k: v for k, v in pandas_options.items() if v is not None}
            
            df = pd.read_parquet(self.config.path, **pandas_options)
            self.logger.info(f"Loaded Parquet data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            raise DataError(f"Failed to load Parquet data: {str(e)}")


class SQLConnector(DataSourceConnector):
    """Connector for SQL databases."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.connection = None
        self._engine = None
    
    def connect(self) -> None:
        """Establish database connection."""
        if not self.config.connection_string:
            raise ConfigurationError("Connection string is required for SQL connector")
        
        try:
            import sqlalchemy
            self._engine = sqlalchemy.create_engine(self.config.connection_string)
            self.connection = self._engine.connect()
            self.logger.info("Database connection established")
        except ImportError:
            raise ConfigurationError("SQLAlchemy is required for SQL connector. Install with: pip install sqlalchemy")
        except Exception as e:
            raise DataError(f"Failed to connect to database: {str(e)}")
    
    def validate_connection(self) -> bool:
        """Validate database connection."""
        try:
            if self.connection is None:
                self.connect()
            
            # Test connection with a simple query
            result = self.connection.execute(sqlalchemy.text("SELECT 1"))
            result.close()
            return True
        except Exception:
            return False
    
    def load_data(self) -> pd.DataFrame:
        """Load data from SQL database."""
        if self.connection is None:
            self.connect()
        
        try:
            # Determine query or table
            if self.config.query:
                query = self.config.query
            elif self.config.table_name:
                query = f"SELECT * FROM {self.config.table_name}"
            else:
                raise ConfigurationError("Either 'query' or 'table_name' must be specified for SQL connector")
            
            # Extract pandas-specific options
            pandas_options = {
                'chunksize': self.config.options.get('chunksize', None),
                'dtype': self.config.options.get('dtype', None),
                'parse_dates': self.config.options.get('parse_dates', None),
                'columns': self.config.options.get('columns', None),
                'coerce_float': self.config.options.get('coerce_float', True)
            }
            
            # Remove None values
            pandas_options = {k: v for k, v in pandas_options.items() if v is not None}
            
            df = pd.read_sql(query, self.connection, **pandas_options)
            self.logger.info(f"Loaded SQL data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            raise DataError(f"Failed to load SQL data: {str(e)}")
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("Database connection closed")


class DataValidator:
    """Validates data against schema definitions."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_schema(self, df: pd.DataFrame, schema: DataSchema) -> List[str]:
        """Validate DataFrame against schema and return list of validation errors."""
        errors = []
        
        # Check required columns
        missing_columns = set(schema.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")
        
        # Check column data types
        for column, expected_type in schema.columns.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if not self._is_compatible_type(actual_type, expected_type):
                    errors.append(f"Column '{column}' has type '{actual_type}', expected '{expected_type}'")
        
        # Check nullable constraints
        non_nullable_columns = set(schema.columns.keys()) - set(schema.nullable_columns)
        for column in non_nullable_columns:
            if column in df.columns and df[column].isnull().any():
                null_count = df[column].isnull().sum()
                errors.append(f"Column '{column}' has {null_count} null values but is not nullable")
        
        # Check custom constraints
        for constraint_name, constraint_config in schema.constraints.items():
            constraint_errors = self._validate_constraint(df, constraint_name, constraint_config)
            errors.extend(constraint_errors)
        
        return errors
    
    def _is_compatible_type(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual data type is compatible with expected type."""
        type_mappings = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32'],
            'string': ['object', 'string'],
            'bool': ['bool'],
            'datetime': ['datetime64[ns]', 'datetime64'],
            'category': ['category']
        }
        
        expected_variants = type_mappings.get(expected_type.lower(), [expected_type])
        return any(variant in actual_type.lower() for variant in expected_variants)
    
    def _validate_constraint(self, df: pd.DataFrame, constraint_name: str, constraint_config: Dict[str, Any]) -> List[str]:
        """Validate custom constraints."""
        errors = []
        
        try:
            constraint_type = constraint_config.get('type')
            column = constraint_config.get('column')
            
            if constraint_type == 'range':
                min_val = constraint_config.get('min')
                max_val = constraint_config.get('max')
                
                if column in df.columns:
                    if min_val is not None and (df[column] < min_val).any():
                        errors.append(f"Column '{column}' has values below minimum {min_val}")
                    if max_val is not None and (df[column] > max_val).any():
                        errors.append(f"Column '{column}' has values above maximum {max_val}")
            
            elif constraint_type == 'unique':
                if column in df.columns and df[column].duplicated().any():
                    duplicate_count = df[column].duplicated().sum()
                    errors.append(f"Column '{column}' has {duplicate_count} duplicate values but should be unique")
            
            elif constraint_type == 'regex':
                pattern = constraint_config.get('pattern')
                if column in df.columns and pattern:
                    import re
                    invalid_count = (~df[column].astype(str).str.match(pattern)).sum()
                    if invalid_count > 0:
                        errors.append(f"Column '{column}' has {invalid_count} values that don't match pattern '{pattern}'")
        
        except Exception as e:
            errors.append(f"Error validating constraint '{constraint_name}': {str(e)}")
        
        return errors


class DataIngestionEngine(PipelineComponent):
    """Main data ingestion engine that coordinates data loading and validation."""
    
    def __init__(self):
        super().__init__(ComponentType.DATA_INGESTION)
        self.connectors = {
            'csv': CSVConnector,
            'json': JSONConnector,
            'parquet': ParquetConnector,
            'sql': SQLConnector
        }
        self.validator = DataValidator()
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute data ingestion process."""
        try:
            # Get data source configurations
            data_sources = context.config.get('data', {}).get('sources', [])
            if not data_sources:
                raise ConfigurationError("No data sources configured")
            
            all_dataframes = []
            artifacts = []
            metrics = {}
            
            for i, source_config in enumerate(data_sources):
                source_name = f"source_{i}"
                self.logger.info(f"Processing data source {i+1}/{len(data_sources)}: {source_config.get('type')}")
                
                # Create data source configuration
                ds_config = DataSourceConfig(
                    source_type=source_config['type'],
                    path=source_config['path'],
                    schema_path=source_config.get('schema_path'),
                    connection_string=source_config.get('connection_string'),
                    table_name=source_config.get('table_name'),
                    query=source_config.get('query'),
                    options=source_config.get('options', {})
                )
                
                # Create and use connector
                connector = self._create_connector(ds_config)
                df, source_artifacts, source_metrics = self._load_and_validate_data(connector, source_name, context)
                
                all_dataframes.append(df)
                artifacts.extend(source_artifacts)
                metrics.update(source_metrics)
            
            # Combine dataframes if multiple sources
            if len(all_dataframes) == 1:
                final_df = all_dataframes[0]
            else:
                final_df = pd.concat(all_dataframes, ignore_index=True)
                self.logger.info(f"Combined {len(all_dataframes)} data sources into single DataFrame")
            
            # Save combined dataset
            output_path = Path(context.artifacts_path) / "ingested_data.parquet"
            final_df.to_parquet(output_path, index=False)
            artifacts.append(str(output_path))
            
            # Update metrics
            metrics.update({
                'total_rows': len(final_df),
                'total_columns': len(final_df.columns),
                'data_sources_count': len(data_sources),
                'memory_usage_mb': final_df.memory_usage(deep=True).sum() / 1024 / 1024
            })
            
            self.logger.info(f"Data ingestion completed: {metrics['total_rows']} rows, {metrics['total_columns']} columns")
            
            return ExecutionResult(
                success=True,
                artifacts=artifacts,
                metrics=metrics,
                metadata={
                    'data_shape': final_df.shape,
                    'column_names': list(final_df.columns),
                    'data_types': {col: str(dtype) for col, dtype in final_df.dtypes.items()}
                }
            )
            
        except Exception as e:
            self.logger.error(f"Data ingestion failed: {str(e)}")
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def _create_connector(self, config: DataSourceConfig) -> DataSourceConnector:
        """Create appropriate connector for data source type."""
        connector_class = self.connectors.get(config.source_type.lower())
        if not connector_class:
            raise ConfigurationError(f"Unsupported data source type: {config.source_type}")
        
        return connector_class(config)
    
    def _load_and_validate_data(self, connector: DataSourceConnector, source_name: str, 
                               context: ExecutionContext) -> tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """Load data using connector and validate against schema."""
        artifacts = []
        metrics = {}
        
        # Connect and validate
        connector.connect()
        if not connector.validate_connection():
            raise DataError(f"Failed to validate connection for {source_name}")
        
        # Load data
        df = connector.load_data()
        
        # Load and validate schema if available
        schema = connector.load_schema()
        if schema:
            validation_errors = self.validator.validate_schema(df, schema)
            if validation_errors:
                error_msg = f"Schema validation failed for {source_name}: {'; '.join(validation_errors)}"
                self.logger.warning(error_msg)
                
                # Save validation report
                validation_report_path = Path(context.artifacts_path) / f"{source_name}_validation_report.json"
                validation_report = {
                    'source_name': source_name,
                    'validation_errors': validation_errors,
                    'data_shape': df.shape,
                    'column_names': list(df.columns)
                }
                
                with open(validation_report_path, 'w') as f:
                    json.dump(validation_report, f, indent=2)
                artifacts.append(str(validation_report_path))
                
                # Decide whether to fail or continue based on configuration
                fail_on_validation = context.config.get('data', {}).get('fail_on_validation_error', False)
                if fail_on_validation:
                    raise DataError(error_msg)
            else:
                self.logger.info(f"Schema validation passed for {source_name}")
        
        # Disconnect
        connector.disconnect()
        
        # Calculate metrics
        metrics[f'{source_name}_rows'] = len(df)
        metrics[f'{source_name}_columns'] = len(df.columns)
        metrics[f'{source_name}_memory_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        return df, artifacts, metrics
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate component configuration."""
        try:
            data_config = config.get('data', {})
            sources = data_config.get('sources', [])
            
            if not sources:
                return False
            
            for source in sources:
                if 'type' not in source or 'path' not in source:
                    return False
                
                source_type = source['type'].lower()
                if source_type not in self.connectors:
                    return False
                
                # SQL-specific validation
                if source_type == 'sql':
                    if not source.get('connection_string'):
                        return False
                    if not source.get('query') and not source.get('table_name'):
                        return False
            
            return True
            
        except Exception:
            return False