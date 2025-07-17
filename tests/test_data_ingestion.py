"""Tests for data ingestion engine and connectors."""

import pytest
import tempfile
import json
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mlpipeline.data.ingestion import (
    DataIngestionEngine, CSVConnector, JSONConnector, ParquetConnector, SQLConnector,
    DataSourceConfig, DataSchema, DataValidator
)
from mlpipeline.core.interfaces import ExecutionContext, ComponentType
from mlpipeline.core.errors import DataError, ConfigurationError


class TestDataSourceConfig:
    """Test data source configuration."""
    
    def test_basic_config(self):
        """Test basic configuration creation."""
        config = DataSourceConfig(
            source_type="csv",
            path="data/test.csv"
        )
        
        assert config.source_type == "csv"
        assert config.path == "data/test.csv"
        assert config.options == {}
    
    def test_full_config(self):
        """Test full configuration with all options."""
        config = DataSourceConfig(
            source_type="sql",
            path="dummy",
            schema_path="schema.json",
            connection_string="sqlite:///test.db",
            table_name="users",
            query="SELECT * FROM users",
            options={"chunksize": 1000}
        )
        
        assert config.source_type == "sql"
        assert config.connection_string == "sqlite:///test.db"
        assert config.table_name == "users"
        assert config.options["chunksize"] == 1000


class TestDataSchema:
    """Test data schema functionality."""
    
    def test_basic_schema(self):
        """Test basic schema creation."""
        schema = DataSchema(
            columns={"id": "int", "name": "string"},
            required_columns=["id", "name"]
        )
        
        assert schema.columns["id"] == "int"
        assert "id" in schema.required_columns
        assert schema.nullable_columns == []
    
    def test_full_schema(self):
        """Test schema with all options."""
        schema = DataSchema(
            columns={"id": "int", "name": "string", "age": "int"},
            required_columns=["id", "name"],
            nullable_columns=["age"],
            constraints={
                "age_range": {
                    "type": "range",
                    "column": "age",
                    "min": 0,
                    "max": 120
                }
            }
        )
        
        assert len(schema.columns) == 3
        assert "age" in schema.nullable_columns
        assert "age_range" in schema.constraints


class TestCSVConnector:
    """Test CSV connector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "test.csv"
        
        # Create test CSV file
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        test_data.to_csv(self.csv_path, index=False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_csv_connector_creation(self):
        """Test CSV connector creation."""
        config = DataSourceConfig("csv", str(self.csv_path))
        connector = CSVConnector(config)
        
        assert connector.config.source_type == "csv"
        assert connector.config.path == str(self.csv_path)
    
    def test_csv_connection_validation(self):
        """Test CSV connection validation."""
        config = DataSourceConfig("csv", str(self.csv_path))
        connector = CSVConnector(config)
        
        # Valid file
        assert connector.validate_connection() is True
        
        # Invalid file
        config_invalid = DataSourceConfig("csv", "nonexistent.csv")
        connector_invalid = CSVConnector(config_invalid)
        assert connector_invalid.validate_connection() is False
    
    def test_csv_connect(self):
        """Test CSV connection."""
        config = DataSourceConfig("csv", str(self.csv_path))
        connector = CSVConnector(config)
        
        # Should not raise exception
        connector.connect()
        
        # Test with non-existent file
        config_invalid = DataSourceConfig("csv", "nonexistent.csv")
        connector_invalid = CSVConnector(config_invalid)
        
        with pytest.raises(DataError, match="CSV file not found"):
            connector_invalid.connect()
    
    def test_csv_load_data(self):
        """Test CSV data loading."""
        config = DataSourceConfig("csv", str(self.csv_path))
        connector = CSVConnector(config)
        
        df = connector.load_data()
        
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name', 'age']
        assert df.iloc[0]['name'] == 'Alice'
    
    def test_csv_load_data_with_options(self):
        """Test CSV data loading with options."""
        # Create CSV with custom separator
        custom_csv_path = Path(self.temp_dir) / "custom.csv"
        with open(custom_csv_path, 'w') as f:
            f.write("id;name;age\n1;Alice;25\n2;Bob;30\n")
        
        config = DataSourceConfig(
            "csv", 
            str(custom_csv_path),
            options={"separator": ";", "nrows": 1}
        )
        connector = CSVConnector(config)
        
        df = connector.load_data()
        
        assert len(df) == 1  # Only loaded 1 row due to nrows option
        assert df.iloc[0]['name'] == 'Alice'


class TestJSONConnector:
    """Test JSON connector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.json_path = Path(self.temp_dir) / "test.json"
        
        # Create test JSON file
        test_data = [
            {'id': 1, 'name': 'Alice', 'age': 25},
            {'id': 2, 'name': 'Bob', 'age': 30},
            {'id': 3, 'name': 'Charlie', 'age': 35}
        ]
        
        with open(self.json_path, 'w') as f:
            json.dump(test_data, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_json_connector_creation(self):
        """Test JSON connector creation."""
        config = DataSourceConfig("json", str(self.json_path))
        connector = JSONConnector(config)
        
        assert connector.config.source_type == "json"
        assert connector.config.path == str(self.json_path)
    
    def test_json_load_data(self):
        """Test JSON data loading."""
        config = DataSourceConfig("json", str(self.json_path))
        connector = JSONConnector(config)
        
        df = connector.load_data()
        
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name', 'age']
        assert df.iloc[0]['name'] == 'Alice'
    
    def test_json_lines_format(self):
        """Test JSON Lines format loading."""
        # Create JSON Lines file
        jsonl_path = Path(self.temp_dir) / "test.jsonl"
        with open(jsonl_path, 'w') as f:
            f.write('{"id": 1, "name": "Alice"}\n')
            f.write('{"id": 2, "name": "Bob"}\n')
        
        config = DataSourceConfig(
            "json",
            str(jsonl_path),
            options={"lines": True}
        )
        connector = JSONConnector(config)
        
        df = connector.load_data()
        
        assert len(df) == 2
        assert df.iloc[0]['name'] == 'Alice'


class TestParquetConnector:
    """Test Parquet connector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = Path(self.temp_dir) / "test.parquet"
        
        # Create test Parquet file
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        test_data.to_parquet(self.parquet_path, index=False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parquet_connector_creation(self):
        """Test Parquet connector creation."""
        config = DataSourceConfig("parquet", str(self.parquet_path))
        connector = ParquetConnector(config)
        
        assert connector.config.source_type == "parquet"
        assert connector.config.path == str(self.parquet_path)
    
    def test_parquet_load_data(self):
        """Test Parquet data loading."""
        config = DataSourceConfig("parquet", str(self.parquet_path))
        connector = ParquetConnector(config)
        
        df = connector.load_data()
        
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name', 'age']
        assert df.iloc[0]['name'] == 'Alice'
    
    def test_parquet_load_specific_columns(self):
        """Test Parquet data loading with column selection."""
        config = DataSourceConfig(
            "parquet",
            str(self.parquet_path),
            options={"columns": ["id", "name"]}
        )
        connector = ParquetConnector(config)
        
        df = connector.load_data()
        
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name']
        assert 'age' not in df.columns


class TestSQLConnector:
    """Test SQL connector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.connection_string = f"sqlite:///{self.db_path}"
        
        # Create test database
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER
            )
        ''')
        
        cursor.execute("INSERT INTO users (name, age) VALUES ('Alice', 25)")
        cursor.execute("INSERT INTO users (name, age) VALUES ('Bob', 30)")
        cursor.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)")
        
        conn.commit()
        conn.close()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sql_connector_creation(self):
        """Test SQL connector creation."""
        config = DataSourceConfig(
            "sql",
            "dummy",
            connection_string=self.connection_string,
            table_name="users"
        )
        connector = SQLConnector(config)
        
        assert connector.config.source_type == "sql"
        assert connector.config.connection_string == self.connection_string
        assert connector.config.table_name == "users"
    
    def test_sql_connector_missing_connection_string(self):
        """Test SQL connector without connection string."""
        config = DataSourceConfig("sql", "dummy")
        connector = SQLConnector(config)
        
        with pytest.raises(ConfigurationError, match="Connection string is required"):
            connector.connect()
    
    def test_sql_connect_sqlalchemy_missing(self):
        """Test SQL connector when SQLAlchemy is not available."""
        config = DataSourceConfig(
            "sql",
            "dummy",
            connection_string=self.connection_string
        )
        connector = SQLConnector(config)
        
        # Mock the import to fail
        with patch('builtins.__import__', side_effect=ImportError("No module named 'sqlalchemy'")):
            with pytest.raises(ConfigurationError, match="SQLAlchemy is required"):
                connector.connect()
    
    def test_sql_load_data_with_table(self):
        """Test SQL data loading with table name."""
        pytest.importorskip("sqlalchemy")
        
        config = DataSourceConfig(
            "sql",
            "dummy",
            connection_string=self.connection_string,
            table_name="users"
        )
        connector = SQLConnector(config)
        
        df = connector.load_data()
        
        assert len(df) == 3
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert df.iloc[0]['name'] == 'Alice'
    
    def test_sql_load_data_with_query(self):
        """Test SQL data loading with custom query."""
        pytest.importorskip("sqlalchemy")
        
        config = DataSourceConfig(
            "sql",
            "dummy",
            connection_string=self.connection_string,
            query="SELECT name, age FROM users WHERE age > 25"
        )
        connector = SQLConnector(config)
        
        df = connector.load_data()
        
        assert len(df) == 2  # Only Bob and Charlie
        assert list(df.columns) == ['name', 'age']
        assert 'Alice' not in df['name'].values
    
    def test_sql_missing_query_and_table(self):
        """Test SQL connector without query or table name."""
        pytest.importorskip("sqlalchemy")
        
        config = DataSourceConfig(
            "sql",
            "dummy",
            connection_string=self.connection_string
        )
        connector = SQLConnector(config)
        
        with pytest.raises(DataError, match="Either 'query' or 'table_name' must be specified"):
            connector.load_data()


class TestDataValidator:
    """Test data validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'email': ['alice@test.com', 'bob@test.com', None]
        })
    
    def test_validate_required_columns(self):
        """Test validation of required columns."""
        schema = DataSchema(
            columns={'id': 'int', 'name': 'string', 'missing': 'string'},
            required_columns=['id', 'name', 'missing']
        )
        
        errors = self.validator.validate_schema(self.test_df, schema)
        
        assert len(errors) == 1
        assert "Missing required columns: ['missing']" in errors[0]
    
    def test_validate_data_types(self):
        """Test validation of data types."""
        schema = DataSchema(
            columns={'id': 'string', 'name': 'int'},  # Wrong types
            required_columns=['id', 'name']
        )
        
        errors = self.validator.validate_schema(self.test_df, schema)
        
        assert len(errors) >= 2
        assert any("Column 'id' has type" in error for error in errors)
        assert any("Column 'name' has type" in error for error in errors)
    
    def test_validate_nullable_constraints(self):
        """Test validation of nullable constraints."""
        schema = DataSchema(
            columns={'id': 'int', 'name': 'string', 'email': 'string'},
            required_columns=['id', 'name', 'email'],
            nullable_columns=['email']  # Only email can be null
        )
        
        # Add null value to name column
        df_with_nulls = self.test_df.copy()
        df_with_nulls.loc[0, 'name'] = None
        
        errors = self.validator.validate_schema(df_with_nulls, schema)
        
        assert len(errors) == 1
        assert "Column 'name' has 1 null values but is not nullable" in errors[0]
    
    def test_validate_range_constraint(self):
        """Test validation of range constraints."""
        schema = DataSchema(
            columns={'age': 'int'},
            required_columns=['age'],
            constraints={
                'age_range': {
                    'type': 'range',
                    'column': 'age',
                    'min': 18,
                    'max': 65
                }
            }
        )
        
        # Add invalid age
        df_with_invalid = self.test_df.copy()
        df_with_invalid.loc[0, 'age'] = 15  # Below minimum
        
        errors = self.validator.validate_schema(df_with_invalid, schema)
        
        assert len(errors) == 1
        assert "Column 'age' has values below minimum 18" in errors[0]
    
    def test_validate_unique_constraint(self):
        """Test validation of unique constraints."""
        schema = DataSchema(
            columns={'id': 'int'},
            required_columns=['id'],
            constraints={
                'id_unique': {
                    'type': 'unique',
                    'column': 'id'
                }
            }
        )
        
        # Add duplicate ID
        df_with_duplicates = self.test_df.copy()
        df_with_duplicates.loc[0, 'id'] = 2  # Duplicate of row 1
        
        errors = self.validator.validate_schema(df_with_duplicates, schema)
        
        assert len(errors) == 1
        assert "Column 'id' has 1 duplicate values but should be unique" in errors[0]
    
    def test_validate_regex_constraint(self):
        """Test validation of regex constraints."""
        schema = DataSchema(
            columns={'email': 'string'},
            required_columns=[],
            nullable_columns=['email'],  # Allow nulls for email
            constraints={
                'email_format': {
                    'type': 'regex',
                    'column': 'email',
                    'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                }
            }
        )
        
        # Add invalid email
        df_with_invalid = self.test_df.copy()
        df_with_invalid.loc[0, 'email'] = 'invalid-email'
        
        errors = self.validator.validate_schema(df_with_invalid, schema)
        
        assert len(errors) == 1
        assert "don't match pattern" in errors[0]


class TestDataIngestionEngine:
    """Test data ingestion engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = DataIngestionEngine()
        
        # Create test CSV file
        self.csv_path = Path(self.temp_dir) / "test.csv"
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        test_data.to_csv(self.csv_path, index=False)
        
        # Create execution context
        self.context = ExecutionContext(
            experiment_id="test_exp",
            stage_name="data_ingestion",
            component_type=ComponentType.DATA_INGESTION,
            config={
                'data': {
                    'sources': [
                        {
                            'type': 'csv',
                            'path': str(self.csv_path)
                        }
                    ]
                }
            },
            artifacts_path=self.temp_dir,
            logger=Mock(),
            metadata={}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_engine_creation(self):
        """Test data ingestion engine creation."""
        assert self.engine.component_type == ComponentType.DATA_INGESTION
        assert 'csv' in self.engine.connectors
        assert 'json' in self.engine.connectors
        assert 'parquet' in self.engine.connectors
        assert 'sql' in self.engine.connectors
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = {
            'data': {
                'sources': [
                    {
                        'type': 'csv',
                        'path': 'test.csv'
                    }
                ]
            }
        }
        
        assert self.engine.validate_config(config) is True
    
    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config."""
        # Missing sources
        config1 = {'data': {}}
        assert self.engine.validate_config(config1) is False
        
        # Missing type
        config2 = {
            'data': {
                'sources': [{'path': 'test.csv'}]
            }
        }
        assert self.engine.validate_config(config2) is False
        
        # Invalid type
        config3 = {
            'data': {
                'sources': [{'type': 'invalid', 'path': 'test.csv'}]
            }
        }
        assert self.engine.validate_config(config3) is False
    
    def test_execute_single_source(self):
        """Test execution with single data source."""
        result = self.engine.execute(self.context)
        
        assert result.success is True
        assert len(result.artifacts) >= 1
        assert result.metrics['total_rows'] == 3
        assert result.metrics['total_columns'] == 3
        assert result.metrics['data_sources_count'] == 1
        
        # Check that output file was created
        output_path = Path(self.temp_dir) / "ingested_data.parquet"
        assert output_path.exists()
    
    def test_execute_multiple_sources(self):
        """Test execution with multiple data sources."""
        # Create second CSV file
        csv2_path = Path(self.temp_dir) / "test2.csv"
        test_data2 = pd.DataFrame({
            'id': [4, 5],
            'name': ['David', 'Eve'],
            'age': [40, 45]
        })
        test_data2.to_csv(csv2_path, index=False)
        
        # Update context with multiple sources
        self.context.config['data']['sources'].append({
            'type': 'csv',
            'path': str(csv2_path)
        })
        
        result = self.engine.execute(self.context)
        
        assert result.success is True
        assert result.metrics['total_rows'] == 5  # 3 + 2
        assert result.metrics['data_sources_count'] == 2
    
    def test_execute_with_schema_validation(self):
        """Test execution with schema validation."""
        # Create schema file with a constraint that will fail
        schema_path = Path(self.temp_dir) / "schema.json"
        schema_data = {
            'columns': {'id': 'int', 'name': 'string', 'age': 'int'},
            'required_columns': ['id', 'name', 'age'],
            'nullable_columns': [],
            'constraints': {
                'age_range': {
                    'type': 'range',
                    'column': 'age',
                    'min': 50,  # This will cause validation errors since our test data has ages 25, 30, 35
                    'max': 100
                }
            }
        }
        
        with open(schema_path, 'w') as f:
            json.dump(schema_data, f)
        
        # Update context with schema
        self.context.config['data']['sources'][0]['schema_path'] = str(schema_path)
        
        result = self.engine.execute(self.context)
        
        assert result.success is True
        # Should have validation report in artifacts due to validation errors
        assert any('validation_report' in artifact for artifact in result.artifacts)
    
    def test_execute_no_sources_configured(self):
        """Test execution with no data sources configured."""
        self.context.config['data']['sources'] = []
        
        result = self.engine.execute(self.context)
        
        assert result.success is False
        assert "No data sources configured" in result.error_message
    
    def test_create_connector(self):
        """Test connector creation."""
        config = DataSourceConfig("csv", "test.csv")
        connector = self.engine._create_connector(config)
        
        assert isinstance(connector, CSVConnector)
        
        # Test unsupported type
        config_invalid = DataSourceConfig("unsupported", "test.file")
        with pytest.raises(ConfigurationError, match="Unsupported data source type"):
            self.engine._create_connector(config_invalid)