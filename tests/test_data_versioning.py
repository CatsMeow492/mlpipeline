"""Tests for data versioning and DVC integration."""

import pytest
import tempfile
import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from mlpipeline.data.versioning import (
    DataVersion, DataLineage, DVCManager, DataVersionManager, DataVersioningIntegrator
)
from mlpipeline.core.interfaces import ExecutionContext, ExecutionResult, ComponentType
from mlpipeline.core.errors import DataError, ConfigurationError


class TestDataVersion:
    """Test data version dataclass."""
    
    def test_basic_data_version(self):
        """Test basic data version creation."""
        version = DataVersion(
            version_id="test_v1",
            file_path="/path/to/data.csv",
            file_hash="abc123",
            size_bytes=1024,
            created_at="2023-01-01T00:00:00",
            metadata={"type": "training_data"}
        )
        
        assert version.version_id == "test_v1"
        assert version.file_path == "/path/to/data.csv"
        assert version.file_hash == "abc123"
        assert version.size_bytes == 1024
        assert version.metadata["type"] == "training_data"
        assert version.parent_versions == []
    
    def test_data_version_with_parents(self):
        """Test data version with parent versions."""
        version = DataVersion(
            version_id="test_v2",
            file_path="/path/to/processed.csv",
            file_hash="def456",
            size_bytes=2048,
            created_at="2023-01-02T00:00:00",
            metadata={"type": "processed_data"},
            parent_versions=["test_v1"]
        )
        
        assert version.parent_versions == ["test_v1"]


class TestDataLineage:
    """Test data lineage dataclass."""
    
    def test_basic_lineage(self):
        """Test basic lineage creation."""
        lineage = DataLineage(
            source_versions=["v1", "v2"],
            transformation_type="preprocessing",
            transformation_config={"scaler": "standard"},
            output_version="v3",
            created_at="2023-01-01T00:00:00"
        )
        
        assert lineage.source_versions == ["v1", "v2"]
        assert lineage.transformation_type == "preprocessing"
        assert lineage.output_version == "v3"
        assert lineage.execution_context == {}
    
    def test_lineage_with_context(self):
        """Test lineage with execution context."""
        lineage = DataLineage(
            source_versions=["v1"],
            transformation_type="feature_engineering",
            transformation_config={"method": "pca"},
            output_version="v2",
            created_at="2023-01-01T00:00:00",
            execution_context={"success": True, "duration": 120}
        )
        
        assert lineage.execution_context["success"] is True
        assert lineage.execution_context["duration"] == 120


class TestDVCManager:
    """Test DVC manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dvc_manager = DVCManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_initialize_dvc_success(self, mock_run):
        """Test successful DVC initialization."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        result = self.dvc_manager.initialize_dvc()
        
        assert result is True
        assert self.dvc_manager._dvc_initialized is True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_initialize_dvc_already_initialized(self, mock_run):
        """Test DVC initialization when already initialized."""
        # Create .dvc directory to simulate existing initialization
        dvc_dir = Path(self.temp_dir) / ".dvc"
        dvc_dir.mkdir()
        
        result = self.dvc_manager.initialize_dvc()
        
        assert result is True
        assert self.dvc_manager._dvc_initialized is True
        mock_run.assert_not_called()
    
    @patch('subprocess.run')
    def test_initialize_dvc_failure(self, mock_run):
        """Test DVC initialization failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "dvc", stderr="Error")
        
        result = self.dvc_manager.initialize_dvc()
        
        assert result is False
        assert self.dvc_manager._dvc_initialized is False
    
    @patch('subprocess.run')
    def test_add_data_file_success(self, mock_run):
        """Test successful file addition to DVC."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test_data.csv"
        test_file.write_text("col1,col2\n1,2\n3,4")
        
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        self.dvc_manager._dvc_initialized = True
        
        result = self.dvc_manager.add_data_file(str(test_file))
        
        assert result is True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_add_data_file_nonexistent(self, mock_run):
        """Test adding nonexistent file to DVC."""
        self.dvc_manager._dvc_initialized = True
        
        result = self.dvc_manager.add_data_file("/nonexistent/file.csv")
        
        assert result is False
        mock_run.assert_not_called()
    
    @patch('subprocess.run')
    def test_push_to_remote_success(self, mock_run):
        """Test successful push to remote."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        result = self.dvc_manager.push_to_remote("data.csv", "myremote")
        
        assert result is True
        mock_run.assert_called_once_with(
            ["dvc", "push", "data.csv", "--remote", "myremote"],
            cwd=self.dvc_manager.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('subprocess.run')
    def test_pull_from_remote_success(self, mock_run):
        """Test successful pull from remote."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        result = self.dvc_manager.pull_from_remote("data.csv", "myremote")
        
        assert result is True
        mock_run.assert_called_once_with(
            ["dvc", "pull", "data.csv", "--remote", "myremote"],
            cwd=self.dvc_manager.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
    
    def test_get_file_info_success(self):
        """Test getting DVC file info."""
        # Create a mock DVC file in the temp directory
        data_file_path = Path(self.temp_dir) / "data.csv"
        dvc_file = Path(self.temp_dir) / "data.csv.dvc"
        dvc_content = {
            'outs': [{'md5': 'abc123', 'size': 1024, 'path': 'data.csv'}]
        }
        
        # Create the actual DVC file with YAML content
        import yaml
        with open(dvc_file, 'w') as f:
            yaml.dump(dvc_content, f)
        
        # Use the full path to the data file
        result = self.dvc_manager.get_file_info(str(data_file_path))
        
        assert result is not None
        assert result['outs'][0]['md5'] == 'abc123'
    
    def test_get_file_info_no_dvc_file(self):
        """Test getting info for file without DVC tracking."""
        result = self.dvc_manager.get_file_info("nonexistent.csv")
        assert result is None
    
    @patch('subprocess.run')
    def test_create_pipeline_stage_success(self, mock_run):
        """Test creating DVC pipeline stage."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        result = self.dvc_manager.create_pipeline_stage(
            stage_name="preprocess",
            command="python preprocess.py",
            dependencies=["raw_data.csv"],
            outputs=["processed_data.csv"]
        )
        
        assert result is True
        mock_run.assert_called_once()
        
        # Check the command structure
        call_args = mock_run.call_args[0][0]
        assert "dvc" in call_args
        assert "stage" in call_args
        assert "add" in call_args
        assert "--name" in call_args
        assert "preprocess" in call_args


class TestDataVersionManager:
    """Test data version manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.version_manager = DataVersionManager(self.temp_dir)
        
        # Create test data file
        self.test_file = Path(self.temp_dir) / "test_data.csv"
        self.test_file.write_text("col1,col2\n1,2\n3,4\n5,6")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        file_hash = self.version_manager._calculate_file_hash(str(self.test_file))
        
        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA256 hash length
        
        # Hash should be consistent
        file_hash2 = self.version_manager._calculate_file_hash(str(self.test_file))
        assert file_hash == file_hash2
    
    def test_generate_version_id(self):
        """Test version ID generation."""
        file_hash = "abc123"
        version_id = self.version_manager._generate_version_id(str(self.test_file), file_hash)
        
        assert isinstance(version_id, str)
        assert len(version_id) == 12  # MD5 hash truncated to 12 chars
        
        # Should be consistent for same inputs
        version_id2 = self.version_manager._generate_version_id(str(self.test_file), file_hash)
        assert version_id != version_id2  # Different due to timestamp
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = {
            'versioning': {
                'enabled': True,
                'remote': {
                    'name': 'myremote',
                    'url': 's3://mybucket/data'
                }
            }
        }
        
        assert self.version_manager.validate_config(config) is True
    
    def test_validate_config_disabled(self):
        """Test configuration validation when disabled."""
        config = {
            'versioning': {
                'enabled': False
            }
        }
        
        assert self.version_manager.validate_config(config) is True
    
    def test_validate_config_invalid_remote(self):
        """Test configuration validation with invalid remote."""
        config = {
            'versioning': {
                'enabled': True,
                'remote': {
                    'name': 'myremote'
                    # Missing URL
                }
            }
        }
        
        assert self.version_manager.validate_config(config) is False
    
    @patch.object(DVCManager, 'add_data_file', return_value=True)
    def test_version_data_success(self, mock_add_file):
        """Test successful data versioning."""
        metadata = {'type': 'training_data', 'source': 'manual'}
        
        version_id = self.version_manager.version_data(
            str(self.test_file),
            metadata=metadata
        )
        
        assert isinstance(version_id, str)
        assert len(version_id) == 12
        
        # Check that version was stored
        assert version_id in self.version_manager.versions
        version = self.version_manager.versions[version_id]
        assert version.file_path == str(self.test_file)
        assert version.metadata == metadata
        
        mock_add_file.assert_called_once_with(str(self.test_file))
    
    def test_version_data_nonexistent_file(self):
        """Test versioning nonexistent file."""
        with pytest.raises(DataError, match="File does not exist"):
            self.version_manager.version_data("/nonexistent/file.csv")
    
    @patch.object(DVCManager, 'add_data_file', return_value=True)
    def test_version_data_duplicate(self, mock_add_file):
        """Test versioning the same file twice."""
        # Version file first time
        version_id1 = self.version_manager.version_data(str(self.test_file))
        
        # Version same file again
        version_id2 = self.version_manager.version_data(str(self.test_file))
        
        # Should return the same version ID
        assert version_id1 == version_id2
        
        # Should only have one version stored
        assert len(self.version_manager.versions) == 1
    
    def test_track_lineage(self):
        """Test lineage tracking."""
        source_versions = ["v1", "v2"]
        transformation_type = "preprocessing"
        transformation_config = {"scaler": "standard"}
        output_version = "v3"
        execution_context = {"success": True}
        
        self.version_manager.track_lineage(
            source_versions=source_versions,
            transformation_type=transformation_type,
            transformation_config=transformation_config,
            output_version=output_version,
            execution_context=execution_context
        )
        
        assert len(self.version_manager.lineage) == 1
        lineage = self.version_manager.lineage[0]
        assert lineage.source_versions == source_versions
        assert lineage.transformation_type == transformation_type
        assert lineage.output_version == output_version
        assert lineage.execution_context == execution_context
    
    @patch.object(DVCManager, 'add_data_file', return_value=True)
    def test_get_version_info(self, mock_add_file):
        """Test getting version information."""
        version_id = self.version_manager.version_data(str(self.test_file))
        
        version_info = self.version_manager.get_version_info(version_id)
        
        assert version_info is not None
        assert version_info.version_id == version_id
        assert version_info.file_path == str(self.test_file)
    
    def test_get_version_info_nonexistent(self):
        """Test getting info for nonexistent version."""
        version_info = self.version_manager.get_version_info("nonexistent")
        assert version_info is None
    
    def test_get_lineage_for_version(self):
        """Test getting lineage for a specific version."""
        # Create some lineage records
        self.version_manager.track_lineage(
            source_versions=["v1"],
            transformation_type="preprocessing",
            transformation_config={},
            output_version="v2"
        )
        
        self.version_manager.track_lineage(
            source_versions=["v2"],
            transformation_type="feature_engineering",
            transformation_config={},
            output_version="v3"
        )
        
        # Get lineage for v2 (should appear as both source and output)
        lineage_records = self.version_manager.get_lineage_for_version("v2")
        
        assert len(lineage_records) == 2
        
        # Check that v2 appears as output in first record
        output_records = [r for r in lineage_records if r.output_version == "v2"]
        assert len(output_records) == 1
        
        # Check that v2 appears as source in second record
        source_records = [r for r in lineage_records if "v2" in r.source_versions]
        assert len(source_records) == 1
    
    @patch.object(DVCManager, 'initialize_dvc', return_value=True)
    def test_setup_success(self, mock_init_dvc):
        """Test successful setup."""
        context = ExecutionContext(
            experiment_id="test_exp",
            stage_name="versioning",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={
                'data': {
                    'versioning': {
                        'enabled': True
                    }
                }
            },
            artifacts_path=str(self.temp_dir),
            logger=Mock(),
            metadata={}
        )
        
        self.version_manager.setup(context)
        
        mock_init_dvc.assert_called_once()
    
    def test_setup_disabled(self):
        """Test setup when versioning is disabled."""
        context = ExecutionContext(
            experiment_id="test_exp",
            stage_name="versioning",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={
                'data': {
                    'versioning': {
                        'enabled': False
                    }
                }
            },
            artifacts_path=str(self.temp_dir),
            logger=Mock(),
            metadata={}
        )
        
        # Should not raise any errors
        self.version_manager.setup(context)
    
    @patch.object(DVCManager, 'initialize_dvc', return_value=False)
    def test_setup_dvc_failure(self, mock_init_dvc):
        """Test setup when DVC initialization fails."""
        context = ExecutionContext(
            experiment_id="test_exp",
            stage_name="versioning",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={
                'data': {
                    'versioning': {
                        'enabled': True
                    }
                }
            },
            artifacts_path=str(self.temp_dir),
            logger=Mock(),
            metadata={}
        )
        
        with pytest.raises(ConfigurationError, match="Failed to initialize DVC"):
            self.version_manager.setup(context)
    
    @patch.object(DataVersionManager, 'version_data')
    def test_execute_success(self, mock_version_data):
        """Test successful execution."""
        # Create test files
        test_file1 = Path(self.temp_dir) / "data1.parquet"
        test_file2 = Path(self.temp_dir) / "data2.csv"
        test_file1.touch()
        test_file2.touch()
        
        mock_version_data.side_effect = ["v1", "v2"]
        
        context = ExecutionContext(
            experiment_id="test_exp",
            stage_name="versioning",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={
                'data': {
                    'versioning': {
                        'enabled': True
                    }
                }
            },
            artifacts_path=str(self.temp_dir),
            logger=Mock(),
            metadata={}
        )
        
        result = self.version_manager.execute(context)
        
        assert result.success is True
        assert result.metrics['files_versioned'] == 2
        assert result.metadata['version_ids'] == ["v1", "v2"]
        assert len(result.artifacts) == 2
    
    def test_execute_disabled(self):
        """Test execution when versioning is disabled."""
        context = ExecutionContext(
            experiment_id="test_exp",
            stage_name="versioning",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={
                'data': {
                    'versioning': {
                        'enabled': False
                    }
                }
            },
            artifacts_path=str(self.temp_dir),
            logger=Mock(),
            metadata={}
        )
        
        result = self.version_manager.execute(context)
        
        assert result.success is True
        assert result.metadata['versioning_enabled'] is False
    
    def test_save_and_load_versions(self):
        """Test saving and loading versions."""
        # Create a version
        with patch.object(DVCManager, 'add_data_file', return_value=True):
            version_id = self.version_manager.version_data(str(self.test_file))
        
        # Create new manager instance (should load existing versions)
        new_manager = DataVersionManager(self.temp_dir)
        
        assert version_id in new_manager.versions
        assert new_manager.versions[version_id].file_path == str(self.test_file)
    
    def test_save_and_load_lineage(self):
        """Test saving and loading lineage."""
        # Create lineage
        self.version_manager.track_lineage(
            source_versions=["v1"],
            transformation_type="test",
            transformation_config={},
            output_version="v2"
        )
        
        # Create new manager instance (should load existing lineage)
        new_manager = DataVersionManager(self.temp_dir)
        
        assert len(new_manager.lineage) == 1
        assert new_manager.lineage[0].transformation_type == "test"


class TestDataVersioningIntegrator:
    """Test data versioning integrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.version_manager = DataVersionManager(self.temp_dir)
        self.integrator = DataVersioningIntegrator(self.version_manager)
        
        # Create test files
        self.test_file1 = Path(self.temp_dir) / "train_data.parquet"
        self.test_file2 = Path(self.temp_dir) / "metadata.json"
        self.test_file1.write_text("mock parquet data")
        self.test_file2.write_text('{"key": "value"}')
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.object(DataVersionManager, 'version_data')
    @patch.object(DataVersionManager, 'track_lineage')
    def test_version_preprocessing_outputs(self, mock_track_lineage, mock_version_data):
        """Test versioning preprocessing outputs."""
        mock_version_data.side_effect = ["v2", "v3"]
        
        # Create mock preprocessing result
        preprocessing_result = ExecutionResult(
            success=True,
            artifacts=[str(self.test_file1), str(self.test_file2)],
            metrics={'processed_rows': 1000},
            metadata={
                'original_shape': (1000, 10),
                'transformed_shape': (1000, 15),
                'feature_names': ['f1', 'f2', 'f3']
            }
        )
        
        preprocessing_config = {
            'steps': [
                {'name': 'scaler', 'transformer': 'standard_scaler'}
            ]
        }
        
        source_version_ids = ["v1"]
        
        output_version_ids = self.integrator.version_preprocessing_outputs(
            preprocessing_result=preprocessing_result,
            preprocessing_config=preprocessing_config,
            source_version_ids=source_version_ids
        )
        
        assert output_version_ids == ["v2", "v3"]
        
        # Check that version_data was called for each artifact
        assert mock_version_data.call_count == 2
        
        # Check that lineage was tracked for each output
        assert mock_track_lineage.call_count == 2
        
        # Verify lineage tracking calls
        for call in mock_track_lineage.call_args_list:
            args, kwargs = call
            assert kwargs['source_versions'] == source_version_ids
            assert kwargs['transformation_type'] == 'preprocessing'
            assert kwargs['transformation_config'] == preprocessing_config
    
    @patch.object(DataVersionManager, 'version_data')
    def test_version_preprocessing_outputs_no_sources(self, mock_version_data):
        """Test versioning preprocessing outputs without source versions."""
        mock_version_data.return_value = "v2"
        
        preprocessing_result = ExecutionResult(
            success=True,
            artifacts=[str(self.test_file1)],
            metrics={},
            metadata={}
        )
        
        preprocessing_config = {'steps': []}
        
        output_version_ids = self.integrator.version_preprocessing_outputs(
            preprocessing_result=preprocessing_result,
            preprocessing_config=preprocessing_config,
            source_version_ids=None
        )
        
        assert output_version_ids == ["v2"]
        
        # Should still version the data but not track lineage
        mock_version_data.assert_called_once()
    
    def test_version_preprocessing_outputs_no_artifacts(self):
        """Test versioning preprocessing outputs with no artifacts."""
        preprocessing_result = ExecutionResult(
            success=True,
            artifacts=[],
            metrics={},
            metadata={}
        )
        
        preprocessing_config = {'steps': []}
        
        output_version_ids = self.integrator.version_preprocessing_outputs(
            preprocessing_result=preprocessing_result,
            preprocessing_config=preprocessing_config
        )
        
        assert output_version_ids == []
    
    @patch.object(DataVersionManager, 'version_data')
    def test_version_preprocessing_outputs_filter_file_types(self, mock_version_data):
        """Test that only supported file types are versioned."""
        # Create files with different extensions
        txt_file = Path(self.temp_dir) / "readme.txt"
        txt_file.write_text("readme content")
        
        mock_version_data.return_value = "v2"
        
        preprocessing_result = ExecutionResult(
            success=True,
            artifacts=[str(self.test_file1), str(txt_file)],  # parquet and txt
            metrics={},
            metadata={}
        )
        
        preprocessing_config = {'steps': []}
        
        output_version_ids = self.integrator.version_preprocessing_outputs(
            preprocessing_result=preprocessing_result,
            preprocessing_config=preprocessing_config
        )
        
        # Should only version the parquet file, not the txt file
        assert output_version_ids == ["v2"]
        mock_version_data.assert_called_once()


# Helper function for mocking file operations
def mock_open(read_data=""):
    """Create a mock for the open function."""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open(read_data=read_data)