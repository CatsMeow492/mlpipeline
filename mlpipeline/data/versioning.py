"""Data versioning integration with DVC for tracking data lineage and versions."""

import logging
import json
import os
import subprocess
import hashlib
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from ..core.interfaces import PipelineComponent, ExecutionContext, ExecutionResult, ComponentType
from ..core.errors import DataError, ConfigurationError


@dataclass
class DataVersion:
    """Represents a data version with metadata."""
    version_id: str
    file_path: str
    file_hash: str
    size_bytes: int
    created_at: str
    metadata: Dict[str, Any]
    parent_versions: List[str] = None
    
    def __post_init__(self):
        if self.parent_versions is None:
            self.parent_versions = []


@dataclass
class DataLineage:
    """Tracks data lineage and transformations."""
    source_versions: List[str]
    transformation_type: str
    transformation_config: Dict[str, Any]
    output_version: str
    created_at: str
    execution_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.execution_context is None:
            self.execution_context = {}


class DVCManager:
    """Manages DVC operations for data versioning."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._dvc_initialized = False
    
    def initialize_dvc(self) -> bool:
        """Initialize DVC repository if not already initialized."""
        try:
            # Check if DVC is already initialized
            if (self.repo_path / ".dvc").exists():
                self._dvc_initialized = True
                self.logger.info("DVC repository already initialized")
                return True
            
            # Initialize DVC
            result = subprocess.run(
                ["dvc", "init"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            self._dvc_initialized = True
            self.logger.info("DVC repository initialized successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to initialize DVC: {e.stderr}")
            return False
        except FileNotFoundError:
            self.logger.error("DVC command not found. Please install DVC first.")
            return False
    
    def add_data_file(self, file_path: str, remote: Optional[str] = None) -> bool:
        """Add a data file to DVC tracking."""
        try:
            if not self._dvc_initialized and not self.initialize_dvc():
                return False
            
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return False
            
            # Add file to DVC
            cmd = ["dvc", "add", str(file_path)]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"Added file to DVC tracking: {file_path}")
            
            # Push to remote if specified
            if remote:
                self.push_to_remote(str(file_path), remote)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to add file to DVC: {e.stderr}")
            return False
    
    def push_to_remote(self, file_path: str, remote: str) -> bool:
        """Push data file to DVC remote storage."""
        try:
            cmd = ["dvc", "push", file_path, "--remote", remote]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"Pushed file to remote '{remote}': {file_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to push to remote: {e.stderr}")
            return False
    
    def pull_from_remote(self, file_path: str, remote: Optional[str] = None) -> bool:
        """Pull data file from DVC remote storage."""
        try:
            cmd = ["dvc", "pull", file_path]
            if remote:
                cmd.extend(["--remote", remote])
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"Pulled file from remote: {file_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to pull from remote: {e.stderr}")
            return False
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get DVC file information."""
        try:
            # DVC file is the original file path + ".dvc"
            dvc_file = Path(str(file_path) + ".dvc")
            
            if not dvc_file.exists():
                return None
            
            with open(dvc_file, 'r') as f:
                import yaml
                dvc_info = yaml.safe_load(f)
            
            return dvc_info
            
        except Exception as e:
            self.logger.error(f"Failed to get file info: {str(e)}")
            return None
    
    def create_pipeline_stage(self, stage_name: str, command: str, 
                            dependencies: List[str], outputs: List[str]) -> bool:
        """Create a DVC pipeline stage."""
        try:
            cmd = [
                "dvc", "stage", "add",
                "--name", stage_name,
                "--command", command
            ]
            
            # Add dependencies
            for dep in dependencies:
                cmd.extend(["--deps", dep])
            
            # Add outputs
            for output in outputs:
                cmd.extend(["--outs", output])
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"Created DVC pipeline stage: {stage_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create pipeline stage: {e.stderr}")
            return False


class DataVersionManager(PipelineComponent):
    """Manages data versioning and lineage tracking."""
    
    def __init__(self, repo_path: str = "."):
        super().__init__(ComponentType.DATA_PREPROCESSING)  # Reuse existing type
        self.dvc_manager = DVCManager(repo_path)
        self.versions_file = Path(repo_path) / ".mlpipeline" / "data_versions.json"
        self.lineage_file = Path(repo_path) / ".mlpipeline" / "data_lineage.json"
        self.versions: Dict[str, DataVersion] = {}
        self.lineage: List[DataLineage] = []
        
        # Ensure metadata directory exists
        self.versions_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing versions and lineage
        self._load_versions()
        self._load_lineage()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _generate_version_id(self, file_path: str, file_hash: str) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.now().isoformat()
        content = f"{file_path}_{file_hash}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _load_versions(self) -> None:
        """Load existing data versions from file."""
        try:
            if self.versions_file.exists():
                with open(self.versions_file, 'r') as f:
                    versions_data = json.load(f)
                    self.versions = {
                        k: DataVersion(**v) for k, v in versions_data.items()
                    }
                self.logger.info(f"Loaded {len(self.versions)} data versions")
        except Exception as e:
            self.logger.error(f"Failed to load versions: {str(e)}")
            self.versions = {}
    
    def _save_versions(self) -> None:
        """Save data versions to file."""
        try:
            versions_data = {
                k: asdict(v) for k, v in self.versions.items()
            }
            with open(self.versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
            self.logger.debug("Saved data versions")
        except Exception as e:
            self.logger.error(f"Failed to save versions: {str(e)}")
    
    def _load_lineage(self) -> None:
        """Load existing data lineage from file."""
        try:
            if self.lineage_file.exists():
                with open(self.lineage_file, 'r') as f:
                    lineage_data = json.load(f)
                    self.lineage = [DataLineage(**item) for item in lineage_data]
                self.logger.info(f"Loaded {len(self.lineage)} lineage records")
        except Exception as e:
            self.logger.error(f"Failed to load lineage: {str(e)}")
            self.lineage = []
    
    def _save_lineage(self) -> None:
        """Save data lineage to file."""
        try:
            lineage_data = [asdict(item) for item in self.lineage]
            with open(self.lineage_file, 'w') as f:
                json.dump(lineage_data, f, indent=2, default=str)
            self.logger.debug("Saved data lineage")
        except Exception as e:
            self.logger.error(f"Failed to save lineage: {str(e)}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate data versioning configuration."""
        try:
            versioning_config = config.get('versioning', {})
            
            # Check if DVC is enabled
            if not versioning_config.get('enabled', True):
                self.logger.info("Data versioning is disabled")
                return True
            
            # Validate remote configuration if specified
            remote_config = versioning_config.get('remote')
            if remote_config:
                if 'name' not in remote_config:
                    self.logger.error("Remote name is required")
                    return False
                
                if 'url' not in remote_config:
                    self.logger.error("Remote URL is required")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def setup(self, context: ExecutionContext) -> None:
        """Setup data versioning system."""
        config = context.config.get('data', {})
        
        if not self.validate_config(config):
            raise ConfigurationError("Invalid data versioning configuration")
        
        versioning_config = config.get('versioning', {})
        
        if not versioning_config.get('enabled', True):
            self.logger.info("Data versioning is disabled, skipping setup")
            return
        
        # Initialize DVC if needed
        if not self.dvc_manager.initialize_dvc():
            raise ConfigurationError("Failed to initialize DVC")
        
        # Configure remote if specified
        remote_config = versioning_config.get('remote')
        if remote_config:
            self._configure_remote(remote_config)
        
        self.logger.info("Data versioning system setup complete")
    
    def _configure_remote(self, remote_config: Dict[str, Any]) -> None:
        """Configure DVC remote storage."""
        try:
            remote_name = remote_config['name']
            remote_url = remote_config['url']
            
            # Add remote
            cmd = ["dvc", "remote", "add", remote_name, remote_url]
            if remote_config.get('default', False):
                cmd.append("--default")
            
            subprocess.run(
                cmd,
                cwd=self.dvc_manager.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Configure authentication if provided
            auth_config = remote_config.get('auth', {})
            for key, value in auth_config.items():
                subprocess.run([
                    "dvc", "remote", "modify", remote_name, key, value
                ], cwd=self.dvc_manager.repo_path, check=True)
            
            self.logger.info(f"Configured DVC remote: {remote_name}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to configure remote: {e}")
            raise ConfigurationError(f"Failed to configure DVC remote: {e}")
    
    def version_data(self, file_path: str, metadata: Optional[Dict[str, Any]] = None,
                    parent_versions: Optional[List[str]] = None) -> str:
        """Create a new version of a data file."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise DataError(f"File does not exist: {file_path}")
            
            # Calculate file hash and size
            file_hash = self._calculate_file_hash(str(file_path))
            file_size = file_path.stat().st_size
            
            # Generate version ID
            version_id = self._generate_version_id(str(file_path), file_hash)
            
            # Check if this version already exists
            existing_version = self._find_version_by_hash(file_hash)
            if existing_version:
                self.logger.info(f"File already versioned: {existing_version.version_id}")
                return existing_version.version_id
            
            # Create data version
            data_version = DataVersion(
                version_id=version_id,
                file_path=str(file_path),
                file_hash=file_hash,
                size_bytes=file_size,
                created_at=datetime.now().isoformat(),
                metadata=metadata or {},
                parent_versions=parent_versions or []
            )
            
            # Add to DVC tracking
            if not self.dvc_manager.add_data_file(str(file_path)):
                self.logger.warning(f"Failed to add file to DVC: {file_path}")
            
            # Store version
            self.versions[version_id] = data_version
            self._save_versions()
            
            self.logger.info(f"Created data version: {version_id} for {file_path}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to version data: {str(e)}")
            raise DataError(f"Failed to version data: {str(e)}")
    
    def _find_version_by_hash(self, file_hash: str) -> Optional[DataVersion]:
        """Find a data version by file hash."""
        for version in self.versions.values():
            if version.file_hash == file_hash:
                return version
        return None
    
    def track_lineage(self, source_versions: List[str], transformation_type: str,
                     transformation_config: Dict[str, Any], output_version: str,
                     execution_context: Optional[Dict[str, Any]] = None) -> None:
        """Track data lineage for a transformation."""
        try:
            lineage = DataLineage(
                source_versions=source_versions,
                transformation_type=transformation_type,
                transformation_config=transformation_config,
                output_version=output_version,
                created_at=datetime.now().isoformat(),
                execution_context=execution_context or {}
            )
            
            self.lineage.append(lineage)
            self._save_lineage()
            
            self.logger.info(f"Tracked lineage: {transformation_type} -> {output_version}")
            
        except Exception as e:
            self.logger.error(f"Failed to track lineage: {str(e)}")
    
    def get_version_info(self, version_id: str) -> Optional[DataVersion]:
        """Get information about a data version."""
        return self.versions.get(version_id)
    
    def get_lineage_for_version(self, version_id: str) -> List[DataLineage]:
        """Get lineage records for a specific version."""
        lineage_records = []
        
        # Find records where this version is an output
        for record in self.lineage:
            if record.output_version == version_id:
                lineage_records.append(record)
        
        # Find records where this version is a source
        for record in self.lineage:
            if version_id in record.source_versions:
                lineage_records.append(record)
        
        return lineage_records
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute data versioning for files in the context."""
        try:
            versioning_config = context.config.get('data', {}).get('versioning', {})
            
            if not versioning_config.get('enabled', True):
                self.logger.info("Data versioning is disabled")
                return ExecutionResult(
                    success=True,
                    artifacts=[],
                    metrics={},
                    metadata={'versioning_enabled': False}
                )
            
            # Get files to version from context metadata
            files_to_version = context.metadata.get('files_to_version', [])
            if not files_to_version:
                # Look for artifacts from previous stages
                artifacts_path = Path(context.artifacts_path)
                files_to_version = list(artifacts_path.glob("*.parquet"))
                files_to_version.extend(artifacts_path.glob("*.csv"))
                files_to_version.extend(artifacts_path.glob("*.json"))
            
            versioned_files = []
            version_ids = []
            
            for file_path in files_to_version:
                try:
                    # Create metadata for this file
                    file_metadata = {
                        'stage': context.stage_name,
                        'experiment_id': context.experiment_id,
                        'component_type': str(context.component_type),
                        'created_by': 'mlpipeline'
                    }
                    
                    # Version the file
                    version_id = self.version_data(
                        str(file_path),
                        metadata=file_metadata
                    )
                    
                    versioned_files.append(str(file_path))
                    version_ids.append(version_id)
                    
                except Exception as e:
                    self.logger.error(f"Failed to version file {file_path}: {str(e)}")
            
            # Calculate metrics
            metrics = {
                'files_versioned': len(versioned_files),
                'total_versions': len(self.versions),
                'total_lineage_records': len(self.lineage)
            }
            
            self.logger.info(f"Versioned {len(versioned_files)} files")
            
            return ExecutionResult(
                success=True,
                artifacts=versioned_files,
                metrics=metrics,
                metadata={
                    'version_ids': version_ids,
                    'versioning_enabled': True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Data versioning failed: {str(e)}")
            return ExecutionResult(
                success=False,
                artifacts=[],
                metrics={},
                metadata={},
                error_message=str(e)
            )
    
    def cleanup(self, context: ExecutionContext) -> None:
        """Cleanup versioning resources."""
        self.logger.info("Data versioning cleanup completed")


class DataVersioningIntegrator:
    """Integrates data versioning with preprocessing pipeline."""
    
    def __init__(self, version_manager: DataVersionManager):
        self.version_manager = version_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def version_preprocessing_outputs(self, preprocessing_result: ExecutionResult,
                                   preprocessing_config: Dict[str, Any],
                                   source_version_ids: Optional[List[str]] = None) -> List[str]:
        """Version the outputs of data preprocessing."""
        try:
            output_version_ids = []
            
            # Version each output artifact
            for artifact_path in preprocessing_result.artifacts:
                if Path(artifact_path).suffix in ['.parquet', '.csv', '.json', '.pkl']:
                    # Create metadata for preprocessing output
                    metadata = {
                        'transformation_type': 'preprocessing',
                        'preprocessing_steps': len(preprocessing_config.get('steps', [])),
                        'original_shape': preprocessing_result.metadata.get('original_shape'),
                        'transformed_shape': preprocessing_result.metadata.get('transformed_shape'),
                        'feature_names': preprocessing_result.metadata.get('feature_names', [])
                    }
                    
                    # Version the file
                    version_id = self.version_manager.version_data(
                        artifact_path,
                        metadata=metadata,
                        parent_versions=source_version_ids
                    )
                    
                    output_version_ids.append(version_id)
            
            # Track lineage if we have source versions
            if source_version_ids and output_version_ids:
                for output_version_id in output_version_ids:
                    self.version_manager.track_lineage(
                        source_versions=source_version_ids,
                        transformation_type='preprocessing',
                        transformation_config=preprocessing_config,
                        output_version=output_version_id,
                        execution_context={
                            'success': preprocessing_result.success,
                            'metrics': preprocessing_result.metrics
                        }
                    )
            
            self.logger.info(f"Versioned {len(output_version_ids)} preprocessing outputs")
            return output_version_ids
            
        except Exception as e:
            self.logger.error(f"Failed to version preprocessing outputs: {str(e)}")
            return []