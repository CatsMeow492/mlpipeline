"""Example demonstrating data versioning integration with preprocessing pipeline."""

import tempfile
import pandas as pd
from pathlib import Path

from mlpipeline.data import DataPreprocessor, DataVersionManager, DataVersioningIntegrator
from mlpipeline.core.interfaces import ExecutionContext, ComponentType


def create_sample_data():
    """Create sample data for demonstration."""
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'target': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    })
    return data


def main():
    """Demonstrate data versioning with preprocessing pipeline."""
    print("Data Versioning Integration Example")
    print("=" * 40)
    
    # Create temporary directory for the example
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Create sample data
        print("1. Creating sample data...")
        sample_data = create_sample_data()
        input_file = temp_path / "raw_data.parquet"
        sample_data.to_parquet(input_file, index=False)
        print(f"   Created: {input_file}")
        
        # 2. Initialize data version manager
        print("\n2. Initializing data version manager...")
        version_manager = DataVersionManager(str(temp_path))
        
        # Version the input data
        input_version_id = version_manager.version_data(
            str(input_file),
            metadata={
                'type': 'raw_data',
                'source': 'example_generation',
                'rows': len(sample_data),
                'columns': len(sample_data.columns)
            }
        )
        print(f"   Input data versioned: {input_version_id}")
        
        # 3. Set up preprocessing pipeline
        print("\n3. Setting up preprocessing pipeline...")
        preprocessor = DataPreprocessor()
        
        # Create execution context
        context = ExecutionContext(
            experiment_id="versioning_example",
            stage_name="preprocessing",
            component_type=ComponentType.DATA_PREPROCESSING,
            config={
                'data': {
                    'preprocessing': {
                        'steps': [
                            {
                                'name': 'scaler',
                                'transformer': 'standard_scaler',
                                'columns': ['feature1', 'feature2']
                            },
                            {
                                'name': 'encoder',
                                'transformer': 'one_hot_encoder',
                                'columns': ['category'],
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
                    },
                    'versioning': {
                        'enabled': True
                    }
                }
            },
            artifacts_path=str(temp_path),
            logger=None,
            metadata={}
        )
        
        # 4. Execute preprocessing
        print("\n4. Executing preprocessing pipeline...")
        preprocessing_result = preprocessor.execute(context)
        
        if preprocessing_result.success:
            print("   Preprocessing completed successfully!")
            print(f"   Generated {len(preprocessing_result.artifacts)} artifacts")
            for artifact in preprocessing_result.artifacts:
                print(f"     - {Path(artifact).name}")
        else:
            print(f"   Preprocessing failed: {preprocessing_result.error_message}")
            return
        
        # 5. Version preprocessing outputs
        print("\n5. Versioning preprocessing outputs...")
        integrator = DataVersioningIntegrator(version_manager)
        
        output_version_ids = integrator.version_preprocessing_outputs(
            preprocessing_result=preprocessing_result,
            preprocessing_config=context.config['data']['preprocessing'],
            source_version_ids=[input_version_id]
        )
        
        print(f"   Created {len(output_version_ids)} output versions:")
        for version_id in output_version_ids:
            print(f"     - {version_id}")
        
        # 6. Demonstrate version tracking and lineage
        print("\n6. Exploring version information...")
        
        # Show input version info
        input_version = version_manager.get_version_info(input_version_id)
        if input_version:
            print(f"   Input version {input_version_id}:")
            print(f"     - File: {Path(input_version.file_path).name}")
            print(f"     - Size: {input_version.size_bytes} bytes")
            print(f"     - Type: {input_version.metadata.get('type', 'unknown')}")
        
        # Show output version info
        if output_version_ids:
            output_version = version_manager.get_version_info(output_version_ids[0])
            if output_version:
                print(f"   Output version {output_version_ids[0]}:")
                print(f"     - File: {Path(output_version.file_path).name}")
                print(f"     - Size: {output_version.size_bytes} bytes")
                print(f"     - Parents: {output_version.parent_versions}")
        
        # 7. Show lineage information
        print("\n7. Data lineage information...")
        for version_id in output_version_ids:
            lineage_records = version_manager.get_lineage_for_version(version_id)
            if lineage_records:
                for record in lineage_records:
                    if record.output_version == version_id:
                        print(f"   Version {version_id} created by:")
                        print(f"     - Transformation: {record.transformation_type}")
                        print(f"     - Sources: {record.source_versions}")
                        print(f"     - Steps: {len(record.transformation_config.get('steps', []))}")
        
        # 8. Show version statistics
        print("\n8. Version statistics...")
        print(f"   Total versions tracked: {len(version_manager.versions)}")
        print(f"   Total lineage records: {len(version_manager.lineage)}")
        
        # List all versions
        print("\n   All versions:")
        for version_id, version in version_manager.versions.items():
            print(f"     - {version_id}: {Path(version.file_path).name} "
                  f"({version.metadata.get('type', 'unknown')})")
        
        print("\n" + "=" * 40)
        print("Data versioning example completed successfully!")
        print("\nKey features demonstrated:")
        print("- Automatic data versioning with hash-based deduplication")
        print("- Integration with preprocessing pipeline")
        print("- Data lineage tracking across transformations")
        print("- Metadata storage for enhanced traceability")
        print("- Version relationship management (parent-child)")


if __name__ == "__main__":
    main()