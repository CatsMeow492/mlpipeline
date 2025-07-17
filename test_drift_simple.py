"""
Simple test to verify drift detection functionality without pytest.
"""

import sys
import traceback
import tempfile
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '.')

try:
    import numpy as np
    import pandas as pd
    from mlpipeline.monitoring.drift_detection import DriftDetector, DriftMonitor
    print("✓ Successfully imported drift detection modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic drift detection functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # Create sample data
        np.random.seed(42)
        baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
        })
        
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(5, 2, 50),
            'feature3': np.random.choice(['A', 'B', 'C'], 50),
        })
        
        print(f"✓ Created sample data - baseline: {baseline_data.shape}, current: {current_data.shape}")
        
        # Test DriftDetector initialization
        with tempfile.TemporaryDirectory() as temp_dir:
            detector = DriftDetector(
                baseline_data=baseline_data,
                output_dir=temp_dir
            )
            print("✓ DriftDetector initialized successfully")
            
            # Test baseline statistics
            stats = detector.get_baseline_statistics()
            assert 'shape' in stats
            assert 'columns' in stats
            print("✓ Baseline statistics computed successfully")
            
            # Test setting baseline
            detector.set_baseline(baseline_data)
            print("✓ Baseline data set successfully")
            
        print("✓ Basic functionality test passed")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_drift_monitor():
    """Test DriftMonitor functionality."""
    print("\n=== Testing DriftMonitor ===")
    
    try:
        # Create sample data
        np.random.seed(42)
        baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
        })
        
        # Create mock detector (simplified)
        with tempfile.TemporaryDirectory() as temp_dir:
            detector = DriftDetector(
                baseline_data=baseline_data,
                output_dir=temp_dir
            )
            
            # Test DriftMonitor
            monitor = DriftMonitor()
            monitor.add_detector('test_detector', detector)
            
            assert 'test_detector' in monitor.detectors
            print("✓ DriftMonitor detector addition successful")
            
            # Test monitoring history
            history = monitor.get_monitoring_history()
            assert isinstance(history, list)
            print("✓ DriftMonitor history retrieval successful")
            
        print("✓ DriftMonitor test passed")
        
    except Exception as e:
        print(f"✗ DriftMonitor test failed: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_error_handling():
    """Test error handling."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test detector without baseline data
        detector = DriftDetector()
        
        try:
            detector.detect_data_drift(pd.DataFrame({'col': [1, 2, 3]}))
            print("✗ Should have raised error for missing baseline")
            return False
        except Exception:
            print("✓ Correctly raised error for missing baseline")
        
        # Test baseline statistics without data
        stats = detector.get_baseline_statistics()
        assert stats == {}
        print("✓ Correctly handled missing baseline for statistics")
        
        print("✓ Error handling test passed")
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("Starting Drift Detection Tests")
    
    tests = [
        test_basic_functionality,
        test_drift_monitor,
        test_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)