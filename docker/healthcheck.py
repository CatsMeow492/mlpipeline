#!/usr/bin/env python3
"""
Health check script for ML Pipeline Docker containers.
This script verifies that all components are working correctly.
"""

import sys
import time
import requests
import psycopg2
import redis
from pathlib import Path

def check_database():
    """Check PostgreSQL database connectivity."""
    try:
        conn = psycopg2.connect(
            host="postgres",
            database="mlpipeline",
            user="mlpipeline",
            password="mlpipeline123"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def check_redis():
    """Check Redis connectivity."""
    try:
        r = redis.Redis(host='redis', port=6379, decode_responses=True)
        r.ping()
        print("✓ Redis connection successful")
        return True
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False

def check_mlflow():
    """Check MLflow server."""
    try:
        response = requests.get("http://mlflow:5000/health", timeout=10)
        if response.status_code == 200:
            print("✓ MLflow server is healthy")
            return True
        else:
            print(f"✗ MLflow server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ MLflow server check failed: {e}")
        return False

def check_application():
    """Check main application."""
    try:
        # Try to import the main package
        import mlpipeline
        print("✓ ML Pipeline package import successful")
        
        # Check if CLI is accessible
        from mlpipeline.cli import app
        print("✓ CLI module accessible")
        
        return True
    except Exception as e:
        print(f"✗ Application check failed: {e}")
        return False

def check_file_permissions():
    """Check file permissions and directories."""
    try:
        required_dirs = ["/app/data", "/app/models", "/app/logs"]
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists() and path.is_dir():
                # Try to create a test file
                test_file = path / "health_check_test.tmp"
                test_file.write_text("test")
                test_file.unlink()
                print(f"✓ Directory {dir_path} is writable")
            else:
                print(f"⚠ Directory {dir_path} does not exist")
        return True
    except Exception as e:
        print(f"✗ File permission check failed: {e}")
        return False

def main():
    """Run all health checks."""
    print("Starting ML Pipeline health checks...")
    print("=" * 50)
    
    checks = [
        ("Application", check_application),
        ("File Permissions", check_file_permissions),
        ("Database", check_database),
        ("Redis", check_redis),
        ("MLflow", check_mlflow),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {name} check failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Health Check Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (name, _) in enumerate(checks):
        status = "PASS" if results[i] else "FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All health checks passed!")
        sys.exit(0)
    else:
        print("❌ Some health checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()