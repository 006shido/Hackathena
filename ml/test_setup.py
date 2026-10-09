#!/usr/bin/env python3

"""
Quick test script to verify the ML pipeline components are set up correctly.
"""

import os
import sys

def test_setup():
    """Test that all required files and directories exist"""

    print("Testing ML pipeline setup...")

    # Check directories exist
    required_dirs = [
        'ml/data',
        'ml/models',
        'ml/training'
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"ERROR: Directory {dir_path} does not exist")
            return False
        else:
            print(f"✓ Directory {dir_path} exists")

    # Check files exist
    required_files = [
        'ml/requirements.txt',
        'ml/training/train.py',
        'ml/training/inference.py',
        'ml/models/model.py',
        'ml/training/prepare_dataset.py'
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: File {file_path} does not exist")
            return False
        else:
            print(f"✓ File {file_path} exists")

    # Check that model.py imports correctly
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from ml.models.model import get_model
        model = get_model(10)
        print("✓ Model creation works correctly")
    except Exception as e:
        print(f"ERROR: Failed to create model - {e}")
        return False

    print("\nAll components are set up correctly!")
    return True

if __name__ == "__main__":
    test_setup()
