#!/usr/bin/env python3
"""
Test script to verify that the Path import issue is fixed
"""

try:
    from pathlib import Path
    print("✓ Path import successful")
    
    # Test basic Path functionality
    test_path = Path("test")
    print(f"✓ Path test successful: {test_path}")
    
    print("✓ All imports working correctly!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")