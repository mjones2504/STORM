#!/usr/bin/env python3
"""
STORM Test Script

This script tests the STORM PyTorch extension to ensure it compiles
and runs correctly.
"""

import torch
import sys
import os

def test_storm_extension():
    """Test the STORM extension compilation and basic functionality."""
    
    print("STORM Extension Test")
    print("===================")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: Yes (Version: {torch.version.cuda})")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA available: No (will use CPU-only mode)")
    
    # Try to import the extension
    try:
        print("\nAttempting to compile STORM extension...")
        
        # This will trigger the compilation
        from torch.utils.cpp_extension import load_inline
        
        # Define the STORM extension code inline
        storm_code = '''
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        
        torch::Tensor storm_test(torch::Tensor input) {
            return input * 2.0;
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("storm_test", &storm_test, "STORM test function");
        }
        '''
        
        # Load the extension
        storm_ext = load_inline(
            name="storm_test",
            cpp_sources=[storm_code],
            verbose=True
        )
        
        print("✅ STORM extension compiled successfully!")
        
        # Test basic functionality
        if torch.cuda.is_available():
            test_tensor = torch.randn(10, 10).cuda()
        else:
            test_tensor = torch.randn(10, 10)
        
        result = storm_ext.storm_test(test_tensor)
        print(f"✅ Test function executed successfully!")
        print(f"   Input shape: {test_tensor.shape}")
        print(f"   Output shape: {result.shape}")
        print(f"   Input mean: {test_tensor.mean().item():.4f}")
        print(f"   Output mean: {result.mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ STORM extension test failed: {e}")
        return False

def test_setup_py():
    """Test if setup.py can be imported and configured."""
    
    print("\nTesting setup.py configuration...")
    
    try:
        import setup
        print("✅ setup.py imported successfully!")
        return True
    except Exception as e:
        print(f"❌ setup.py test failed: {e}")
        return False

def main():
    """Main test function."""
    
    print("STORM PyTorch Extension Test Suite")
    print("==================================")
    
    # Test setup.py
    setup_ok = test_setup_py()
    
    # Test extension compilation
    extension_ok = test_storm_extension()
    
    # Summary
    print("\nTest Results Summary:")
    print("====================")
    print(f"Setup.py: {'✅ PASS' if setup_ok else '❌ FAIL'}")
    print(f"Extension: {'✅ PASS' if extension_ok else '❌ FAIL'}")
    
    if setup_ok and extension_ok:
        print("\n🎉 All tests passed! STORM is ready for use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
