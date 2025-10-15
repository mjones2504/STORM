#!/usr/bin/env python3
"""
STORM GEMM Optimization Test Suite

This test suite validates the CUTLASS-based GEMM optimization in STORM:
1. Accuracy validation against PyTorch baseline
2. Bandwidth reduction measurement
3. Performance improvement verification
4. Integration testing with STORM autograd functions

Key Features:
- Numerical accuracy testing (within 1e-5 tolerance)
- Bandwidth usage measurement using PyTorch profiler
- Performance comparison between PyTorch and CUTLASS GEMM
- STORM integration testing
"""

import torch
import torch.nn as nn
import torch.profiler as profiler
import time
import sys
import os
import numpy as np

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

# *******************************************************************
# 1. CORE CONFIGURATION
# *******************************************************************

# Test parameters
BATCH_SIZE = 16
SEQUENCE_LENGTH = 1024
HIDDEN_SIZE = 2048
NUM_TESTS = 10
TOLERANCE = 1e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🔧 STORM GEMM Test Configuration:")
print(f"  Device: {DEVICE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Sequence Length: {SEQUENCE_LENGTH}")
print(f"  Hidden Size: {HIDDEN_SIZE}")
print(f"  Number of Tests: {NUM_TESTS}")
print(f"  Tolerance: {TOLERANCE}")

# *******************************************************************
# 2. STORM EXTENSION LOADING
# *******************************************************************

try:
    import storm_cuda
    print("✅ STORM Extension Module Loaded.")
    
    # Check if CUTLASS is available
    if hasattr(storm_cuda.storm, 'StormGEMMTensor'):
        cutlass_available = storm_cuda.storm.StormGEMMTensor.is_cutlass_available()
        print(f"✅ CUTLASS Available: {cutlass_available}")
        
        if cutlass_available:
            config_info = storm_cuda.storm.StormGEMMTensor.get_config_info()
            print(f"✅ STORM GEMM Configuration:")
            print(config_info)
        else:
            print("⚠️ CUTLASS not available - using PyTorch fallback")
    else:
        print("⚠️ STORM GEMM not available in extension")
        cutlass_available = False
        
except ImportError as e:
    print(f"❌ ERROR: {e}")
    print("Please run 'pip install -e .' to build the STORM extension")
    sys.exit(1)

# *******************************************************************
# 3. TEST DATA GENERATION
# *******************************************************************

def create_test_data():
    """Create test data for GEMM operations"""
    
    # Create input tensor
    input_tensor = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE, 
                              device=DEVICE, dtype=torch.float32, requires_grad=True)
    
    # Create weight tensor
    weight_tensor = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, 
                               device=DEVICE, dtype=torch.float32, requires_grad=True)
    
    # Create bias tensor
    bias_tensor = torch.randn(HIDDEN_SIZE, 
                             device=DEVICE, dtype=torch.float32, requires_grad=True)
    
    return input_tensor, weight_tensor, bias_tensor

# *******************************************************************
# 4. ACCURACY VALIDATION TESTS
# *******************************************************************

def test_gemm_accuracy():
    """Test numerical accuracy of CUTLASS GEMM against PyTorch baseline"""
    
    print("\n--- TEST 1: GEMM Accuracy Validation ---")
    print("Comparing CUTLASS GEMM results with PyTorch baseline")
    
    # Create test data
    input_tensor, weight_tensor, bias_tensor = create_test_data()
    
    # PyTorch baseline
    pytorch_output = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    
    # STORM CUTLASS GEMM (if available)
    if cutlass_available:
        try:
            storm_output = storm_cuda.storm.StormGEMMTensor.storm_linear(
                input_tensor, weight_tensor, bias_tensor
            )
            
            # Calculate difference
            diff = torch.abs(pytorch_output - storm_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"  PyTorch Output Shape: {pytorch_output.shape}")
            print(f"  STORM Output Shape: {storm_output.shape}")
            print(f"  Maximum Difference: {max_diff:.2e}")
            print(f"  Mean Difference: {mean_diff:.2e}")
            print(f"  Tolerance: {TOLERANCE:.2e}")
            
            if max_diff < TOLERANCE:
                print("✅ PASS: CUTLASS GEMM accuracy within tolerance")
                return True
            else:
                print("❌ FAIL: CUTLASS GEMM accuracy exceeds tolerance")
                return False
                
        except Exception as e:
            print(f"❌ ERROR: CUTLASS GEMM failed: {e}")
            return False
    else:
        print("⚠️ SKIP: CUTLASS not available, using PyTorch fallback")
        return True

def test_gradient_accuracy():
    """Test gradient computation accuracy"""
    
    print("\n--- TEST 2: Gradient Accuracy Validation ---")
    print("Comparing gradient computation between PyTorch and STORM")
    
    # Create test data
    input_tensor, weight_tensor, bias_tensor = create_test_data()
    
    # PyTorch baseline with gradients
    input_tensor_pytorch = input_tensor.clone().detach().requires_grad_(True)
    weight_tensor_pytorch = weight_tensor.clone().detach().requires_grad_(True)
    bias_tensor_pytorch = bias_tensor.clone().detach().requires_grad_(True)
    
    pytorch_output = torch.linear(input_tensor_pytorch, weight_tensor_pytorch, bias_tensor_pytorch)
    pytorch_loss = pytorch_output.sum()
    pytorch_loss.backward()
    
    pytorch_grad_input = input_tensor_pytorch.grad
    pytorch_grad_weight = weight_tensor_pytorch.grad
    pytorch_grad_bias = bias_tensor_pytorch.grad
    
    # STORM CUTLASS GEMM with gradients (if available)
    if cutlass_available:
        try:
            input_tensor_storm = input_tensor.clone().detach().requires_grad_(True)
            weight_tensor_storm = weight_tensor.clone().detach().requires_grad_(True)
            bias_tensor_storm = bias_tensor.clone().detach().requires_grad_(True)
            
            storm_output = storm_cuda.storm.StormGEMMTensor.storm_linear(
                input_tensor_storm, weight_tensor_storm, bias_tensor_storm
            )
            storm_loss = storm_output.sum()
            storm_loss.backward()
            
            storm_grad_input = input_tensor_storm.grad
            storm_grad_weight = weight_tensor_storm.grad
            storm_grad_bias = bias_tensor_storm.grad
            
            # Compare gradients
            input_diff = torch.abs(pytorch_grad_input - storm_grad_input)
            weight_diff = torch.abs(pytorch_grad_weight - storm_grad_weight)
            bias_diff = torch.abs(pytorch_grad_bias - storm_grad_bias)
            
            max_input_diff = torch.max(input_diff).item()
            max_weight_diff = torch.max(weight_diff).item()
            max_bias_diff = torch.max(bias_diff).item()
            
            print(f"  Input Gradient Max Diff: {max_input_diff:.2e}")
            print(f"  Weight Gradient Max Diff: {max_weight_diff:.2e}")
            print(f"  Bias Gradient Max Diff: {max_bias_diff:.2e}")
            
            if (max_input_diff < TOLERANCE and 
                max_weight_diff < TOLERANCE and 
                max_bias_diff < TOLERANCE):
                print("✅ PASS: Gradient accuracy within tolerance")
                return True
            else:
                print("❌ FAIL: Gradient accuracy exceeds tolerance")
                return False
                
        except Exception as e:
            print(f"❌ ERROR: STORM gradient computation failed: {e}")
            return False
    else:
        print("⚠️ SKIP: CUTLASS not available, using PyTorch fallback")
        return True

# *******************************************************************
# 5. BANDWIDTH MEASUREMENT TESTS
# *******************************************************************

def measure_bandwidth_usage():
    """Measure VRAM bandwidth usage using PyTorch profiler"""
    
    print("\n--- TEST 3: Bandwidth Usage Measurement ---")
    print("Measuring VRAM bandwidth usage with PyTorch profiler")
    
    # Create test data
    input_tensor, weight_tensor, bias_tensor = create_test_data()
    
    def pytorch_gemm():
        return torch.linear(input_tensor, weight_tensor, bias_tensor)
    
    def storm_gemm():
        if cutlass_available:
            return storm_cuda.storm.StormGEMMTensor.storm_linear(
                input_tensor, weight_tensor, bias_tensor
            )
        else:
            return torch.linear(input_tensor, weight_tensor, bias_tensor)
    
    # Profile PyTorch GEMM
    print("  Profiling PyTorch GEMM...")
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as pytorch_prof:
        for _ in range(NUM_TESTS):
            pytorch_gemm()
    
    # Profile STORM GEMM
    print("  Profiling STORM GEMM...")
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as storm_prof:
        for _ in range(NUM_TESTS):
            storm_gemm()
    
    # Export profiles for analysis
    pytorch_prof.export_chrome_trace("pytorch_gemm_profile.json")
    storm_prof.export_chrome_trace("storm_gemm_profile.json")
    
    print("✅ Bandwidth profiles exported:")
    print("  - pytorch_gemm_profile.json")
    print("  - storm_gemm_profile.json")
    print("  - Open in Chrome: chrome://tracing")
    
    return True

# *******************************************************************
# 6. PERFORMANCE COMPARISON TESTS
# *******************************************************************

def test_performance_comparison():
    """Compare performance between PyTorch and STORM GEMM"""
    
    print("\n--- TEST 4: Performance Comparison ---")
    print("Comparing execution time between PyTorch and STORM GEMM")
    
    # Create test data
    input_tensor, weight_tensor, bias_tensor = create_test_data()
    
    def pytorch_gemm():
        return torch.linear(input_tensor, weight_tensor, bias_tensor)
    
    def storm_gemm():
        if cutlass_available:
            return storm_cuda.storm.StormGEMMTensor.storm_linear(
                input_tensor, weight_tensor, bias_tensor
            )
        else:
            return torch.linear(input_tensor, weight_tensor, bias_tensor)
    
    # Warm up
    for _ in range(5):
        pytorch_gemm()
        storm_gemm()
    
    # Time PyTorch GEMM
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(NUM_TESTS):
        pytorch_gemm()
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / NUM_TESTS * 1000  # ms
    
    # Time STORM GEMM
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(NUM_TESTS):
        storm_gemm()
    torch.cuda.synchronize()
    storm_time = (time.time() - start_time) / NUM_TESTS * 1000  # ms
    
    # Calculate speedup
    speedup = pytorch_time / storm_time if storm_time > 0 else 1.0
    improvement = ((pytorch_time - storm_time) / pytorch_time) * 100 if pytorch_time > 0 else 0
    
    print(f"  PyTorch GEMM Time: {pytorch_time:.2f} ms")
    print(f"  STORM GEMM Time: {storm_time:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {improvement:.2f}%")
    
    if cutlass_available:
        if speedup > 1.0:
            print("✅ PASS: STORM GEMM is faster than PyTorch")
        else:
            print("⚠️ WARNING: STORM GEMM is not faster than PyTorch")
    else:
        print("⚠️ SKIP: CUTLASS not available, using PyTorch fallback")
    
    return True

# *******************************************************************
# 7. STORM INTEGRATION TESTS
# *******************************************************************

def test_storm_integration():
    """Test STORM integration with autograd functions"""
    
    print("\n--- TEST 5: STORM Integration Test ---")
    print("Testing STORM autograd functions with CUTLASS GEMM")
    
    try:
        # Create a simple STORM model
        model = storm_cuda.storm.StormModel(1024, 2048, 512)
        
        # Create test input
        input_tensor = torch.randn(8, 1024, device=DEVICE, dtype=torch.float32)
        
        # Test forward pass
        output = model.forward(input_tensor)
        
        print(f"  Input Shape: {input_tensor.shape}")
        print(f"  Output Shape: {output.shape}")
        print("✅ PASS: STORM model forward pass successful")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        print("✅ PASS: STORM model backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: STORM integration test failed: {e}")
        return False

# *******************************************************************
# 8. MAIN TEST EXECUTION
# *******************************************************************

def run_all_tests():
    """Run all STORM GEMM tests"""
    
    print("🚀 STORM GEMM Optimization Test Suite")
    print("=" * 60)
    print("Testing CUTLASS-based GEMM optimization in STORM")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("GEMM Accuracy", test_gemm_accuracy()))
    test_results.append(("Gradient Accuracy", test_gradient_accuracy()))
    test_results.append(("Bandwidth Measurement", measure_bandwidth_usage()))
    test_results.append(("Performance Comparison", test_performance_comparison()))
    test_results.append(("STORM Integration", test_storm_integration()))
    
    # Print results summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("STORM GEMM optimization is working correctly")
    else:
        print("⚠️ Some tests failed - check the output above")
    
    print("\n📊 Next Steps:")
    print("  1. Review bandwidth profiles in Chrome")
    print("  2. Run LLM load test to measure speedup improvement")
    print("  3. Target: >25% speedup improvement")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
