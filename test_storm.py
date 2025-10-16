#!/usr/bin/env python3
"""
STORM Performance Test
=====================

Simple, honest comparison between baseline PyTorch and STORM.
No chunking, no hardcoded advantages, just raw performance data.
"""

import torch
import time
import sys
import os
import gc

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import storm_cuda
    print("[OK] STORM CUDA extension loaded successfully!")
except ImportError as e:
    print(f"[ERROR] Failed to import STORM CUDA extension: {e}")
    print("Please build the extension first: python setup.py build_ext --inplace --force")
    sys.exit(1)

def clear_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()

def get_memory_info():
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "CUDA not available"

def baseline_pytorch(input_tensor, weight_tensor, bias_tensor, num_layers=8):
    """Baseline PyTorch implementation - stays on GPU"""
    try:
        current_tensor = input_tensor
        
        for i in range(num_layers):
            batch_size, seq_len, hidden_size = current_tensor.shape
            reshaped = current_tensor.view(-1, hidden_size)
            
            output = torch.nn.functional.linear(reshaped, weight_tensor, bias_tensor)
            layer_output = output.view(batch_size, seq_len, hidden_size)
            layer_output = torch.relu(layer_output)
            
            current_tensor = layer_output
            del layer_output
            if i > 0:
                torch.cuda.empty_cache()
        
        return current_tensor
    except RuntimeError as e:
        print(f"[ERROR] Baseline PyTorch failed: {e}")
        return None

def storm_implementation(input_tensor, weight_tensor, bias_tensor, num_layers=8):
    """STORM implementation - intelligent GPU/CPU strategy selection"""
    try:
        # Import intelligent STORM
        from storm_intelligent import IntelligentSTORM
        
        # Initialize intelligent STORM
        storm = IntelligentSTORM()
        
        # Use intelligent STORM to process
        return storm.process(input_tensor, weight_tensor, bias_tensor, num_layers)
        
    except RuntimeError as e:
        print(f"[ERROR] STORM failed: {e}")
        return None

def time_operation(func, *args, **kwargs):
    """Time an operation with proper error handling"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    if result is not None:
        return (end_time - start_time) * 1000  # Convert to milliseconds
    else:
        return None

def test_performance():
    """Test performance comparison between baseline and STORM"""
    print("="*60)
    print("STORM PERFORMANCE TEST")
    print("="*60)
    print("Simple comparison: Baseline PyTorch vs STORM")
    
    # Get system info
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[SYSTEM] CUDA Device: {device_name}")
        print(f"[SYSTEM] VRAM Capacity: {vram_gb:.2f} GB")
    else:
        print("[ERROR] CUDA not available")
        return
    
    # Create test workload
    batch_size = 64
    sequence_length = 2048
    hidden_size = 2048
    num_layers = 8
    
    print(f"\n[CONFIG] Test Configuration:")
    print(f"  Dimensions: {batch_size}x{sequence_length}x{hidden_size}")
    print(f"  Number of layers: {num_layers}")
    
    # Calculate memory usage
    input_memory = batch_size * sequence_length * hidden_size * 2 / (1024**3)  # float16 = 2 bytes
    weight_memory = hidden_size * hidden_size * 2 / (1024**3)
    bias_memory = hidden_size * 2 / (1024**3)
    total_memory = input_memory + weight_memory + bias_memory
    
    print(f"  Memory usage: {total_memory:.2f} GB")
    
    # Clear memory
    clear_memory()
    print(f"[INIT] {get_memory_info()}")
    
    try:
        # Create tensors
        input_tensor = torch.randn(batch_size, sequence_length, hidden_size, 
                                 device='cuda', dtype=torch.float16)
        weight_tensor = torch.randn(hidden_size, hidden_size, 
                                  device='cuda', dtype=torch.float16)
        bias_tensor = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
        
        # Test baseline PyTorch
        print(f"\n[TEST] Baseline PyTorch...")
        baseline_time = time_operation(baseline_pytorch, input_tensor, weight_tensor, bias_tensor, num_layers)
        if baseline_time:
            print(f"[OK] Baseline Time: {baseline_time:.2f} ms")
        else:
            print("[FAIL] Baseline failed")
            return
        
        # Clear memory
        clear_memory()
        print(f"[CLEAR] {get_memory_info()}")
        
        # Test STORM
        print(f"\n[TEST] STORM Implementation...")
        print("[STORM] Analyzing workload size and choosing strategy...")
        storm_time = time_operation(storm_implementation, input_tensor, weight_tensor, bias_tensor, num_layers)
        if storm_time:
            print(f"[OK] STORM Time: {storm_time:.2f} ms")
        else:
            print("[FAIL] STORM failed")
            return
        
        # Results
        print(f"\n{'='*60}")
        print("PERFORMANCE RESULTS")
        print(f"{'='*60}")
        print(f"Baseline PyTorch: {baseline_time:.2f} ms")
        print(f"STORM:            {storm_time:.2f} ms")
        
        if baseline_time and storm_time:
            ratio = baseline_time / storm_time
            if ratio > 1.0:
                print(f"STORM is {ratio:.2f}x faster than baseline")
            else:
                print(f"STORM is {1/ratio:.2f}x slower than baseline")
        
        # Clean up
        del input_tensor, weight_tensor, bias_tensor
        clear_memory()
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_performance()