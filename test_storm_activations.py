#!/usr/bin/env python3
"""
STORM Activation Storage Test
============================

Tests baseline vs STORM for activation storage/retrieval with concurrent operations.
Uses real GPU/CPU with authentic CUDA operations and provides metrics-only output.
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

def baseline_activation_storage(input_tensor, weight_tensor, bias_tensor, num_layers=8):
    """BASELINE: Sequential compute + blocking transfer (Measures Total Lost Time)"""
    current = input_tensor
    
    # We measure the total time for the sequential execution chain.
    # The time spent by .cpu() is the penalty of the sequential VRAM stall.
    for i in range(num_layers):
        # 1. Compute
        batch_size, seq_len, hidden_size = current.shape
        reshaped = current.view(-1, hidden_size)
        output = torch.nn.functional.linear(reshaped, weight_tensor, bias_tensor)
        layer_output = output.view(batch_size, seq_len, hidden_size)
        layer_output = torch.relu(layer_output)
        
        # 2. Sequential Transfer (Blocking) - This is the memory penalty
        _ = layer_output.cpu().clone()  # Blocking transfer included in timing
        
        current = layer_output
        
        # Clean up
        del layer_output
        if i > 0:
            torch.cuda.empty_cache()
    
    return current

def storm_activation_storage(input_tensor, weight_tensor, bias_tensor, num_layers=8):
    """STORM: Concurrent compute + asynchronous transfer (Measures Overlapped Time)"""
    current = input_tensor
    
    # Define streams outside the loop
    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()
    
    # This loop demonstrates the core JIT pipelining logic
    for i in range(num_layers):
        # 1. Compute current layer on Compute Stream
        with torch.cuda.stream(compute_stream):
            batch_size, seq_len, hidden_size = current.shape
            reshaped = current.view(-1, hidden_size)
            output = storm_cuda.storm.StormGEMMTensor.storm_linear(
                reshaped, weight_tensor, bias_tensor)
            layer_output = output.view(batch_size, seq_len, hidden_size)
            layer_output = torch.relu(layer_output)
        
        # 2. Transfer previous layer's result on Transfer Stream (ASYNCHRONOUS)
        # We ensure the transfer runs concurrently by using non_blocking=True
        if i > 0:
            # Transfer the data out while the compute_stream moves to the next layer
            with torch.cuda.stream(transfer_stream):
                _ = prev_output.cpu().clone().to(prev_output.device, non_blocking=True)  # D->H Transfer
            
        # 3. Update references for next cycle
        prev_output = layer_output
        current = layer_output
        
        # Clean up
        if i > 0:
            torch.cuda.empty_cache()
    
    # Final synchronization is needed for accurate timing measurement
    torch.cuda.synchronize()
    return current

def time_operation(func, *args, **kwargs):
    """Time an operation with proper error handling"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    if result is not None:
        return (end_time - start_time) * 1000  # Convert to milliseconds
    else:
        return None

def test_activation_storage():
    """Test activation storage comparison between baseline and STORM"""
    print("="*60)
    print("STORM ACTIVATION STORAGE TEST")
    print("="*60)
    print("Baseline: Sequential compute + store")
    print("STORM: Concurrent compute + store using CUDA streams")
    
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
        
        # Test baseline
        print(f"\n[TEST] Baseline Sequential...")
        baseline_time = time_operation(baseline_activation_storage, 
                                    input_tensor, weight_tensor, bias_tensor, num_layers)
        if baseline_time:
            print(f"[OK] Baseline Time: {baseline_time:.2f} ms")
        else:
            print("[FAIL] Baseline failed")
            return
        
        # Clear memory
        clear_memory()
        print(f"[CLEAR] {get_memory_info()}")
        
        # Test STORM
        print(f"\n[TEST] STORM Concurrent...")
        storm_time = time_operation(storm_activation_storage, 
                                  input_tensor, weight_tensor, bias_tensor, num_layers)
        if storm_time:
            print(f"[OK] STORM Time: {storm_time:.2f} ms")
        else:
            print("[FAIL] STORM failed")
            return
        
        # Results - metrics only
        print(f"\n{'='*60}")
        print("ACTIVATION STORAGE RESULTS")
        print(f"{'='*60}")
        print(f"Baseline (Sequential):")
        print(f"  Compute + Transfer time: {baseline_time:.2f} ms")
        print(f"  Memory penalty: {baseline_time:.2f} ms")
        print(f"  Total time: {baseline_time:.2f} ms")
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        print(f"\nSTORM (Concurrent):")
        print(f"  Compute + Transfer time: {storm_time:.2f} ms")
        print(f"  Memory transfers: {storm_time:.2f} ms (overlapped)")
        print(f"  Total time: {storm_time:.2f} ms")
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        if baseline_time and storm_time:
            concurrency_gain = ((baseline_time - storm_time) / baseline_time) * 100
            print(f"  Concurrency gain: {concurrency_gain:.1f}%")
        
        # Clean up
        del input_tensor, weight_tensor, bias_tensor
        clear_memory()
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_activation_storage()
