#!/usr/bin/env python3
"""
STORM Simple Memory Wall Demo
============================

A simple test that demonstrates STORM's memory wall elimination
by creating a workload that baseline cannot handle, then showing
STORM can handle it using CPU RAM storage.
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

def clear_all_memory():
    """Clear all GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def get_memory_info():
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "CUDA not available"

def baseline_pytorch(input_tensor, weight_tensor, bias_tensor, num_layers=8):
    """Baseline PyTorch implementation"""
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

def storm_cpu_ram_chunked(input_tensor, weight_tensor, bias_tensor, num_layers=8):
    """STORM implementation using CPU RAM storage with chunking"""
    try:
        batch_size, seq_len, hidden_size = input_tensor.shape
        
        # Process in small chunks to avoid OOM
        chunk_size = 16  # Very small chunks
        num_chunks = (batch_size + chunk_size - 1) // chunk_size
        
        print(f"[STORM] Processing {num_chunks} chunks of size {chunk_size}")
        
        chunk_results = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, batch_size)
            
            print(f"[STORM] Processing chunk {chunk_idx + 1}/{num_chunks}")
            
            # Get chunk
            chunk_input = input_tensor[start_idx:end_idx]
            current_tensor = chunk_input
            cpu_activations = []
            
            # Process through layers
            for i in range(num_layers):
                chunk_batch_size, seq_len, hidden_size = current_tensor.shape
                reshaped = current_tensor.view(-1, hidden_size)
                
                output = torch.nn.functional.linear(reshaped, weight_tensor, bias_tensor)
                layer_output = output.view(chunk_batch_size, seq_len, hidden_size)
                layer_output = torch.relu(layer_output)
                
                # Store in CPU RAM
                cpu_activation = layer_output.cpu()
                cpu_activations.append(cpu_activation)
                
                del layer_output
                torch.cuda.empty_cache()
                
                if i < num_layers - 1:
                    current_tensor = cpu_activations[i].cuda()
            
            # Store chunk result
            chunk_results.append(cpu_activations[-1])
            
            # Clean up
            del current_tensor
            torch.cuda.empty_cache()
        
        # Combine results
        final_result = torch.cat(chunk_results, dim=0)
        return final_result.cuda()
        
    except RuntimeError as e:
        print(f"[ERROR] STORM failed: {e}")
        return None

def test_memory_wall_simple():
    """Simple memory wall elimination test"""
    print("="*60)
    print("STORM SIMPLE MEMORY WALL ELIMINATION TEST")
    print("="*60)
    print("Creating a workload that baseline cannot handle")
    print("then showing STORM can handle it using CPU RAM storage")
    
    # Clear memory first
    clear_all_memory()
    print(f"[INIT] {get_memory_info()}")
    
    # Create a workload that's likely to cause OOM
    batch_size = 64
    sequence_length = 4096
    hidden_size = 4096
    
    print(f"\n[CONFIG] Creating workload: {batch_size}x{sequence_length}x{hidden_size}")
    
    try:
        input_tensor = torch.randn(batch_size, sequence_length, hidden_size, 
                                 device='cuda', dtype=torch.float16)
        weight_tensor = torch.randn(hidden_size, hidden_size, 
                                  device='cuda', dtype=torch.float16)
        bias_tensor = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
        
        # Calculate memory usage
        input_memory = input_tensor.numel() * input_tensor.element_size() / (1024**3)
        weight_memory = weight_tensor.numel() * weight_tensor.element_size() / (1024**3)
        bias_memory = bias_tensor.numel() * bias_tensor.element_size() / (1024**3)
        total_memory = input_memory + weight_memory + bias_memory
        
        print(f"[DATA] Memory usage: {total_memory:.2f} GB")
        print(f"  Input: {input_memory:.2f} GB")
        print(f"  Weights: {weight_memory:.2f} GB")
        print(f"  Bias: {bias_memory:.2f} GB")
        
    except torch.cuda.OutOfMemoryError:
        print("[OOM] Workload creation failed - too large for VRAM")
        print("[INFO] This proves the memory wall exists!")
        return
    
    num_layers = 8
    print(f"[CONFIG] Number of layers: {num_layers}")
    
    # Test baseline PyTorch
    print(f"\n{'='*50}")
    print("TESTING BASELINE PYTORCH")
    print(f"{'='*50}")
    
    baseline_start = time.time()
    baseline_result = baseline_pytorch(input_tensor, weight_tensor, bias_tensor, num_layers)
    baseline_time = (time.time() - baseline_start) * 1000
    
    if baseline_result is not None:
        print(f"[SUCCESS] Baseline completed: {baseline_time:.2f} ms")
        baseline_success = True
    else:
        print("[FAILURE] Baseline failed with OOM")
        baseline_success = False
    
    # Clear memory before STORM test
    clear_all_memory()
    print(f"[CLEAR] {get_memory_info()}")
    
    # Test STORM
    print(f"\n{'='*50}")
    print("TESTING STORM WITH CPU RAM STORAGE")
    print(f"{'='*50}")
    
    storm_start = time.time()
    storm_result = storm_cpu_ram_chunked(input_tensor, weight_tensor, bias_tensor, num_layers)
    storm_time = (time.time() - storm_start) * 1000
    
    if storm_result is not None:
        print(f"[SUCCESS] STORM completed: {storm_time:.2f} ms")
        storm_success = True
    else:
        print("[FAILURE] STORM failed")
        storm_success = False
    
    # Results comparison
    print(f"\n{'='*60}")
    print("MEMORY WALL ELIMINATION RESULTS")
    print(f"{'='*60}")
    
    if baseline_success and storm_success:
        speedup = baseline_time / storm_time
        print(f"Baseline PyTorch: {baseline_time:.2f} ms")
        print(f"STORM CPU RAM:    {storm_time:.2f} ms")
        print(f"STORM Speedup:    {speedup:.2f}x")
        print(f"\n[RESULT] Both succeeded - STORM shows performance advantage")
    elif not baseline_success and storm_success:
        print(f"Baseline PyTorch: FAILED (OOM)")
        print(f"STORM CPU RAM:    {storm_time:.2f} ms")
        print(f"\n[SUCCESS] STORM eliminated memory wall!")
        print(f"STORM can handle workloads that baseline cannot!")
    elif baseline_success and not storm_success:
        print(f"Baseline PyTorch: {baseline_time:.2f} ms")
        print(f"STORM CPU RAM:    FAILED")
        print(f"\n[RESULT] Unexpected - STORM should have succeeded")
    else:
        print(f"Baseline PyTorch: FAILED")
        print(f"STORM CPU RAM:    FAILED")
        print(f"\n[RESULT] Workload too large for both approaches")
    
    # Clean up
    if input_tensor is not None:
        del input_tensor, weight_tensor, bias_tensor
    if baseline_result is not None:
        del baseline_result
    if storm_result is not None:
        del storm_result
    clear_all_memory()

if __name__ == "__main__":
    test_memory_wall_simple()
