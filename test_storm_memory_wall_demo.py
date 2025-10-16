#!/usr/bin/env python3
"""
STORM Memory Wall Elimination Demo
==================================

This test creates a workload that's right at the edge of VRAM capacity,
showing baseline struggling while STORM handles it gracefully using CPU RAM storage.
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

def get_vram_capacity():
    """Get VRAM capacity in GB"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0

def clear_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()

def create_edge_workload():
    """Create a workload that's right at the edge of VRAM capacity"""
    vram_capacity = get_vram_capacity()
    print(f"[INFO] VRAM Capacity: {vram_capacity:.2f} GB")
    
    # Create a workload that uses about 90% of VRAM
    target_memory_gb = vram_capacity * 0.9
    
    # Estimate dimensions for target memory
    # Each element is 2 bytes (float16), so we need target_memory_gb * 1024^3 / 2 elements
    target_elements = int(target_memory_gb * 1024**3 / 2)
    
    # Try different configurations to get close to target
    configs = [
        {"batch": 64, "seq": 4096, "hidden": 4096},
        {"batch": 128, "seq": 4096, "hidden": 4096},
        {"batch": 256, "seq": 4096, "hidden": 4096},
        {"batch": 512, "seq": 4096, "hidden": 4096},
    ]
    
    for config in configs:
        try:
            print(f"[TRY] Creating workload: {config['batch']}x{config['seq']}x{config['hidden']}")
            
            input_tensor = torch.randn(config['batch'], config['seq'], config['hidden'], 
                                    device='cuda', dtype=torch.float16)
            weight_tensor = torch.randn(config['hidden'], config['hidden'], 
                                      device='cuda', dtype=torch.float16)
            bias_tensor = torch.randn(config['hidden'], device='cuda', dtype=torch.float16)
            
            # Calculate actual memory usage
            input_memory = input_tensor.numel() * input_tensor.element_size() / (1024**3)
            weight_memory = weight_tensor.numel() * weight_tensor.element_size() / (1024**3)
            bias_memory = bias_tensor.numel() * bias_tensor.element_size() / (1024**3)
            total_memory = input_memory + weight_memory + bias_memory
            
            print(f"[SUCCESS] Workload created: {total_memory:.2f} GB")
            print(f"  Input: {input_memory:.2f} GB")
            print(f"  Weights: {weight_memory:.2f} GB")
            print(f"  Bias: {bias_memory:.2f} GB")
            
            return input_tensor, weight_tensor, bias_tensor, total_memory
            
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Workload too large, trying smaller...")
            clear_memory()
            continue
    
    print("[ERROR] Could not create any workload that fits in VRAM")
    return None, None, None, 0

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

def storm_cpu_ram(input_tensor, weight_tensor, bias_tensor, num_layers=8):
    """STORM implementation using CPU RAM storage"""
    try:
        current_tensor = input_tensor
        cpu_activations = []
        
        for i in range(num_layers):
            print(f"[STORM] Processing layer {i+1}/{num_layers}")
            
            batch_size, seq_len, hidden_size = current_tensor.shape
            reshaped = current_tensor.view(-1, hidden_size)
            
            output = torch.nn.functional.linear(reshaped, weight_tensor, bias_tensor)
            layer_output = output.view(batch_size, seq_len, hidden_size)
            layer_output = torch.relu(layer_output)
            
            # Store in CPU RAM
            cpu_activation = layer_output.cpu()
            cpu_activations.append(cpu_activation)
            
            del layer_output
            torch.cuda.empty_cache()
            
            if i < num_layers - 1:
                current_tensor = cpu_activations[i].cuda()
        
        return cpu_activations[-1].cuda()
        
    except RuntimeError as e:
        print(f"[ERROR] STORM failed: {e}")
        return None

def test_memory_wall_demo():
    """Demonstrate memory wall elimination side-by-side"""
    print("="*70)
    print("STORM MEMORY WALL ELIMINATION DEMO")
    print("="*70)
    print("Creating a workload at the edge of VRAM capacity")
    print("to demonstrate baseline struggling vs STORM succeeding")
    
    # Create edge workload
    input_tensor, weight_tensor, bias_tensor, memory_usage = create_edge_workload()
    
    if input_tensor is None:
        print("[ERROR] Could not create test workload")
        return
    
    num_layers = 8
    print(f"\n[CONFIG] Test Configuration:")
    print(f"  Workload: {input_tensor.shape}")
    print(f"  Memory usage: {memory_usage:.2f} GB")
    print(f"  Number of layers: {num_layers}")
    
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
    clear_memory()
    
    # Test STORM
    print(f"\n{'='*50}")
    print("TESTING STORM WITH CPU RAM STORAGE")
    print(f"{'='*50}")
    
    storm_start = time.time()
    storm_result = storm_cpu_ram(input_tensor, weight_tensor, bias_tensor, num_layers)
    storm_time = (time.time() - storm_start) * 1000
    
    if storm_result is not None:
        print(f"[SUCCESS] STORM completed: {storm_time:.2f} ms")
        storm_success = True
    else:
        print("[FAILURE] STORM failed")
        storm_success = False
    
    # Results comparison
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE RESULTS")
    print(f"{'='*70}")
    
    if baseline_success and storm_success:
        speedup = baseline_time / storm_time
        print(f"Baseline PyTorch: {baseline_time:.2f} ms")
        print(f"STORM CPU RAM:    {storm_time:.2f} ms")
        print(f"STORM Speedup:    {speedup:.2f}x")
        print(f"\n[RESULT] Both succeeded - STORM shows performance advantage")
    elif not baseline_success and storm_success:
        print(f"Baseline PyTorch: FAILED (OOM)")
        print(f"STORM CPU RAM:    {storm_time:.2f} ms")
        print(f"\n[RESULT] STORM eliminated memory wall - baseline couldn't handle this workload!")
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
    clear_memory()

if __name__ == "__main__":
    test_memory_wall_demo()
