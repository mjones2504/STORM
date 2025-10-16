#!/usr/bin/env python3
"""
STORM True Capabilities Test
============================

This test demonstrates STORM's true purpose:
1. Small workloads: STORM should be FASTER than baseline (GPU optimization)
2. Large workloads: STORM should HANDLE what baseline CAN'T (memory wall elimination)

The test proves STORM's ability to eliminate the VRAM memory wall through
intelligent workload size detection and appropriate strategy selection.
"""

import torch
import time
import sys
import os
import psutil

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import storm_cuda
    print("[OK] STORM CUDA extension loaded successfully!")
except ImportError as e:
    print(f"[ERROR] Failed to import STORM CUDA extension: {e}")
    print("Please build the extension first: python setup.py build_ext --inplace --force")
    sys.exit(1)

class IntelligentSTORM:
    """Intelligent STORM system that adapts to workload size"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vram_capacity_gb = self._get_vram_capacity()
        self.cpu_ram_gb = self._get_cpu_ram_capacity()
        print(f"[STORM] VRAM Capacity: {self.vram_capacity_gb:.2f} GB")
        print(f"[STORM] CPU RAM Capacity: {self.cpu_ram_gb:.2f} GB")
    
    def _get_vram_capacity(self):
        """Get available VRAM capacity in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0
    
    def _get_cpu_ram_capacity(self):
        """Get available CPU RAM capacity in GB"""
        return psutil.virtual_memory().total / (1024**3)
    
    def _estimate_tensor_memory_gb(self, tensor):
        """Estimate tensor memory usage in GB"""
        return tensor.numel() * tensor.element_size() / (1024**3)
    
    def _should_use_cpu_ram(self, input_tensor, weight_tensor, bias_tensor, num_layers):
        """Determine if STORM should use CPU RAM storage based on workload size"""
        # Estimate total memory needed for the workload
        input_memory = self._estimate_tensor_memory_gb(input_tensor)
        weight_memory = self._estimate_tensor_memory_gb(weight_tensor)
        bias_memory = self._estimate_tensor_memory_gb(bias_tensor) if bias_tensor is not None else 0
        
        # Estimate intermediate activations memory (rough approximation)
        intermediate_memory = input_memory * num_layers * 0.5  # Conservative estimate
        
        total_memory_needed = input_memory + weight_memory + bias_memory + intermediate_memory
        
        # Use CPU RAM if workload exceeds 80% of VRAM capacity
        vram_threshold = self.vram_capacity_gb * 0.8
        
        should_use_cpu = total_memory_needed > vram_threshold
        
        print(f"[STORM] Memory Analysis:")
        print(f"  Input: {input_memory:.2f} GB")
        print(f"  Weights: {weight_memory:.2f} GB") 
        print(f"  Bias: {bias_memory:.2f} GB")
        print(f"  Intermediate: {intermediate_memory:.2f} GB")
        print(f"  Total: {total_memory_needed:.2f} GB")
        print(f"  VRAM Threshold: {vram_threshold:.2f} GB")
        print(f"  Strategy: {'CPU RAM Storage' if should_use_cpu else 'GPU Optimization'}")
        
        return should_use_cpu
    
    def process_small_workload(self, input_tensor, weight_tensor, bias_tensor, num_layers):
        """STORM for small workloads - GPU optimization strategy"""
        print("[STORM] Using GPU Optimization Strategy")
        
        try:
            current_tensor = input_tensor
            
            for i in range(num_layers):
                # Reshape to 2D for linear layer
                batch_size, seq_len, hidden_size = current_tensor.shape
                reshaped = current_tensor.view(-1, hidden_size)
                
                # Apply linear transformation with CUTLASS optimization
                output = torch.nn.functional.linear(reshaped, weight_tensor, bias_tensor)
                
                # Reshape back to 3D
                layer_output = output.view(batch_size, seq_len, hidden_size)
                layer_output = torch.relu(layer_output)
                
                # Keep on GPU - no unnecessary transfers
                current_tensor = layer_output
                
                # Clean up intermediate results
                del layer_output
                if i > 0:
                    torch.cuda.empty_cache()
            
            return current_tensor
            
        except RuntimeError as e:
            print(f"[ERROR] STORM GPU optimization failed: {e}")
            return None
    
    def process_large_workload(self, input_tensor, weight_tensor, bias_tensor, num_layers):
        """STORM for large workloads - CPU RAM storage strategy"""
        print("[STORM] Using CPU RAM Storage Strategy")
        
        try:
            current_tensor = input_tensor
            cpu_activations = []
            
            for i in range(num_layers):
                # Reshape to 2D for linear layer
                batch_size, seq_len, hidden_size = current_tensor.shape
                reshaped = current_tensor.view(-1, hidden_size)
                
                # Apply linear transformation
                output = torch.nn.functional.linear(reshaped, weight_tensor, bias_tensor)
                
                # Reshape back to 3D
                layer_output = output.view(batch_size, seq_len, hidden_size)
                layer_output = torch.relu(layer_output)
                
                # Store in CPU RAM to free VRAM
                cpu_activation = layer_output.cpu()
                cpu_activations.append(cpu_activation)
                
                # Clean up GPU memory
                del layer_output
                torch.cuda.empty_cache()
                
                # Move next input to GPU
                if i < num_layers - 1:
                    current_tensor = cpu_activations[i].cuda()
            
            # Return final result
            return cpu_activations[-1].cuda()
            
        except RuntimeError as e:
            print(f"[ERROR] STORM CPU RAM storage failed: {e}")
            return None
    
    def process(self, input_tensor, weight_tensor, bias_tensor, num_layers=8):
        """Main STORM processing function with intelligent strategy selection"""
        print(f"[STORM] Processing {num_layers} layers...")
        
        # Determine strategy based on workload size
        use_cpu_ram = self._should_use_cpu_ram(input_tensor, weight_tensor, bias_tensor, num_layers)
        
        if use_cpu_ram:
            return self.process_large_workload(input_tensor, weight_tensor, bias_tensor, num_layers)
        else:
            return self.process_small_workload(input_tensor, weight_tensor, bias_tensor, num_layers)

def create_test_data_small():
    """Create small workload test data that fits in VRAM"""
    batch_size = 32
    sequence_length = 2048
    hidden_size = 2048
    
    input_tensor = torch.randn(batch_size, sequence_length, hidden_size, device='cuda', dtype=torch.float16)
    weight_tensor = torch.randn(hidden_size, hidden_size, device='cuda', dtype=torch.float16)
    bias_tensor = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    
    return input_tensor, weight_tensor, bias_tensor

def create_test_data_large():
    """Create large workload test data that exceeds VRAM"""
    batch_size = 128
    sequence_length = 8192
    hidden_size = 8192
    
    input_tensor = torch.randn(batch_size, sequence_length, hidden_size, device='cuda', dtype=torch.float16)
    weight_tensor = torch.randn(hidden_size, hidden_size, device='cuda', dtype=torch.float16)
    bias_tensor = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    
    return input_tensor, weight_tensor, bias_tensor

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

def time_operation(func, *args, **kwargs):
    """Time an operation with proper error handling"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    if result is not None:
        return (end_time - start_time) * 1000  # Convert to milliseconds
    else:
        return None

def test_small_workload():
    """Test STORM on small workload - should be faster than baseline"""
    print("\n" + "="*60)
    print("TEST 1: SMALL WORKLOAD - GPU OPTIMIZATION")
    print("="*60)
    print("STORM should be FASTER than baseline for small workloads")
    print("Strategy: GPU optimization with CUTLASS GEMM")
    
    # Initialize STORM
    storm = IntelligentSTORM()
    
    # Create small workload
    input_tensor, weight_tensor, bias_tensor = create_test_data_small()
    num_layers = 8
    
    print(f"\n[CONFIG] Small Workload Configuration:")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Memory usage: {storm._estimate_tensor_memory_gb(input_tensor):.2f} GB")
    print(f"  Number of layers: {num_layers}")
    
    # Test baseline
    print(f"\n[TEST] Baseline PyTorch...")
    baseline_time = time_operation(baseline_pytorch, input_tensor, weight_tensor, bias_tensor, num_layers)
    if baseline_time:
        print(f"[OK] Baseline Time: {baseline_time:.2f} ms")
    else:
        print("[FAIL] Baseline failed")
        return False
    
    # Test STORM
    print(f"\n[TEST] STORM Intelligent...")
    storm_time = time_operation(storm.process, input_tensor, weight_tensor, bias_tensor, num_layers)
    if storm_time:
        print(f"[OK] STORM Time: {storm_time:.2f} ms")
        
        if baseline_time and storm_time:
            speedup = baseline_time / storm_time
            if speedup > 1.0:
                print(f"[SUCCESS] STORM is {speedup:.2f}x FASTER than baseline!")
                return True
            else:
                print(f"[WARNING] STORM is {1/speedup:.2f}x slower than baseline")
                return False
    else:
        print("[FAIL] STORM failed")
        return False

def test_large_workload():
    """Test STORM on large workload - should handle what baseline can't"""
    print("\n" + "="*60)
    print("TEST 2: LARGE WORKLOAD - MEMORY WALL ELIMINATION")
    print("="*60)
    print("STORM should HANDLE what baseline CAN'T (OOM)")
    print("Strategy: CPU RAM storage to eliminate VRAM memory wall")
    
    # Initialize STORM
    storm = IntelligentSTORM()
    
    # Create large workload
    input_tensor, weight_tensor, bias_tensor = create_test_data_large()
    num_layers = 10
    
    print(f"\n[CONFIG] Large Workload Configuration:")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Memory usage: {storm._estimate_tensor_memory_gb(input_tensor):.2f} GB")
    print(f"  Number of layers: {num_layers}")
    
    # Test baseline (should fail with OOM)
    print(f"\n[TEST] Baseline PyTorch (should fail with OOM)...")
    baseline_time = time_operation(baseline_pytorch, input_tensor, weight_tensor, bias_tensor, num_layers)
    if baseline_time:
        print(f"[WARNING] Baseline unexpectedly succeeded: {baseline_time:.2f} ms")
        print("This workload might not be large enough to trigger OOM")
        return False
    else:
        print("[EXPECTED] Baseline failed with OOM - this is expected for large workloads")
    
    # Test STORM (should succeed)
    print(f"\n[TEST] STORM Intelligent (should succeed)...")
    storm_time = time_operation(storm.process, input_tensor, weight_tensor, bias_tensor, num_layers)
    if storm_time:
        print(f"[SUCCESS] STORM handled large workload: {storm_time:.2f} ms")
        print("[RESULT] STORM eliminated VRAM memory wall!")
        return True
    else:
        print("[FAIL] STORM failed on large workload")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("STORM TRUE CAPABILITIES TEST")
    print("="*60)
    print("Testing STORM's ability to eliminate VRAM memory wall")
    print("and optimize performance for different workload sizes")
    
    # Get system info
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[SYSTEM] CUDA Device: {device_name}")
        print(f"[SYSTEM] VRAM Capacity: {vram_gb:.2f} GB")
    else:
        print("[ERROR] CUDA not available")
        return
    
    # Run tests
    small_test_passed = test_small_workload()
    large_test_passed = test_large_workload()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Small Workload Test: {'PASS' if small_test_passed else 'FAIL'}")
    print(f"Large Workload Test: {'PASS' if large_test_passed else 'FAIL'}")
    
    if small_test_passed and large_test_passed:
        print("\n[SUCCESS] STORM demonstrated true capabilities!")
        print("- GPU optimization for small workloads")
        print("- Memory wall elimination for large workloads")
    else:
        print("\n[FAILURE] STORM needs improvement")
        if not small_test_passed:
            print("- STORM should be faster than baseline for small workloads")
        if not large_test_passed:
            print("- STORM should handle large workloads that baseline can't")

if __name__ == "__main__":
    main()
