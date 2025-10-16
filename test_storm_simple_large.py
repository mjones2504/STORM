#!/usr/bin/env python3
"""
Simple STORM Large Workload Test
================================

This test bypasses the complex CPU transfer logic and focuses on the core issue:
Why does storm_linear produce 1/8th the expected elements on large tensors?
"""

import torch
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import storm_cuda
    print("[OK] STORM CUDA extension loaded successfully!")
except ImportError as e:
    print(f"[ERROR] Failed to import STORM CUDA extension: {e}")
    sys.exit(1)

def test_simple_large_workload():
    """Test STORM on large workload without CPU transfers"""
    print("="*60)
    print("SIMPLE STORM LARGE WORKLOAD TEST")
    print("="*60)
    
    # Create large workload (same as the failing test)
    batch_size = 64
    sequence_length = 4096
    hidden_size = 4096
    
    print(f"[CONFIG] Large Workload Configuration:")
    print(f"  Dimensions: {batch_size}x{sequence_length}x{hidden_size}")
    print(f"  Expected elements: {batch_size * sequence_length * hidden_size:,}")
    
    # Create tensors
    input_tensor = torch.randn(batch_size, sequence_length, hidden_size, 
                               device='cuda', dtype=torch.float16)
    weight_tensor = torch.randn(hidden_size, hidden_size, 
                                device='cuda', dtype=torch.float16)
    bias_tensor = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    
    print(f"[INPUT] Input tensor shape: {input_tensor.shape}")
    print(f"[INPUT] Input tensor elements: {input_tensor.numel():,}")
    
    # Reshape for linear layer
    reshaped = input_tensor.view(-1, hidden_size)
    print(f"[RESHAPED] Reshaped input shape: {reshaped.shape}")
    print(f"[RESHAPED] Reshaped input elements: {reshaped.numel():,}")
    
    # Test PyTorch linear (baseline)
    print(f"\n[TEST] PyTorch Linear (Baseline)...")
    pytorch_output = torch.nn.functional.linear(reshaped, weight_tensor, bias_tensor)
    print(f"[PYTORCH] Output shape: {pytorch_output.shape}")
    print(f"[PYTORCH] Output elements: {pytorch_output.numel():,}")
    
    # Test STORM linear (the problematic call)
    print(f"\n[TEST] STORM Linear (This is where it fails)...")
    try:
        storm_output = storm_cuda.storm.StormGEMMTensor.storm_linear(
            reshaped, weight_tensor, bias_tensor, layer_id=0)
        print(f"[STORM] Output shape: {storm_output.shape}")
        print(f"[STORM] Output elements: {storm_output.numel():,}")
        
        # Check if sizes match
        if pytorch_output.shape == storm_output.shape:
            print("✅ STORM output shape matches PyTorch!")
        else:
            print("❌ STORM output shape does NOT match PyTorch!")
            print(f"  PyTorch: {pytorch_output.shape}")
            print(f"  STORM:   {storm_output.shape}")
            
        # Check element counts
        if pytorch_output.numel() == storm_output.numel():
            print("✅ STORM output element count matches PyTorch!")
        else:
            print("❌ STORM output element count does NOT match PyTorch!")
            print(f"  PyTorch elements: {pytorch_output.numel():,}")
            print(f"  STORM elements:   {storm_output.numel():,}")
            print(f"  Ratio: {pytorch_output.numel() / storm_output.numel():.1f}x")
            
        # Test reshaping back to 3D
        print(f"\n[TEST] Reshaping back to 3D...")
        try:
            pytorch_3d = pytorch_output.view(batch_size, sequence_length, hidden_size)
            storm_3d = storm_output.view(batch_size, sequence_length, hidden_size)
            
            print(f"[PYTORCH 3D] Shape: {pytorch_3d.shape}")
            print(f"[STORM 3D] Shape: {storm_3d.shape}")
            
            if pytorch_3d.shape == storm_3d.shape:
                print("✅ 3D reshaping successful for both!")
            else:
                print("❌ 3D reshaping failed!")
                
        except Exception as e:
            print(f"[ERROR] 3D reshaping failed: {e}")
            print(f"  This is the exact error from the large workload test!")
            
    except Exception as e:
        print(f"[ERROR] STORM linear failed: {e}")
        return

if __name__ == "__main__":
    test_simple_large_workload()
