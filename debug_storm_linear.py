#!/usr/bin/env python3
"""
Debug STORM Linear Function
==========================

This script isolates the issue with storm_linear producing incorrect output sizes.
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

def debug_storm_linear():
    """Debug the storm_linear function to identify the size issue"""
    print("="*60)
    print("DEBUG STORM LINEAR FUNCTION")
    print("="*60)
    
    # Create test tensors with known sizes
    batch_size = 64
    seq_len = 4096
    hidden_size = 4096
    
    print(f"[CONFIG] Test Configuration:")
    print(f"  Input shape: [{batch_size}, {seq_len}, {hidden_size}]")
    print(f"  Expected elements: {batch_size * seq_len * hidden_size:,}")
    
    # Create input tensor
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    print(f"[INPUT] Input tensor shape: {input_tensor.shape}")
    print(f"[INPUT] Input tensor elements: {input_tensor.numel():,}")
    
    # Create weight and bias tensors
    weight_tensor = torch.randn(hidden_size, hidden_size, device='cuda', dtype=torch.float16)
    bias_tensor = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    
    print(f"[WEIGHT] Weight tensor shape: {weight_tensor.shape}")
    print(f"[BIAS] Bias tensor shape: {bias_tensor.shape}")
    
    # Reshape input for linear layer
    reshaped = input_tensor.view(-1, hidden_size)
    print(f"[RESHAPED] Reshaped input shape: {reshaped.shape}")
    print(f"[RESHAPED] Reshaped input elements: {reshaped.numel():,}")
    
    # Test PyTorch linear (baseline)
    print(f"\n[TEST] PyTorch Linear (Baseline)...")
    pytorch_output = torch.nn.functional.linear(reshaped, weight_tensor, bias_tensor)
    print(f"[PYTORCH] Output shape: {pytorch_output.shape}")
    print(f"[PYTORCH] Output elements: {pytorch_output.numel():,}")
    
    # Test STORM linear
    print(f"\n[TEST] STORM Linear...")
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
            
    except Exception as e:
        print(f"[ERROR] STORM linear failed: {e}")
        return
    
    # Test reshaping back to 3D
    print(f"\n[TEST] Reshaping back to 3D...")
    try:
        pytorch_3d = pytorch_output.view(batch_size, seq_len, hidden_size)
        storm_3d = storm_output.view(batch_size, seq_len, hidden_size)
        
        print(f"[PYTORCH 3D] Shape: {pytorch_3d.shape}")
        print(f"[STORM 3D] Shape: {storm_3d.shape}")
        
        if pytorch_3d.shape == storm_3d.shape:
            print("✅ 3D reshaping successful for both!")
        else:
            print("❌ 3D reshaping failed!")
            
    except Exception as e:
        print(f"[ERROR] 3D reshaping failed: {e}")
        print(f"  This is likely the source of the large workload test failure!")

if __name__ == "__main__":
    debug_storm_linear()
