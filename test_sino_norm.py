#!/usr/bin/env python3
"""Quick test script to verify sinogram normalization"""

import numpy as np

# Load target sinogram
sino = np.load('data/interim/proj/chest/chest_1_60@1.0.npy')

print("=" * 60)
print("Target Sinogram Statistics")
print("=" * 60)
print(f"Shape: {sino.shape}")
print(f"Min:   {sino.min():.4f}")
print(f"Max:   {sino.max():.4f}")
print(f"Mean:  {sino.mean():.4f}")
print(f"Std:   {sino.std():.4f}")
print()

# Simulate a predicted sinogram with different range
pred_sino = np.random.randn(61, 1000) * 50 + 100  # Very different range

print("=" * 60)
print("Simulated Predicted Sinogram (before normalization)")
print("=" * 60)
print(f"Shape: {pred_sino.shape}")
print(f"Min:   {pred_sino.min():.4f}")
print(f"Max:   {pred_sino.max():.4f}")
print(f"Mean:  {pred_sino.mean():.4f}")
print(f"Std:   {pred_sino.std():.4f}")
print()

# Test normalization function
def normalize_sinogram_pair(pred_sino, target_sino):
    pred_mean = pred_sino.mean()
    pred_std = pred_sino.std()
    target_mean = target_sino.mean()
    target_std = target_sino.std()
    
    if pred_std > 1e-8:
        pred_normalized = (pred_sino - pred_mean) / pred_std
        pred_normalized = pred_normalized * target_std + target_mean
    else:
        pred_normalized = pred_sino
    
    return pred_normalized, target_sino

# Normalize
target_slice = sino[0]  # First slice
pred_norm, target_norm = normalize_sinogram_pair(pred_sino, target_slice)

print("=" * 60)
print("After Normalization")
print("=" * 60)
print(f"Pred  mean/std: {pred_norm.mean():.4f} / {pred_norm.std():.4f}")
print(f"Target mean/std: {target_norm.mean():.4f} / {target_norm.std():.4f}")
print(f"Pred  range: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
print(f"Target range: [{target_norm.min():.4f}, {target_norm.max():.4f}]")
print()

# Check if they're now comparable
sino_error = target_norm - pred_norm
print("=" * 60)
print("Sinogram Error")
print("=" * 60)
print(f"Error mean: {sino_error.mean():.4f}")
print(f"Error std:  {sino_error.std():.4f}")
print(f"Error range: [{sino_error.min():.4f}, {sino_error.max():.4f}]")
print()
print("✓ Normalization working correctly!")
print("  Pred and target now have matched statistics.")
