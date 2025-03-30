#!/usr/bin/env python
import os
import sys
import torch
import traceback

# Add project root to path so we can import modules
sys.path.append('/home/sinitsky96/project/GeoClip_adv')

# Monkey patch torch.Tensor.__bool__ to help trace where the boolean ambiguity error occurs
original_bool = torch.Tensor.__bool__

def traced_bool(self):
    if self.numel() > 1:
        stack = traceback.extract_stack()
        # Skip the current frame
        caller = stack[-2]
        print(f"Boolean tensor ambiguity at: {caller.filename}:{caller.lineno}")
        print(f"Called from: {caller.name}")
        print(f"Tensor shape: {self.shape}, dtype: {self.dtype}")
        
        # Optionally raise the error to stop execution immediately
        # raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
    
    return original_bool(self)

# Install the monkey patch
torch.Tensor.__bool__ = traced_bool

# Import attack classes after patching
from SparsePatches.attacks.pgd_attacks.PGDTrim import PGDTrim
from SparsePatches.attacks.pgd_attacks.PGDTrimKernel import PGDTrimKernel
from SparsePatches.attack_sparse_patches import AttackGeoCLIP_SparsePatches_Kernel

# Create a simplified test case
def run_test():
    print("Testing tensor boolean operations in PGDTrim...")
    
    # Create a dummy model and criterion
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            # Return a tuple of two 2D tensors (lon, lat) for each image
            batch_size = x.size(0)
            return torch.randn(batch_size, 2)
    
    class DummyCriterion:
        def __call__(self, pred, target):
            # Simple MSE loss
            return ((pred - target) ** 2).sum(dim=1)
    
    # Set up test parameters
    batch_size = 2
    image_size = 32  # Smaller for testing
    kernel_size = 4
    
    # Initialize the model and criterion
    model = DummyModel()
    criterion = DummyCriterion()
    
    # Configure necessary arguments
    misc_args = {
        'data_shape': [3, image_size, image_size],
        'batch_size': batch_size,
        'device': 'cpu'  # Use CPU for testing
    }
    
    kernel_args = {
        'kernel_size': kernel_size,
        'kernel_sparsity': 8,
        'max_kernel_sparsity': 16
    }
    
    pgd_args = {
        'w_iter_ratio': 0.5,
        'n_iter': 2,  # Small number for testing
        'n_restarts': 1
    }
    
    trim_args = {
        'sparsity': 16
    }
    
    # Create a dummy input and target
    x = torch.rand(batch_size, 3, image_size, image_size)
    y = torch.rand(batch_size, 2)  # Random GPS coordinates
    
    try:
        # Initialize attack wrappers
        print("Initializing PGDTrimKernel...")
        pgd_trim_kernel = PGDTrimKernel(
            model=model,
            criterion=criterion,
            misc_args=misc_args,
            pgd_args=pgd_args,
            kernel_args=kernel_args,
            trim_args=trim_args
        )
        
        # Initialize the GeoCLIP attack wrapper
        print("Initializing AttackGeoCLIP_SparsePatches_Kernel...")
        attack = AttackGeoCLIP_SparsePatches_Kernel(
            model=model,
            kernel_size=kernel_size,
            kernel_sparsity=8,
            pgd_args=pgd_args,
            misc_args=misc_args
        )
        
        # Run the attack
        print("Running attack.perturb()...")
        pert_x = attack.perturb(x, y)
        
        print("Attack completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()