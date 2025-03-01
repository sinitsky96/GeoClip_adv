import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from geoclip_adv_attacks.attacks.attack import Attack

class PGD(Attack):
    """
    Implementation of the Projected Gradient Descent (PGD) attack for GeoCLIP with L0 norm.
    """
    def __init__(self, model, criterion, misc_args, pgd_args):
        """
        Initialize the PGD attack.
        
        Args:
            model: The model to attack
            criterion: The loss function to optimize
            misc_args: Dictionary containing miscellaneous arguments
                - device: Device to use (cuda or cpu)
                - dtype: Data type to use
                - batch_size: Batch size
                - data_shape: Shape of the data [C, H, W]
                - data_RGB_start: Minimum pixel values after normalization
                - data_RGB_end: Maximum pixel values after normalization
                - data_RGB_size: Range of pixel values
                - verbose: Whether to print verbose information
                - report_info: Whether to report information about the attack
            pgd_args: Dictionary containing PGD-specific arguments
                - eps: Number of pixels to modify (L0 budget)
                - n_restarts: Number of restarts
                - n_iter: Number of iterations
                - alpha: Step size
                - rand_init: Whether to use random initialization
        """
        super().__init__(model, criterion, misc_args, pgd_args)
        
        # Set PGD-specific arguments
        self.norm = 'L0'  # Only L0 norm is supported
        self.eps = int(pgd_args.get('eps', 10))  # Number of pixels to modify
        self.n_restarts = pgd_args.get('n_restarts', 1)
        self.n_iter = pgd_args.get('n_iter', 40)
        self.alpha = pgd_args.get('alpha', 1.0)
        self.rand_init = pgd_args.get('rand_init', True)
    
    def report_schematics(self):
        """
        Print information about the attack.
        """
        if not self.report_info:
            return
        
        print("=" * 50)
        print("PGD Attack Parameters:")
        print(f"Norm: {self.norm}")
        print(f"Epsilon: {self.eps}")
        print(f"Number of restarts: {self.n_restarts}")
        print(f"Number of iterations: {self.n_iter}")
        print(f"Step size: {self.alpha}")
        print(f"Random initialization: {self.rand_init}")
        print(f"Batch size: {self.batch_size}")
        print(f"Data shape: {self.data_shape}")
        print(f"Device: {self.device}")
        print("=" * 50)
    
    def perturb(self, X, y):
        """
        Perturb the input X to maximize the loss using L0 norm.
        
        Args:
            X: Input data
            y: Target data
            
        Returns:
            Perturbed input
        """
        # Initialize best perturbation and loss
        best_loss = -float('inf') * torch.ones(X.shape[0], device=self.device)
        best_delta = torch.zeros_like(X)
        
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Multiple restarts for finding the best perturbation
        for i_restart in range(self.n_restarts):
            # Initialize perturbation
            delta = torch.zeros_like(X, requires_grad=True)
            
            # Random initialization if specified
            if self.rand_init:
                # Randomly select pixels to modify
                batch_size = X.shape[0]
                n_features = int(np.prod(X.shape[1:]))  # Total number of features (C*H*W)
                
                # For each sample in the batch, randomly select eps pixels
                for i in range(batch_size):
                    perm = torch.randperm(n_features)[:self.eps]
                    delta.data[i].view(-1)[perm] = torch.randn(self.eps, device=self.device)
            
            # Project perturbation to ensure it's within the valid range
            min_val = torch.as_tensor(self.data_RGB_start, device=self.device).clone().detach()
            max_val = torch.as_tensor(self.data_RGB_end, device=self.device).clone().detach()
            delta.data = torch.clamp(X + delta.data, min=min_val, max=max_val) - X
            
            # Keep track of which pixels can be modified
            pixel_mask = (delta.data != 0).float()
            
            # Optimization loop
            for i_iter in range(self.n_iter):
                # Zero gradients
                if delta.grad is not None:
                    delta.grad.zero_()
                
                # Forward pass
                loss = self.criterion(X + delta)
                
                # Backward pass
                loss.sum().backward()
                
                # Update perturbation
                with torch.no_grad():
                    # Get the gradient magnitude for each pixel
                    grad_abs = torch.abs(delta.grad.view(delta.shape[0], -1))
                    
                    # For each sample, update only the top-k pixels by gradient magnitude
                    # where k is the number of non-zero pixels in the current perturbation
                    for i in range(X.shape[0]):
                        n_active = int(pixel_mask[i].sum().item())
                        if n_active > 0:
                            # Get indices of current non-zero pixels
                            active_indices = torch.nonzero(pixel_mask[i].view(-1)).squeeze()
                            
                            # Update these pixels
                            delta.data[i].view(-1)[active_indices] += self.alpha * torch.sign(delta.grad[i].view(-1)[active_indices])
                
                # Project perturbation to ensure the perturbed input is within the valid range
                min_val = torch.as_tensor(self.data_RGB_start, device=self.device).clone().detach()
                max_val = torch.as_tensor(self.data_RGB_end, device=self.device).clone().detach()
                delta.data = torch.clamp(X + delta.data, min=min_val, max=max_val) - X
                
                # Ensure we maintain the L0 constraint by keeping only the top-k pixels
                delta_abs = torch.abs(delta.data.view(delta.shape[0], -1))
                for i in range(X.shape[0]):
                    if delta_abs[i].sum() > 0:  # If there are any non-zero perturbations
                        _, top_indices = torch.topk(delta_abs[i], min(self.eps, delta_abs[i].shape[0]))
                        mask = torch.zeros_like(delta_abs[i])
                        mask[top_indices] = 1
                        delta.data[i] = (delta.data[i].view(-1) * mask).view(delta.data[i].shape)
                        pixel_mask[i] = (mask > 0).float().view(pixel_mask[i].shape)
            
            # Compute final loss for this restart
            with torch.no_grad():
                final_loss = self.criterion(X + delta)
            
            # Update best perturbation if the current one is better
            idx_improved = final_loss > best_loss
            if idx_improved.any():
                best_loss[idx_improved] = final_loss[idx_improved]
                best_delta[idx_improved] = delta.data[idx_improved]
        
        # Return perturbed input
        return X + best_delta 