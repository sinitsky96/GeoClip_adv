import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from geoclip_adv_attacks.attacks.attack import Attack

class UPGD(Attack):
    """
    Implementation of the Universal Projected Gradient Descent (UPGD) attack for GeoCLIP.
    This attack generates a single perturbation that can be applied to multiple images.
    """
    def __init__(self, model, criterion, misc_args, pgd_args):
        """
        Initialize the UPGD attack.
        
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
                - norm: Norm to use for the attack (L1, L2, Linf)
                - eps: Epsilon for the attack
                - n_restarts: Number of restarts
                - n_iter: Number of iterations
                - alpha: Step size
                - rand_init: Whether to use random initialization
        """
        super().__init__(model, criterion, misc_args, pgd_args)
        
        # Set PGD-specific arguments
        self.norm = pgd_args.get('norm', 'Linf')
        self.eps = pgd_args.get('eps', 0.03)
        self.n_restarts = pgd_args.get('n_restarts', 1)
        self.n_iter = pgd_args.get('n_iter', 40)
        self.alpha = pgd_args.get('alpha', 0.01)
        self.rand_init = pgd_args.get('rand_init', True)
        
        # Validate norm
        assert self.norm in ['L1', 'L2', 'Linf'], f"Norm {self.norm} not supported"
    
    def report_schematics(self):
        """
        Print information about the attack.
        """
        if not self.report_info:
            return
        
        print("=" * 50)
        print("Universal PGD Attack Parameters:")
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
    
    def perturb(self, X_list, y_list):
        """
        Generate a universal perturbation for a list of inputs.
        
        Args:
            X_list: List of input data batches
            y_list: List of target data batches
            
        Returns:
            Universal perturbation that can be applied to any input
        """
        # Initialize best perturbation and loss
        best_loss = -float('inf')
        best_delta = torch.zeros(self.data_shape, device=self.device)
        
        # Multiple restarts for finding the best perturbation
        for i_restart in range(self.n_restarts):
            # Initialize perturbation
            delta = torch.zeros(self.data_shape, device=self.device, requires_grad=True)
            
            # Random initialization if specified
            if self.rand_init:
                if self.norm == 'Linf':
                    delta.data.uniform_(-self.eps, self.eps)
                elif self.norm == 'L2':
                    delta.data.normal_()
                    delta_norm = torch.norm(delta.data.view(-1))
                    delta.data = delta.data / delta_norm * self.eps
                elif self.norm == 'L1':
                    delta.data.uniform_(-1, 1)
                    delta.data = delta.data / torch.norm(delta.data.view(-1), p=1) * self.eps
            
            # Optimization loop
            for i_iter in range(self.n_iter):
                # Zero gradients
                if delta.grad is not None:
                    delta.grad.zero_()
                
                # Compute total loss across all batches
                total_loss = 0
                for X, y in zip(X_list, y_list):
                    # Move data to device
                    X = X.to(self.device)
                    y = y.to(self.device)
                    
                    # Expand delta to match batch size
                    delta_expanded = delta.unsqueeze(0).expand(X.shape)
                    
                    # Project perturbation to ensure the perturbed input is within the valid range
                    delta_projected = torch.clamp(X + delta_expanded, 
                                                min=torch.tensor(self.data_RGB_start, device=self.device),
                                                max=torch.tensor(self.data_RGB_end, device=self.device)) - X
                    
                    # Compute loss
                    loss = self.criterion(X + delta_projected)
                    total_loss += loss.sum()
                
                # Backward pass
                total_loss.backward()
                
                # Update perturbation
                with torch.no_grad():
                    if self.norm == 'Linf':
                        delta.data = delta.data + self.alpha * delta.grad.sign()
                        delta.data = torch.clamp(delta.data, -self.eps, self.eps)
                    elif self.norm == 'L2':
                        grad_norm = torch.norm(delta.grad.view(-1))
                        delta.data = delta.data + self.alpha * delta.grad / (grad_norm + 1e-8)
                        delta_norm = torch.norm(delta.data.view(-1))
                        delta.data = delta.data / torch.clamp(delta_norm / self.eps, min=1.0)
                    elif self.norm == 'L1':
                        grad_abs = torch.abs(delta.grad.view(-1))
                        grad_max, grad_max_idx = torch.max(grad_abs, dim=0)
                        delta.data.view(-1)[grad_max_idx] += self.alpha * torch.sign(delta.grad.view(-1)[grad_max_idx])
                        delta.data = delta.data / torch.clamp(torch.norm(delta.data.view(-1), p=1) / self.eps, min=1.0)
            
            # Compute final loss for this restart
            with torch.no_grad():
                final_loss = 0
                for X, y in zip(X_list, y_list):
                    # Move data to device
                    X = X.to(self.device)
                    y = y.to(self.device)
                    
                    # Expand delta to match batch size
                    delta_expanded = delta.unsqueeze(0).expand(X.shape)
                    
                    # Project perturbation to ensure the perturbed input is within the valid range
                    delta_projected = torch.clamp(X + delta_expanded, 
                                                min=torch.tensor(self.data_RGB_start, device=self.device),
                                                max=torch.tensor(self.data_RGB_end, device=self.device)) - X
                    
                    # Compute loss
                    loss = self.criterion(X + delta_projected)
                    final_loss += loss.sum()
            
            # Update best perturbation if the current one is better
            if final_loss > best_loss:
                best_loss = final_loss
                best_delta = delta.data.clone()
        
        # Return universal perturbation
        return best_delta 