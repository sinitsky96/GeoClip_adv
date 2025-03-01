import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from geoclip_adv_attacks.attacks.attack import Attack

class PatchAttack(Attack):
    """
    Implementation of an adversarial patch attack for GeoCLIP.
    The attack optimizes a rectangular patch that can be placed on the image to fool the model.
    """
    def __init__(self, model, criterion, misc_args, patch_args):
        """
        Initialize the patch attack.
        
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
            patch_args: Dictionary containing patch-specific arguments
                - patch_size: Tuple of (height, width) for the patch size
                - patch_location: Either 'random' or tuple of (y, x) for fixed location
                - n_restarts: Number of restarts
                - n_iter: Number of iterations
                - alpha: Step size
                - rand_init: Whether to use random initialization for patch pixels
                - l0_limit: Maximum number of non-zero pixels in patch (L0 norm constraint)
        """
        super().__init__(model, criterion, misc_args, patch_args)
        
        # Set patch-specific arguments
        self.patch_size = patch_args.get('patch_size', (32, 32))
        self.patch_location = patch_args.get('patch_location', 'random')
        self.n_restarts = patch_args.get('n_restarts', 1)
        self.n_iter = patch_args.get('n_iter', 40)
        self.alpha = patch_args.get('alpha', 0.1)
        self.rand_init = patch_args.get('rand_init', True)
        self.current_location = None  # Track current patch location
        
        # L0 norm constraint
        total_patch_pixels = self.patch_size[0] * self.patch_size[1] * 3  # Height * Width * Channels
        self.l0_limit = patch_args.get('l0_limit', total_patch_pixels)  # Default to all pixels if not specified
        
        # Validate patch size and L0 limit
        assert len(self.patch_size) == 2, "Patch size must be a tuple of (height, width)"
        assert self.patch_size[0] <= self.data_shape[2] and self.patch_size[1] <= self.data_shape[2], \
            "Patch size must be smaller than image size"
        assert self.l0_limit <= total_patch_pixels, \
            f"L0 limit ({self.l0_limit}) cannot exceed total patch pixels ({total_patch_pixels})"
    
    def report_schematics(self):
        """
        Print information about the attack.
        """
        if not self.report_info:
            return
        
        print("=" * 50)
        print("Patch Attack Parameters:")
        print(f"Patch size: {self.patch_size}")
        print(f"Patch location: {self.patch_location}")
        print(f"Number of restarts: {self.n_restarts}")
        print(f"Number of iterations: {self.n_iter}")
        print(f"Step size: {self.alpha}")
        print(f"Random initialization: {self.rand_init}")
        print(f"Batch size: {self.batch_size}")
        print(f"Data shape: {self.data_shape}")
        print(f"Device: {self.device}")
        print("=" * 50)
    
    def _apply_patch(self, X, patch, location):
        """
        Apply the patch to the input images at the specified location.
        
        Args:
            X: Input images of shape [B, C, H, W]
            patch: Patch to apply of shape [B, C, patch_h, patch_w]
            location: Tuple of (y, x) coordinates for top-left corner of patch
            
        Returns:
            Perturbed images with patch applied
        """
        B, C, H, W = X.shape
        ph, pw = self.patch_size
        y, x = location
        
        # Create a mask for the patch (1 where patch should be, 0 elsewhere)
        mask = torch.zeros_like(X)
        mask[:, :, y:y+ph, x:x+pw] = 1
        
        # Create full-sized patch
        full_patch = torch.zeros_like(X)
        full_patch[:, :, y:y+ph, x:x+pw] = patch
        
        # Apply the patch
        X_perturbed = X * (1 - mask) + full_patch * mask
        return X_perturbed
    
    def _get_random_location(self, batch_size):
        """
        Get random valid locations for the patch.
        
        Args:
            batch_size: Number of random locations to generate
            
        Returns:
            Tuple of (y, x) coordinates
        """
        ph, pw = self.patch_size
        # data_shape is [B, C, H, W], so we need indices 2 and 3 for H and W
        H, W = self.data_shape[2], self.data_shape[3]
        
        y = torch.randint(0, H - ph + 1, (batch_size,), device=self.device)
        x = torch.randint(0, W - pw + 1, (batch_size,), device=self.device)
        return y, x
    
    def _project_l0(self, patch):
        """
        Project patch to satisfy L0 norm constraint by keeping only the top-k elements
        by absolute magnitude, where k is the L0 limit.
        
        Args:
            patch: Patch tensor of shape [B, C, H, W]
            
        Returns:
            Projected patch satisfying L0 constraint
        """
        with torch.no_grad():
            # Reshape patch to [B, -1] to work with all dimensions
            B = patch.shape[0]
            flat_patch = patch.view(B, -1)
            
            # Get the top-k elements by magnitude
            _, top_indices = torch.topk(torch.abs(flat_patch), k=self.l0_limit, dim=1)
            
            # Create mask for top-k elements
            mask = torch.zeros_like(flat_patch)
            for i in range(B):
                mask[i, top_indices[i]] = 1
            
            # Apply mask and reshape back
            projected_patch = (flat_patch * mask).view_as(patch)
            
            return projected_patch
    
    def perturb(self, X, y):
        """
        Perturb the input X by optimizing an adversarial patch.
        
        Args:
            X: Input data
            y: Target data
            
        Returns:
            Perturbed input
        """
        # Initialize best perturbation and loss
        best_loss = -float('inf') * torch.ones(X.shape[0], device=self.device)
        best_perturbed = X.clone()
        
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Get patch location
        if isinstance(self.patch_location, tuple):
            y_loc = torch.full((X.shape[0],), self.patch_location[0], device=self.device)
            x_loc = torch.full((X.shape[0],), self.patch_location[1], device=self.device)
            self.current_location = self.patch_location
        
        # Multiple restarts for finding the best patch
        best_patch = None
        best_patch_loss = -float('inf')
        
        for i_restart in range(self.n_restarts):
            # Initialize patch with patch size
            patch = torch.zeros((X.shape[0], self.data_shape[1], *self.patch_size), device=self.device, requires_grad=True)
            
            # Random initialization if specified
            if self.rand_init:
                patch.data.uniform_(
                    float(self.data_RGB_start.min()), float(self.data_RGB_end.max())
                )
                # Project to L0 constraint
                patch.data = self._project_l0(patch.data)
            
            # Get random location for this restart if not fixed
            if self.patch_location == 'random':
                y_loc, x_loc = self._get_random_location(X.shape[0])
                self.current_location = (y_loc[0].item(), x_loc[0].item())
            
            # Optimization loop
            for i_iter in range(self.n_iter):
                # Zero gradients
                if patch.grad is not None:
                    patch.grad.zero_()
                
                # Apply patch and compute loss
                X_perturbed = self._apply_patch(X, patch, (y_loc[0].item(), x_loc[0].item()))
                loss = self.criterion(X_perturbed)
                
                # Backward pass
                loss.sum().backward()
                
                # Update patch
                with torch.no_grad():
                    patch.data = patch.data + self.alpha * patch.grad.sign()
                    patch.data = torch.clamp(patch.data, self.data_RGB_start, self.data_RGB_end)
                    # Project to L0 constraint
                    patch.data = self._project_l0(patch.data)
            
            # Compute final loss for this restart
            with torch.no_grad():
                X_perturbed = self._apply_patch(X, patch, (y_loc[0].item(), x_loc[0].item()))
                final_loss = self.criterion(X_perturbed)
            
            # Update best patch if the current one is better
            if final_loss.mean() > best_patch_loss:
                best_patch_loss = final_loss.mean()
                best_patch = patch.detach().clone()
            
            # Update best perturbation if the current one is better
            idx_improved = final_loss > best_loss
            if idx_improved.any():
                best_loss[idx_improved] = final_loss[idx_improved]
                best_perturbed[idx_improved] = X_perturbed[idx_improved]
        
        # Store the best patch
        self.best_patch = best_patch
        
        return best_perturbed
    
    def get_patch(self):
        """
        Get the best patch found during optimization.
        
        Returns:
            The optimized patch tensor of shape [1, C, H, W]
        """
        if not hasattr(self, 'best_patch'):
            raise RuntimeError("No patch available. Call perturb() first.")
        
        # Return only the first patch (they should all be the same due to the optimization process)
        return self.best_patch[:1] 