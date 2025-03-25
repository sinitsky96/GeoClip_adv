import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image

from attacks.pgd_attacks.PGDTrim import PGDTrim
from attacks.geoattack import GeoAttackPGDTrim, GeoCLIPPredictor, GeoLocationLoss, haversine_distance, GeoAttackPGDTrimKernel
from sparse_rs.util import CONTINENT_R, STREET_R
from data.Im2GPS3k.download import load_places365_categories
from transformers import CLIPProcessor

def find_nearest_neighbor_index(gps_gallery, coord):
    """Find the index of the closest GPS coordinate in the gallery"""
    # Ensure both are on the same device
    device = gps_gallery.device
    coord = coord.to(device)
    distances = haversine_distance(gps_gallery, coord.unsqueeze(0))  # shape: (N,)
    nn_index = torch.argmin(distances).item()
    return nn_index

def coords_to_class_indices_nn(gps_gallery, coords):
    """Convert GPS coordinates to class indices based on nearest neighbor"""
    # Ensure both are on the same device
    device = gps_gallery.device
    coords = coords.to(device)
    
    # coords: (B, 2)
    label_indices = []
    for i in range(coords.shape[0]):
        index = find_nearest_neighbor_index(gps_gallery, coords[i])
        label_indices.append(index)
    return torch.LongTensor(label_indices).to(device)

class ClipWrap:
    """
    Wrapper for CLIP model to make it compatible with the attack interface.
    Exactly matches the implementation from sparse_rs/attack_sparse_rs.py
    """
    def __init__(self, model, data_path, device):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",
                                                      do_rescale=False,
                                                      do_resize=False,
                                                      do_center_crop=False,
                                                      do_normalize=False
                                                     )
        self.prompts = load_places365_categories(os.path.join(data_path, 'places365_cat.txt'))
        self.device = device
        self.model = model

    def __call__(self, x):
        return self.get_logits(x)

    def get_logits(self, x):
        inputs = self.processor(images=x,               
                                text=self.prompts,       
                                return_tensors="pt",
                                padding=True).to(self.device)
        
        # Use torch.no_grad() for evaluation, not during attack
        if not x.requires_grad:
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        return outputs.logits_per_image

class AttackGeoCLIP_SparsePatches:
    """
    A PGDTrim attack class for GeoCLIP that uses a geodesic loss based on the top-k predictions.
    This attack leverages the SparsePatches PGDTrim method to create sparse adversarial perturbations.
    
    For untargeted attacks, the goal is to push the predicted GPS coordinates far from the ground-truth.
    For targeted attacks, the goal is to make the model predict a specific target location.
    """
    def __init__(self, model, norm='L0', sparsity=224, eps_l_inf=0.03, n_iter=40, 
                 n_restarts=1, targeted=False, loss='margin', top_k=5, device='cuda',
                 verbose=True, seed=42, constant_schedule=False,
                 data_loader=None, resample_loc=None, log_path=None):
        self.model = model
        self.device = device
        self.targeted = targeted
        self.loss_type = loss
        self.verbose = verbose
        self.seed = seed
        self.log_path = log_path
        self.logger = Logger(log_path) if log_path is not None else None
        self.sparsity = sparsity
        
        # Create the GeoCLIP predictor wrapper
        self.geoclip_predictor = GeoCLIPPredictor(model)
        
        # Create the GeoLocation loss function
        self.geo_loss = GeoLocationLoss(
            predictor=self.geoclip_predictor,
            top_k=top_k,
            targeted=targeted
        )
        
        # Calculate total image pixels and trim steps according to the paper
        total_pixels = 224 * 224  # Only count spatial dimensions (224x224)
        trim_steps = [
            int(total_pixels / 8),    # 12.5% of pixels
            int(total_pixels / 16),   # 6.25% of pixels
            int(total_pixels / 32),   # 3.125% of pixels
            int(total_pixels / 64),   # 1.5% of pixels
            sparsity                  # Final target sparsity
        ]
        
        # Log initial setup
        if self.verbose:
            self.log(f"Initializing {self.__class__.__name__} with:")
            self.log(f"  - Target sparsity: {sparsity} pixels")
            self.log(f"  - L∞ constraint: {eps_l_inf}")
            self.log(f"  - Iterations: {n_iter}")
            self.log(f"  - Restarts: {n_restarts}")
            self.log(f"  - Loss type: {loss}")
            self.log(f"  - Targeted: {targeted}")
            self.log(f"  - Trim steps: {trim_steps}")
        
        misc_args = {
            'device': device,
            'n_restarts': n_restarts,
            'report_info': True,
            'verbose': verbose,
            'seed': seed,
            'dtype': torch.float32,
            'batch_size': 32,
            'data_shape': [3, 224, 224],  # Image shape [channels, width, height]
            'data_RGB_start': [0.0, 0.0, 0.0],  # Min RGB values
            'data_RGB_end': [1.0, 1.0, 1.0],    # Max RGB values
            'data_RGB_size': [1.0, 1.0, 1.0],    # Range of RGB values
            'targeted': targeted  # Pass targeted flag to the attack
        }
        
        pgd_args = {
            'eps_ratio': eps_l_inf,
            'eps': eps_l_inf,  # L_inf constraint
            'norm': norm,
            'n_iter': n_iter,
            'alpha': eps_l_inf * 0.5 if eps_l_inf > 0 else 0.0,  # Increased step size (alpha) for faster convergence
            'alpha_ratio': eps_l_inf * 0.5 if eps_l_inf > 0 else 0.0,  # Increased alpha ratio as well
            'restarts_interval': 1,
            'w_iter_ratio': 1.0,
            'n_restarts': n_restarts,
            'rand_init': True
        }
        
        dropout_args = {
            'dpo_mu': 0.0,
            'dpo_sigma': 0.0,
            'dpo_mu_sched': 0.0,
            'dpo_sigma_sched': 0.0,
            'dropout_mean': 0.0,
            'dropout_std': 0.0,
            'dropout_dist': 'none',
            'dropout_std_bernoulli': False
        }
        
        trim_args = {
            'sparsity': sparsity,
            'trim_steps': trim_steps,
            'max_trim_steps': len(trim_steps),
            'trim_steps_reduce': 'none',
            'scale_dpo_mean': True,
            'post_trim_dpo': True,
            'dynamic_trim': True,
            'l0_methods': ['full'],
            'sparsity_distribution': 'constant',
            'trim_with_mask': 'single'
        }
        
        mask_args = {
            'mask_dist': 'topk',
            'mask_prob_amp_rate': 1.0,
            'norm_mask_amp': True,
            'mask_opt_iter': 10,
            'n_mask_samples': 1,
            'sample_all_masks': False,
            'trim_best_mask': True
        }
        
        kernel_args = None  # Not using kernel attacks
                
        # Create the GeoAttackPGDTrim instance
        self.attack = GeoAttackPGDTrim(
            model=model,
            criterion=self.compute_loss,
            misc_args=misc_args,
            pgd_args=pgd_args,
            dropout_args=dropout_args,
            trim_args=trim_args,
            mask_args=mask_args,
            kernel_args=kernel_args
        )
    
    def compute_loss(self, x, y):
        """
        Compute the loss for the attack based on the specified loss type
        """
        if self.loss_type == 'margin':
            return self.compute_margin_loss(x, y)
        elif self.loss_type == 'ce':
            return self.compute_ce_loss(x, y)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def compute_margin_loss(self, x, y):
        """
        Compute margin loss based on haversine distance
        """
        output, _ = self.model.predict_from_tensor(x)
        # Ensure y is on the same device as output
        y = y.to(output.device)
        distance = haversine_distance(output, y)
        
        if not self.targeted:
            margin = torch.sub(CONTINENT_R, distance)
            loss = -margin  # maximize distance (minimize negative distance)
        else:
            margin = torch.sub(distance, STREET_R)
            loss = margin   # minimize distance
            
        # Log loss and distance information
        if self.verbose and x.shape[0] == 1:  # Only log for single samples to avoid clutter
            mean_dist = distance.mean().item()
            mean_loss = loss.mean().item()
            success = (distance > CONTINENT_R).any().item() if not self.targeted else (distance < STREET_R).any().item()
            self.log(f"Distance: {mean_dist:.2f} km, Loss: {mean_loss:.6f}, Success: {success}")
            
        return loss
    
    def compute_ce_loss(self, x, y):
        """
        Compute cross-entropy loss
        """
        logits = self.model.predict_logits(x)
        # Ensure y is on the same device as logits
        y = y.to(logits.device)
        label_indices = coords_to_class_indices_nn(self.model.gps_gallery.to(logits.device), y).to(logits.device)
        
        if not self.targeted:
            loss = -1.0 * F.cross_entropy(logits, label_indices, reduction='none')
        else:
            loss = F.cross_entropy(logits, label_indices, reduction='none')
            
        return loss
    
    def perturb(self, x, y):
        """
        Generate adversarial examples for a batch of inputs.
        
        Args:
            x (torch.Tensor): Original images of shape (B, 3, H, W)
            y (torch.Tensor): Ground-truth coordinates of shape (B, 2)
            
        Returns:
            torch.Tensor: adversarial_examples
        """
        # Call the attack method directly
        adv_x = self.attack.perturb(x, y)
        
        # Return adversarial examples
        return adv_x
    
    def log(self, message):
        """Log a message if verbose"""
        if self.verbose:
            print(message)
            if self.logger is not None:
                self.logger.log(message)


# Create a wrapped model class that has a forward method
class ModelWrapper(torch.nn.Module):
    def __init__(self, get_logits_func):
        super().__init__()
        self.get_logits_func = get_logits_func
        
    def forward(self, x):
        return self.get_logits_func(x)

class AttackCLIP_SparsePatches:
    """
    PGDTrim attack for CLIP models
    """
    def __init__(self, model, data_path, norm='L0', sparsity=100, eps_l_inf=0.03, n_iter=40, 
                 n_restarts=1, targeted=False, loss='ce', device='cuda',
                 verbose=True, seed=42, constant_schedule=False,
                 data_loader=None, resample_loc=None, log_path=None):
        self.model = model
        self.device = device
        self.targeted = targeted
        self.loss_type = loss
        self.verbose = verbose
        self.seed = seed
        self.log_path = log_path
        self.logger = Logger(log_path) if log_path is not None else None
        self.sparsity = sparsity
        self.data_path = data_path
        
        # Create a ClipWrap instance to handle CLIP model processing
        self.clip_wrap = ClipWrap(model, data_path, device)
        
        # Create a wrapped model class that has a forward method
        class ModelWrapper(torch.nn.Module):
            def __init__(self, clip_wrap):
                super().__init__()
                self.clip_wrap = clip_wrap
                
            def forward(self, x):
                return self.clip_wrap(x)
        
        # Calculate total image pixels and trim steps according to the paper
        total_pixels =  224 * 224  # 224x224 image
        trim_steps = [
            int(total_pixels / 2),    # 50% of pixels
            int(total_pixels / 4),    # 25% of pixels
            int(total_pixels / 8),    # 12.5% of pixels
            int(total_pixels / 16),   # 6.25% of pixels
            int(total_pixels / 32),   # 3.125% of pixels
            int(total_pixels / 64),   # 1.5% of pixels
            sparsity                  # Final target sparsity
        ]
        
        misc_args = {
            'device': device,
            'n_restarts': n_restarts,
            'report_info': True,
            'verbose': verbose,
            'seed': seed,
            'dtype': torch.float32,
            'batch_size': 32,
            'data_shape': [3, 224, 224],  # Image shape [channels, width, height]
            'data_RGB_start': [0.0, 0.0, 0.0],  # Min RGB values
            'data_RGB_end': [1.0, 1.0, 1.0],    # Max RGB values
            'data_RGB_size': [1.0, 1.0, 1.0],    # Range of RGB values
            'targeted': targeted  # Pass targeted flag to the attack
        }
        
        pgd_args = {
            'eps_ratio': eps_l_inf,
            'eps': eps_l_inf,  # L_inf constraint
            'norm': norm,
            'n_iter': n_iter,
            'alpha': eps_l_inf * 0.5 if eps_l_inf > 0 else 0.0,  # Increased step size (alpha) for faster convergence
            'alpha_ratio': eps_l_inf * 0.5 if eps_l_inf > 0 else 0.0,  # Increased alpha ratio as well
            'restarts_interval': 1,
            'w_iter_ratio': 1.0,
            'n_restarts': n_restarts,
            'rand_init': True
        }
        
        dropout_args = {
            'dpo_mu': 0.0,
            'dpo_sigma': 0.0,
            'dpo_mu_sched': 0.0,
            'dpo_sigma_sched': 0.0,
            'dropout_mean': 0.0,
            'dropout_std': 0.0,
            'dropout_dist': 'none',
            'dropout_std_bernoulli': False
        }
        
        trim_args = {
            'sparsity': sparsity,
            'trim_steps': trim_steps,
            'max_trim_steps': len(trim_steps),
            'trim_steps_reduce': 'none',
            'scale_dpo_mean': True,
            'post_trim_dpo': True,
            'dynamic_trim': True,
            'l0_methods': ['full'],
            'sparsity_distribution': 'constant',
            'trim_with_mask': 'single'
        }
        
        mask_args = {
            'mask_dist': 'topk',
            'mask_prob_amp_rate': 1.0,
            'norm_mask_amp': True,
            'mask_opt_iter': 10,
            'n_mask_samples': 1,
            'sample_all_masks': False,
            'trim_best_mask': True
        }
        
        kernel_args = None  # Not using kernel attacks
        
        # Create a wrapper model to handle the 'forward' issue
        self.model_wrapper = ModelWrapper(self.clip_wrap)
        
        # Create the PGDTrim attack instance
        self.attack = PGDTrim(
            model=self.model_wrapper,  # Use the wrapper model
            criterion=self.compute_loss,
            misc_args=misc_args,
            pgd_args=pgd_args,
            dropout_args=dropout_args,
            trim_args=trim_args,
            mask_args=mask_args,
            kernel_args=kernel_args
        )
    
    def __call__(self, x):
        """Make this class callable similar to ClipWrap in sparse_rs"""
        return self.clip_wrap(x)
    
    def get_logits(self, x):
        """Get logits from the CLIP model"""
        return self.clip_wrap(x)
    
    def predict(self, x):
        """Get predictions from the CLIP model"""
        logits = self.get_logits(x)
        probs = logits.softmax(dim=1)
        predictions = probs.argmax(dim=1)
        return predictions
    
    def compute_loss(self, x, y):
        """Compute the loss for the attack"""
        logits = self.get_logits(x)
        xent = F.cross_entropy(logits, y, reduction='none')
        
        u = torch.arange(len(x), device=self.device)
        y_corr = logits[u, y].clone()
        logits[u, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]
        
        if not self.targeted:
            if self.loss_type == 'ce':
                return -1. * xent
            elif self.loss_type == 'margin':
                return y_corr - y_others
        else:
            # targeted
            if self.loss_type == 'ce':
                return xent
            elif self.loss_type == 'margin':
                return y_others - y_corr
    
    def perturb(self, x, y):
        """
        Generate adversarial examples for a batch of inputs.
        
        Args:
            x (torch.Tensor): Original images of shape (B, 3, H, W)
            y (torch.Tensor): Ground-truth coordinates of shape (B, 2)
            
        Returns:
            torch.Tensor: adversarial_examples
        """
        # Call the attack method directly
        adv_x = self.attack.perturb(x, y)
        
        # Return adversarial examples
        return adv_x
    
    def log(self, message):
        """Log a message if verbose"""
        if self.verbose:
            print(message)
            if self.logger is not None:
                self.logger.log(message)


class AttackGeoCLIP_SparsePatches_Kernel:
    """
    A kernel-based PGDTrim attack class for GeoCLIP.
    
    This attack applies structured kernel perturbations to create adversarial examples.
    Each kernel is a small patch (e.g., 3x3 or 4x4) with a specified number of perturbed pixels.
    """
    def __init__(self, model, norm='L0', sparsity=100, eps_l_inf=0.03, n_iter=40, 
                 kernel_size=3, kernel_sparsity=9, 
                 n_restarts=1, targeted=False, loss='margin', top_k=5, device='cuda',
                 verbose=True, seed=42, constant_schedule=False,
                 data_loader=None, resample_loc=None, log_path=None):
        
        self.model = model
        self.device = device
        self.targeted = targeted
        self.loss_type = loss
        self.verbose = verbose
        self.seed = seed
        self.log_path = log_path
        self.logger = Logger(log_path) if log_path is not None else None
        self.sparsity = sparsity
        
        # Create the GeoCLIP predictor wrapper
        self.geoclip_predictor = GeoCLIPPredictor(model)
        
        # Create the GeoLocation loss function
        self.geo_loss = GeoLocationLoss(
            predictor=self.geoclip_predictor,
            top_k=top_k,
            targeted=targeted
        )
        
        # Calculate total image pixels and trim steps according to the paper
        total_pixels = 224 * 224  # Only count spatial dimensions (224x224)
        trim_steps = [
            int(total_pixels / 8),    # 12.5% of pixels
            int(total_pixels / 16),   # 6.25% of pixels
            int(total_pixels / 32),   # 3.125% of pixels
            int(total_pixels / 64),   # 1.5% of pixels
            sparsity                  # Final target sparsity
        ]
        
        # For kernel attack, configure the kernel parameters
        kernel_args = {
            'kernel_size': kernel_size,  # Fixed 4x4 kernel
            'n_kernel_pixels': kernel_size * kernel_size,  # 4x4 = 16 pixels per kernel
            'kernel_sparsity': kernel_size * kernel_size,  # Allow all pixels in kernel to be active
            'max_kernel_sparsity': kernel_size * kernel_size,  # Maximum sparsity per kernel
            'kernel_min_active': 2,  # Minimum 2 active pixels per kernel
            'kernel_group': 'pixels',  # Group pixels by kernels
        }
        
        misc_args = {
            'device': device,
            'n_restarts': n_restarts,
            'report_info': True,
            'verbose': verbose,
            'seed': seed,
            'dtype': torch.float32,
            'batch_size': 32,
            'data_shape': [3, 224, 224],  # Image shape [channels, width, height]
            'data_RGB_start': [0.0, 0.0, 0.0],  # Min RGB values
            'data_RGB_end': [1.0, 1.0, 1.0],    # Max RGB values
            'data_RGB_size': [1.0, 1.0, 1.0],    # Range of RGB values
            'targeted': targeted  # Pass targeted flag to the attack
        }
        
        pgd_args = {
            'eps_ratio': eps_l_inf,
            'eps': eps_l_inf,  # L_inf constraint
            'norm': norm,
            'n_iter': n_iter,
            'alpha': eps_l_inf * 0.5 if eps_l_inf > 0 else 0.0,  # Increased step size (alpha) for faster convergence
            'alpha_ratio': eps_l_inf * 0.5 if eps_l_inf > 0 else 0.0,  # Increased alpha ratio as well
            'restarts_interval': 1,
            'w_iter_ratio': 1.0,
            'n_restarts': n_restarts,
            'rand_init': True
        }
        
        dropout_args = {
            'dpo_mu': 0.0,
            'dpo_sigma': 0.0,
            'dpo_mu_sched': 0.0,
            'dpo_sigma_sched': 0.0,
            'dropout_mean': 0.0,
            'dropout_std': 0.0,
            'dropout_dist': 'none',
            'dropout_std_bernoulli': False
        }
        
        trim_args = {
            'sparsity': 16,  # Maximum sparsity per kernel (4x4)
            'trim_steps': trim_steps,  # Use calculated trim steps
            'max_trim_steps': len(trim_steps),
            'trim_steps_reduce': 'none',
            'scale_dpo_mean': True,
            'post_trim_dpo': True,
            'dynamic_trim': True,
            'l0_methods': ['full'],
            'sparsity_distribution': 'constant',
            'trim_with_mask': 'single'
        }
        
        mask_args = {
            'mask_dist': 'topk',  # Use topk for deterministic selection of best pixels
            'mask_prob_amp_rate': 1.0,
            'norm_mask_amp': True,
            'mask_opt_iter': 10,
            'n_mask_samples': 1,
            'sample_all_masks': False,
            'trim_best_mask': True
        }
        
        # Create the GeoAttackPGDTrimKernel instance
        self.attack = GeoAttackPGDTrimKernel(
            model=model,
            criterion=self.compute_loss,
            misc_args=misc_args,
            pgd_args=pgd_args,
            dropout_args=dropout_args,
            trim_args=trim_args,
            mask_args=mask_args,
            kernel_args=kernel_args
        )
    
    def compute_loss(self, x, y):
        """
        Compute the loss for the attack based on the specified loss type
        """
        if self.loss_type == 'margin':
            return self.compute_margin_loss(x, y)
        elif self.loss_type == 'ce':
            return self.compute_ce_loss(x, y)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def compute_margin_loss(self, x, y):
        """
        Compute margin loss based on haversine distance
        """
        output, _ = self.model.predict_from_tensor(x)
        # Ensure y is on the same device as output
        y = y.to(output.device)
        distance = haversine_distance(output, y)
        
        if not self.targeted:
            margin = torch.sub(CONTINENT_R, distance)
            loss = -margin  # maximize distance (minimize negative distance)
        else:
            margin = torch.sub(distance, STREET_R)
            loss = margin   # minimize distance
            
        # Log loss and distance information
        if self.verbose and x.shape[0] == 1:  # Only log for single samples to avoid clutter
            mean_dist = distance.mean().item()
            mean_loss = loss.mean().item()
            success = (distance > CONTINENT_R).any().item() if not self.targeted else (distance < STREET_R).any().item()
            self.log(f"Step - Distance: {mean_dist:.2f} km, Loss: {mean_loss:.6f}, Success: {success}")
            
        return loss
    
    def compute_ce_loss(self, x, y):
        """
        Compute cross-entropy loss
        """
        logits = self.model.predict_logits(x)
        # Ensure y is on the same device as logits
        y = y.to(logits.device)
        label_indices = coords_to_class_indices_nn(self.model.gps_gallery.to(logits.device), y).to(logits.device)
        
        if not self.targeted:
            loss = -1.0 * F.cross_entropy(logits, label_indices, reduction='none')
        else:
            loss = F.cross_entropy(logits, label_indices, reduction='none')
            
        return loss
    
    def perturb(self, x, y):
        """
        Generate adversarial examples for a batch of inputs.
        
        Args:
            x (torch.Tensor): Original images of shape (B, 3, H, W)
            y (torch.Tensor): Ground-truth coordinates of shape (B, 2)
            
        Returns:
            torch.Tensor: adversarial_examples
        """
        print(f"\nStarting kernel-based attack with {self.sparsity} target sparsity...")
        print(f"Input shape: {x.shape}, Target shape: {y.shape}")
        print(f"Using device: {x.device}")
        
        # Get initial prediction
        with torch.no_grad():
            initial_output, _ = self.model.predict_from_tensor(x)
            initial_distance = haversine_distance(initial_output, y).mean().item()
            print(f"Initial distance: {initial_distance:.2f} km")
        
        # Print trim steps
        total_pixels = 224 * 224  # Only count spatial dimensions (224x224)
        trim_steps = [
            int(total_pixels / 8),    # 12.5% of pixels
            int(total_pixels / 16),   # 6.25% of pixels
            int(total_pixels / 32),   # 3.125% of pixels
            int(total_pixels / 64),   # 1.5% of pixels
            self.sparsity            # Final target sparsity
        ]
        print("\nTrim steps:")
        for i, step in enumerate(trim_steps):
            print(f"Step {i+1}: {step} pixels ({step/total_pixels*100:.2f}% of total pixels)")
        
        # Call the attack method directly
        print("\nStarting PGDTrim attack...")
        adv_x = self.attack.perturb(x, y)
        
        # Get final prediction
        with torch.no_grad():
            final_output, _ = self.model.predict_from_tensor(adv_x)
            final_distance = haversine_distance(final_output, y).mean().item()
            print(f"Final distance: {final_distance:.2f} km")
            print(f"Distance improvement: {final_distance - initial_distance:.2f} km")
        
        # Return adversarial examples
        return adv_x
    
    def log(self, message):
        """Log a message if verbose"""
        if self.verbose:
            print(message)
            if self.logger is not None:
                self.logger.log(message)


class AttackCLIP_SparsePatches_Kernel:
    """
    A kernel-based PGDTrim attack for CLIP models.
    
    This attack applies structured kernel perturbations to create adversarial examples
    for CLIP models. Each kernel is a small patch (e.g., 3x3 or 4x4) with a 
    specified number of perturbed pixels.
    """
    def __init__(self, model, data_path, norm='L0', sparsity=100, eps_l_inf=0.03, n_iter=40, 
                 kernel_size=3, kernel_sparsity=9, 
                 n_restarts=1, targeted=False, loss='margin', device='cuda',
                 verbose=True, seed=42, constant_schedule=False,
                 data_loader=None, resample_loc=None, log_path=None):
        
        self.model = model
        self.device = device
        self.targeted = targeted
        self.loss_type = loss
        self.verbose = verbose
        self.seed = seed
        self.log_path = log_path
        self.logger = Logger(log_path) if log_path is not None else None
        self.sparsity = sparsity
        self.data_path = data_path
        
        # Create a ClipWrap instance to handle CLIP model processing
        self.clip_wrap = ClipWrap(model, data_path, device)
        
        # Create a wrapped model class that has a forward method
        class ModelWrapper(torch.nn.Module):
            def __init__(self, clip_wrap):
                super().__init__()
                self.clip_wrap = clip_wrap
                
            def forward(self, x):
                return self.clip_wrap(x)
            
        # Calculate total image pixels and trim steps according to the paper
        total_pixels =  224 * 224  # 224x224 image
        trim_steps = [
            int(total_pixels / 2),    # 50% of pixels
            int(total_pixels / 4),    # 25% of pixels
            int(total_pixels / 8),    # 12.5% of pixels
            int(total_pixels / 16),   # 6.25% of pixels
            int(total_pixels / 32),   # 3.125% of pixels
            int(total_pixels / 64),   # 1.5% of pixels
            sparsity                  # Final target sparsity
        ]
        
        # For kernel attack, configure the kernel parameters
        kernel_args = {
            'kernel_size': kernel_size,  # Kernel size (e.g., 4x4)
            'n_kernel_pixels': kernel_size * kernel_size,  # Total pixels per kernel
            'kernel_sparsity': kernel_sparsity,  # Number of active pixels per kernel
            'max_kernel_sparsity': kernel_size * kernel_size,  # Maximum sparsity per kernel
            'kernel_min_active': 2,  # Minimum number of active pixels per kernel
            'kernel_group': 'pixels',  # Group pixels by kernels
        }
        
        misc_args = {
            'device': device,
            'n_restarts': n_restarts,
            'report_info': True,
            'verbose': verbose,
            'seed': seed,
            'dtype': torch.float32,
            'batch_size': 32,
            'data_shape': [3, 224, 224],  # Image shape [channels, width, height]
            'data_RGB_start': [0.0, 0.0, 0.0],  # Min RGB values
            'data_RGB_end': [1.0, 1.0, 1.0],    # Max RGB values
            'data_RGB_size': [1.0, 1.0, 1.0],    # Range of RGB values
            'targeted': targeted  # Pass targeted flag to the attack
        }
        
        pgd_args = {
            'eps_ratio': eps_l_inf,
            'eps': eps_l_inf,  # L_inf constraint
            'norm': norm,
            'n_iter': n_iter,
            'alpha': eps_l_inf * 0.5 if eps_l_inf > 0 else 0.0,  # Increased step size (alpha) for faster convergence
            'alpha_ratio': eps_l_inf * 0.5 if eps_l_inf > 0 else 0.0,  # Increased alpha ratio as well
            'restarts_interval': 1,
            'w_iter_ratio': 1.0,
            'n_restarts': n_restarts,
            'rand_init': True
        }
        
        dropout_args = {
            'dpo_mu': 0.0,
            'dpo_sigma': 0.0,
            'dpo_mu_sched': 0.0,
            'dpo_sigma_sched': 0.0,
            'dropout_mean': 0.0,
            'dropout_std': 0.0,
            'dropout_dist': 'none',
            'dropout_std_bernoulli': False
        }
        
        trim_args = {
            'sparsity': sparsity,
            'trim_steps': trim_steps,
            'max_trim_steps': len(trim_steps),
            'trim_steps_reduce': 'none',
            'scale_dpo_mean': True,
            'post_trim_dpo': True,
            'dynamic_trim': True,
            'l0_methods': ['full'],
            'sparsity_distribution': 'constant',
            'trim_with_mask': 'single'
        }
        
        mask_args = {
            'mask_dist': 'topk',  # Use topk for deterministic selection of best pixels
            'mask_prob_amp_rate': 1.0,
            'norm_mask_amp': True,
            'mask_opt_iter': 10,
            'n_mask_samples': 1,
            'sample_all_masks': False,
            'trim_best_mask': True
        }
        
        # Create the PGDTrim attack instance with kernel arguments
        from attacks.pgd_attacks.PGDTrim import PGDTrim
        
        # Create a wrapper model to handle the 'forward' issue
        self.model_wrapper = ModelWrapper(self.clip_wrap)
        
        self.attack = PGDTrim(
            model=self.model_wrapper,  # Use the wrapper model
            criterion=self.compute_loss,
            misc_args=misc_args,
            pgd_args=pgd_args,
            dropout_args=dropout_args,
            trim_args=trim_args,
            mask_args=mask_args,
            kernel_args=kernel_args  # Pass kernel args to regular PGDTrim
        )
    
    def __call__(self, x):
        """Make this class callable similar to ClipWrap in sparse_rs"""
        return self.clip_wrap(x)
    
    def get_logits(self, x):
        """Get logits from the CLIP model"""
        return self.clip_wrap(x)
    
    def predict(self, x):
        """Get predictions from the CLIP model"""
        logits = self.get_logits(x)
        probs = logits.softmax(dim=1)
        predictions = probs.argmax(dim=1)
        return predictions
    
    def compute_loss(self, x, y):
        """Compute the loss for the attack"""
        logits = self.get_logits(x)
        xent = F.cross_entropy(logits, y, reduction='none')
        
        u = torch.arange(len(x), device=self.device)
        y_corr = logits[u, y].clone()
        logits[u, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]
        
        if not self.targeted:
            if self.loss_type == 'ce':
                return -1. * xent
            elif self.loss_type == 'margin':
                return y_corr - y_others
        else:
            # targeted
            if self.loss_type == 'ce':
                return xent
            elif self.loss_type == 'margin':
                return y_others - y_corr
    
    def perturb(self, x, y):
        """
        Generate adversarial examples for a batch of inputs.
        
        Args:
            x (torch.Tensor): Original images of shape (B, 3, H, W)
            y (torch.Tensor): Target class indices
            
        Returns:
            torch.Tensor: adversarial_examples
        """
        print(f"\nStarting kernel-based attack for CLIP with {self.sparsity} target sparsity...")
        print(f"Input shape: {x.shape}, Target shape: {y.shape}")
        print(f"Using device: {x.device}")
        
        # Get initial prediction
        with torch.no_grad():
            initial_output = self.get_logits(x)
            initial_preds = initial_output.argmax(dim=1)
            correct = (initial_preds == y).float().mean().item() * 100
            print(f"Initial accuracy: {correct:.2f}%")
        
        # Call the attack method directly
        adv_x = self.attack.perturb(x, y)
        
        # Get final prediction
        with torch.no_grad():
            final_output = self.get_logits(adv_x)
            final_preds = final_output.argmax(dim=1)
            if not self.targeted:
                success = (final_preds != y).float().mean().item() * 100
                print(f"Attack success rate: {success:.2f}%")
            else:
                success = (final_preds == y).float().mean().item() * 100
                print(f"Targeted attack success rate: {success:.2f}%")
        
        # Return adversarial examples
        return adv_x
    
    def log(self, message):
        """Log a message if verbose"""
        if self.verbose:
            print(message)
            if self.logger is not None:
                self.logger.log(message)


class Logger:
    """Simple logger to record attack progress"""
    def __init__(self, path):
        self.path = path
        self.log_file = open(path, 'w')
        
    def log(self, message):
        """Write message to log file"""
        self.log_file.write(message + '\n')
        self.log_file.flush()
        
    def close(self):
        """Close log file"""
        self.log_file.close() 