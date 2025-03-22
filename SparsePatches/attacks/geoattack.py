import torch
import numpy as np
from torch.nn import functional as F
from SparsePatches.attacks.pgd_attacks.PGDTrim import PGDTrim
from SparsePatches.attacks.pgd_attacks.PGDTrimKernel import PGDTrimKernel

def haversine_distance(point1, point2):
    """
    Calculate the Haversine distance between two sets of GPS coordinates.
    
    Args:
        point1 (torch.Tensor): First set of GPS coordinates of shape (..., 2) (lat, lon in degrees)
        point2 (torch.Tensor): Second set of GPS coordinates of shape (..., 2) (lat, lon in degrees)
        
    Returns:
        torch.Tensor: Distance in kilometers between the points
    """
    # Ensure both tensors are on the same device
    device = point1.device
    point2 = point2.to(device)
    
    # Convert degrees to radians
    lat1, lon1 = point1[..., 0] * np.pi / 180, point1[..., 1] * np.pi / 180
    lat2, lon2 = point2[..., 0] * np.pi / 180, point2[..., 1] * np.pi / 180
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    
    # Radius of earth in kilometers = 6371
    distance = 6371 * c
    
    return distance

class GeoCLIPPredictor:
    """
    Wraps a GeoCLIP model so that given a batch of image tensors it returns predicted GPS coordinates.
    
    This predictor provides both a default call (which uses a weighted average over the entire gallery)
    and a predict_topk function that returns the top-k predictions and their probabilities.
    """
    def __init__(self, model):
        self.model = model
        # Ensure the GPS gallery is on the same device as the model parameters.
        self.gps_gallery = self.model.gps_gallery.to(next(model.parameters()).device)
    
    def __call__(self, x):
        """
        Returns a single prediction per image computed as the softmax-weighted average over the GPS gallery.
        
        Args:
            x (torch.Tensor): Batch of image tensors of shape (B, 3, H, W).
            
        Returns:
            pred (torch.Tensor): Predicted GPS coordinates of shape (B, 2).
        """
        logits = self.model.forward(x, self.gps_gallery)  # shape: (B, num_gallery)
        probs = torch.softmax(logits, dim=1)              # shape: (B, num_gallery)
        pred = probs @ self.gps_gallery                   # shape: (B, 2)
        return pred

    def predict_topk(self, x, top_k):
        """
        Uses GeoCLIP's predict_from_tensor function to obtain the top-k predictions and probabilities.
        
        Args:
            x (torch.Tensor): Batch of image tensors (B, 3, H, W).
            top_k (int): Number of top predictions to return.
            
        Returns:
            top_pred_gps (torch.Tensor): Tensor of shape (B, top_k, 2) with the top-k GPS coordinates.
            top_pred_prob (torch.Tensor): Tensor of shape (B, top_k) with the corresponding probabilities.
        """
        # Process one image at a time to get proper batch handling
        batch_size = x.shape[0]
        all_top_gps = []
        all_top_probs = []
        
        for i in range(batch_size):
            top_pred_gps, top_pred_prob = self.model.predict_from_tensor(x[i:i+1], top_k=top_k, apply_transforms=False)
            all_top_gps.append(top_pred_gps)
            all_top_probs.append(top_pred_prob)
            
        # Stack results
        top_pred_gps = torch.stack(all_top_gps, dim=0)
        top_pred_prob = torch.stack(all_top_probs, dim=0)
        
        return top_pred_gps, top_pred_prob

class GeoLocationLoss:
    """
    Loss function for the GeoCLIP attack.
    
    For untargeted attacks: margin = 2500.0 - min_distance
    For targeted attacks: margin = distance - 100.0
    """
    def __init__(self, predictor, top_k=5, targeted=False, tau_target=100.0):
        self.predictor = predictor
        self.top_k = top_k
        self.targeted = targeted
        self.tau_target = tau_target
    
    def __call__(self, x, y):
        """
        Compute the margin and loss for a batch of perturbed images using top-k predictions.
        
        Args:
            x (torch.Tensor): Batch of perturbed images of shape (B, 3, H, W).
            y (torch.Tensor): Ground-truth or target GPS coordinates of shape (B, 2) (lat, lon in degrees).
            
        Returns:
            loss (torch.Tensor): Tensor of shape (B,) containing the loss for each sample.
        """
        # Obtain top-k predictions (B, top_k, 2) and their probabilities (B, top_k)
        top_pred_gps, top_pred_prob = self.predictor.predict_topk(x, self.top_k)
        
        # Expand ground-truth coordinates to shape (B, top_k, 2)
        y_expanded = y.unsqueeze(1).expand_as(top_pred_gps)
        
        # Compute haversine distances for each top prediction; result is shape (B, top_k)
        distances = haversine_distance(top_pred_gps, y_expanded)
        
        # For each sample, take the minimum distance among the top-k predictions
        min_distance, _ = distances.min(dim=1)
        
        if not self.targeted:
            # For untargeted attacks: the objective is achieved if min_distance > 2500 km.
            # The margin (and loss) is high when min_distance is low.
            margin = 2500.0 - min_distance
            loss = margin
        else:
            # For targeted attacks: the objective is achieved if min_distance < 100 km.
            # The margin (and loss) is high when min_distance is high.
            margin = min_distance - self.tau_target
            loss = margin
            
        return loss

class GeoAttackPGDTrim(PGDTrim):
    """
    PGDTrim attack adapted for GeoCLIP using the haversine distance loss.
    
    This attack aims to create sparse perturbations that affect the predicted geolocation
    of images by maximizing the haversine distance between the predicted and ground-truth locations.
    """
    def __init__(self, model, criterion, misc_args=None, pgd_args=None, dropout_args=None, 
                 trim_args=None, mask_args=None, kernel_args=None):
        # Initialize parent PGDTrim class
        super(GeoAttackPGDTrim, self).__init__(
            model, criterion, misc_args, pgd_args, dropout_args, trim_args, mask_args, kernel_args
        )
        
        # Create the GeoCLIP predictor wrapper
        self.geoclip_predictor = GeoCLIPPredictor(model)
        
        # Override the name
        self.name = "GeoAttackPGDTrim"
        
        # Set targeted flag (default to False for untargeted attacks)
        self.targeted = misc_args.get('targeted', False) if misc_args else False
        
    def report_schematics(self):
        """Report attack configuration details"""
        print("Running GeoAttackPGDTrim for GeoCLIP")
        print("The attack will gradually trim a dense perturbation to the specified sparsity: " + str(self.sparsity))
        
        print("Perturbations will be computed for the L0 norms:")
        print(self.l0_norms)
        print("The best performing perturbations will be reported for the L0 norms:")
        print(self.output_l0_norms)
        print("perturbations L_inf norm limitation:")
        print(self.eps_ratio)
        print("Number of iterations for optimizing perturbations in each trim step:")
        print(self.n_iter)
        print("perturbations will be optimized with the dropout distribution:")
        print(self.dropout_str)
        print("L0 trim steps schedule for the attack:")
        
        self.report_trim_schematics()
        
        print("L0 pixel trimming will be based on masks sampled from the distribution:")
        print(self.mask_dist_str)
        
    def forward_pass(self, x):
        """
        Forward pass of the model.
        For GeoCLIP, we need to extract the top prediction.
        """
        with torch.no_grad():
            # GeoCLIP model forward requires a location argument
            pred, _ = self.model.predict_from_tensor(x)
        return pred
        
    def perturb(self, x_orig, y_target):
        """
        Generate adversarial examples for a batch of inputs using a modified PGDTrim approach
        with random search elements for GeoCLIP compatibility.
        
        Args:
            x_orig (torch.Tensor): Original images of shape (B, 3, H, W)
            y_target (torch.Tensor): Target or ground-truth coordinates of shape (B, 2)
            
        Returns:
            torch.Tensor: Adversarial examples of shape (B, 3, H, W)
        """
        # Initialize
        device = self.device
        x_orig = x_orig.clone().detach().to(device)
        y_target = y_target.clone().detach().to(device)
        
        # Set parameters for the attack
        self.set_params(x_orig, self.targeted)
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Initialize perturbation
        best_pert = torch.zeros_like(x_orig)
        
        # Get initial loss for comparison
        with torch.no_grad():
            pred, _ = self.model.predict_from_tensor(x_orig)
            
            if not self.targeted:
                # For untargeted attacks, maximize distance
                distance = haversine_distance(pred, y_target)
                best_loss = -torch.sub(2500.0, distance)  # Negative to maximize distance
            else:
                # For targeted attacks, minimize distance
                distance = haversine_distance(pred, y_target)
                best_loss = torch.sub(distance, 100.0)
            
            # Initialize current best perturbation for each sample
            curr_pert = torch.zeros_like(x_orig)
        
        # Get dimensions
        batch_size, channels, height, width = x_orig.shape
        n_pixels = height * width
        
        # Optimization loop for multiple restarts
        for restart_idx in range(self.n_restarts):
            # Initial perturbation
            if self.rand_init:
                # Initialize with random perturbation
                for i in range(batch_size):
                    # Select initial random pixels to perturb
                    n_perturb = min(self.trim_steps[0], n_pixels) if len(self.trim_steps) > 0 else min(1000, n_pixels)
                    indices = torch.randperm(n_pixels)[:n_perturb]
                    h_idx, w_idx = indices // width, indices % width
                    
                    # Apply random perturbation
                    for c in range(channels):
                        rand_values = torch.rand(n_perturb, device=device) * 2 * self.eps_ratio - self.eps_ratio
                        curr_pert[i, c, h_idx, w_idx] = rand_values
                
                # Ensure perturbed image is valid
                curr_pert = torch.clamp(x_orig + curr_pert, 0, 1) - x_orig
            
            # For each trim step
            for trim_idx, sparsity in enumerate(self.trim_steps):
                # Run iterations of random search
                for iter_idx in range(self.n_iter):
                    # Clone current perturbation for modification
                    new_pert = curr_pert.clone()
                    
                    # For each sample in the batch
                    for i in range(batch_size):
                        # Flatten current perturbation
                        curr_pert_flat = curr_pert[i].view(channels, -1)
                        
                        # Identify pixels that are currently perturbed (across all channels)
                        perturbed_pixels = curr_pert_flat.abs().sum(dim=0) > 0
                        perturbed_indices = perturbed_pixels.nonzero().squeeze(1)
                        
                        # Number of pixels to modify in this iteration
                        n_modify = max(1, min(sparsity // 10, 50))
                        
                        if len(perturbed_indices) > 0:
                            # Select random subset of perturbed pixels to modify
                            if len(perturbed_indices) > n_modify:
                                idx_to_modify = perturbed_indices[torch.randperm(len(perturbed_indices))[:n_modify]]
                                
                                # Reshape indices back to 2D coordinates
                                h_idx, w_idx = idx_to_modify // width, idx_to_modify % width
                                
                                # Create random perturbations
                                for c in range(channels):
                                    # Either modify or zero out existing perturbation
                                    if torch.rand(1).item() > 0.5:
                                        rand_values = torch.rand(len(idx_to_modify), device=device) * 2 * self.eps_ratio - self.eps_ratio
                                        new_pert[i, c, h_idx, w_idx] = rand_values
                                    else:
                                        new_pert[i, c, h_idx, w_idx] = 0
                        
                        # Also add some new perturbations to currently unperturbed pixels
                        unperturbed_pixels = ~perturbed_pixels
                        unperturbed_indices = unperturbed_pixels.nonzero().squeeze(1)
                        
                        if len(unperturbed_indices) > 0 and len(unperturbed_indices) > n_modify:
                            idx_to_add = unperturbed_indices[torch.randperm(len(unperturbed_indices))[:n_modify]]
                            
                            # Reshape indices back to 2D coordinates
                            h_idx, w_idx = idx_to_add // width, idx_to_add % width
                            
                            # Add new perturbations
                            for c in range(channels):
                                rand_values = torch.rand(len(idx_to_add), device=device) * 2 * self.eps_ratio - self.eps_ratio
                                new_pert[i, c, h_idx, w_idx] = rand_values
                    
                    # Ensure perturbed image is valid
                    new_pert = torch.clamp(x_orig + new_pert, 0, 1) - x_orig
                    
                    # Evaluate new perturbation
                    with torch.no_grad():
                        adv_x = x_orig + new_pert
                        pred, _ = self.model.predict_from_tensor(adv_x)
                        
                        if not self.targeted:
                            # For untargeted attacks, maximize distance
                            distance = haversine_distance(pred, y_target)
                            new_loss = -torch.sub(2500.0, distance)
                        else:
                            # For targeted attacks, minimize distance
                            distance = haversine_distance(pred, y_target)
                            new_loss = torch.sub(distance, 100.0)
                        
                        # Update if new perturbation is better
                        improved = new_loss < best_loss
                        if improved.any():
                            # Update perturbation for improved samples
                            curr_pert[improved] = new_pert[improved]
                            best_loss[improved] = new_loss[improved]
                            best_pert[improved] = new_pert[improved]
                
                # Apply sparsity constraints after iteration
                if sparsity < n_pixels:
                    with torch.no_grad():
                        # Get perturbation magnitude
                        for i in range(batch_size):
                            pert_flat = curr_pert[i].view(channels, -1)
                            pert_abs = pert_flat.abs().sum(dim=0)  # Sum across channels
                            
                            # Keep only top-k perturbations
                            _, indices = torch.topk(pert_abs, sparsity)
                            
                            # Create a mask to keep only top-k perturbations
                            mask = torch.zeros_like(pert_abs, dtype=torch.bool)
                            mask[indices] = True
                            
                            # Apply mask
                            for c in range(channels):
                                new_channel = torch.zeros_like(pert_flat[c])
                                new_channel[mask] = pert_flat[c][mask]
                                curr_pert[i, c] = new_channel.view(height, width)
                        
                        # Update best perturbation
                        adv_x = x_orig + curr_pert
                        pred, _ = self.model.predict_from_tensor(adv_x)
                        
                        if not self.targeted:
                            # For untargeted attacks, maximize distance
                            distance = haversine_distance(pred, y_target)
                            new_loss = -torch.sub(2500.0, distance)
                        else:
                            # For targeted attacks, minimize distance
                            distance = haversine_distance(pred, y_target)
                            new_loss = torch.sub(distance, 100.0)
                        
                        # Update if new perturbation is better
                        improved = new_loss < best_loss
                        if improved.any():
                            best_pert[improved] = curr_pert[improved]
                            best_loss[improved] = new_loss[improved]
        
        # Apply best perturbation to get final adversarial examples
        adv_x = torch.clamp(x_orig + best_pert, 0, 1)
        
        return adv_x

class GeoAttackPGDTrimKernel(PGDTrimKernel):
    """
    PGDTrimKernel attack adapted for GeoCLIP using the haversine distance loss.
    
    This attack aims to create sparse kernel-based perturbations that affect the predicted geolocation
    of images by maximizing the haversine distance between the predicted and ground-truth locations.
    """
    def __init__(self, model, criterion, misc_args=None, pgd_args=None, dropout_args=None, 
                 trim_args=None, mask_args=None, kernel_args=None):
        # Initialize parent PGDTrimKernel class
        super(GeoAttackPGDTrimKernel, self).__init__(
            model, criterion, misc_args, pgd_args, dropout_args, trim_args, mask_args, kernel_args
        )
        
        # Create the GeoCLIP predictor wrapper
        self.geoclip_predictor = GeoCLIPPredictor(model)
        
        # Override the name
        self.name = "GeoAttackPGDTrimKernel"
        
        # Set targeted flag (default to False for untargeted attacks)
        self.targeted = misc_args.get('targeted', False) if misc_args else False
        
    def report_schematics(self):
        """Report attack configuration details"""
        print("Running GeoAttackPGDTrimKernel for GeoCLIP")
        print("The attack will gradually trim a dense kernel-based perturbation to the specified sparsity: " + str(self.sparsity))
        
        print("Perturbations will be computed for the L0 norms:")
        print(self.l0_norms)
        
        print("Using kernel size:", self.kernel_size)
        print("Number of kernel pixels:", self.n_kernel_pixels)
        print("Kernel sparsity:", self.kernel_sparsity)
        
    def forward_pass(self, x):
        """
        Forward pass of the model.
        For GeoCLIP, we need to extract the top prediction.
        """
        with torch.no_grad():
            # GeoCLIP model forward requires a location argument
            pred, _ = self.model.predict_from_tensor(x)
        return pred
        
    def perturb(self, x_orig, y_target):
        """
        Generate adversarial examples for a batch of inputs using a modified PGDTrim approach
        with random search elements for GeoCLIP compatibility.
        
        Args:
            x_orig (torch.Tensor): Original images of shape (B, 3, H, W)
            y_target (torch.Tensor): Target or ground-truth coordinates of shape (B, 2)
            
        Returns:
            torch.Tensor: Adversarial examples of shape (B, 3, H, W)
        """
        # Initialize
        device = self.device
        x_orig = x_orig.clone().detach().to(device)
        y_target = y_target.clone().detach().to(device)
        
        # Set parameters for the attack
        self.set_params(x_orig, self.targeted)
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Initialize perturbation
        best_pert = torch.zeros_like(x_orig)
        
        # Get initial loss for comparison
        with torch.no_grad():
            pred, _ = self.model.predict_from_tensor(x_orig)
            
            if not self.targeted:
                # For untargeted attacks, maximize distance
                distance = haversine_distance(pred, y_target)
                best_loss = -torch.sub(2500.0, distance)  # Negative to maximize distance
            else:
                # For targeted attacks, minimize distance
                distance = haversine_distance(pred, y_target)
                best_loss = torch.sub(distance, 100.0)
            
            # Initialize current best perturbation for each sample
            curr_pert = torch.zeros_like(x_orig)
        
        # Get dimensions
        batch_size, channels, height, width = x_orig.shape
        n_pixels = height * width
        
        # Optimization loop for multiple restarts
        for restart_idx in range(self.n_restarts):
            # Initial perturbation
            if self.rand_init:
                # Initialize with random perturbation
                for i in range(batch_size):
                    # Select initial random pixels to perturb
                    n_perturb = min(self.trim_steps[0], n_pixels) if len(self.trim_steps) > 0 else min(1000, n_pixels)
                    indices = torch.randperm(n_pixels)[:n_perturb]
                    h_idx, w_idx = indices // width, indices % width
                    
                    # Apply random perturbation
                    for c in range(channels):
                        rand_values = torch.rand(n_perturb, device=device) * 2 * self.eps_ratio - self.eps_ratio
                        curr_pert[i, c, h_idx, w_idx] = rand_values
                
                # Ensure perturbed image is valid
                curr_pert = torch.clamp(x_orig + curr_pert, 0, 1) - x_orig
            
            # For each trim step
            for trim_idx, sparsity in enumerate(self.trim_steps):
                # Run iterations of random search
                for iter_idx in range(self.n_iter):
                    # Clone current perturbation for modification
                    new_pert = curr_pert.clone()
                    
                    # For each sample in the batch
                    for i in range(batch_size):
                        # Flatten current perturbation
                        curr_pert_flat = curr_pert[i].view(channels, -1)
                        
                        # Identify pixels that are currently perturbed (across all channels)
                        perturbed_pixels = curr_pert_flat.abs().sum(dim=0) > 0
                        perturbed_indices = perturbed_pixels.nonzero().squeeze(1)
                        
                        # Number of pixels to modify in this iteration
                        n_modify = max(1, min(sparsity // 10, 50))
                        
                        if len(perturbed_indices) > 0:
                            # Select random subset of perturbed pixels to modify
                            if len(perturbed_indices) > n_modify:
                                idx_to_modify = perturbed_indices[torch.randperm(len(perturbed_indices))[:n_modify]]
                                
                                # Reshape indices back to 2D coordinates
                                h_idx, w_idx = idx_to_modify // width, idx_to_modify % width
                                
                                # Create random perturbations
                                for c in range(channels):
                                    # Either modify or zero out existing perturbation
                                    if torch.rand(1).item() > 0.5:
                                        rand_values = torch.rand(len(idx_to_modify), device=device) * 2 * self.eps_ratio - self.eps_ratio
                                        new_pert[i, c, h_idx, w_idx] = rand_values
                                    else:
                                        new_pert[i, c, h_idx, w_idx] = 0
                        
                        # Also add some new perturbations to currently unperturbed pixels
                        unperturbed_pixels = ~perturbed_pixels
                        unperturbed_indices = unperturbed_pixels.nonzero().squeeze(1)
                        
                        if len(unperturbed_indices) > 0 and len(unperturbed_indices) > n_modify:
                            idx_to_add = unperturbed_indices[torch.randperm(len(unperturbed_indices))[:n_modify]]
                            
                            # Reshape indices back to 2D coordinates
                            h_idx, w_idx = idx_to_add // width, idx_to_add % width
                            
                            # Add new perturbations
                            for c in range(channels):
                                rand_values = torch.rand(len(idx_to_add), device=device) * 2 * self.eps_ratio - self.eps_ratio
                                new_pert[i, c, h_idx, w_idx] = rand_values
                    
                    # Ensure perturbed image is valid
                    new_pert = torch.clamp(x_orig + new_pert, 0, 1) - x_orig
                    
                    # Evaluate new perturbation
                    with torch.no_grad():
                        adv_x = x_orig + new_pert
                        pred, _ = self.model.predict_from_tensor(adv_x)
                        
                        if not self.targeted:
                            # For untargeted attacks, maximize distance
                            distance = haversine_distance(pred, y_target)
                            new_loss = -torch.sub(2500.0, distance)
                        else:
                            # For targeted attacks, minimize distance
                            distance = haversine_distance(pred, y_target)
                            new_loss = torch.sub(distance, 100.0)
                        
                        # Update if new perturbation is better
                        improved = new_loss < best_loss
                        if improved.any():
                            # Update perturbation for improved samples
                            curr_pert[improved] = new_pert[improved]
                            best_loss[improved] = new_loss[improved]
                            best_pert[improved] = new_pert[improved]
                
                # Apply sparsity constraints after iteration
                if sparsity < n_pixels:
                    with torch.no_grad():
                        # Get perturbation magnitude
                        for i in range(batch_size):
                            pert_flat = curr_pert[i].view(channels, -1)
                            pert_abs = pert_flat.abs().sum(dim=0)  # Sum across channels
                            
                            # Keep only top-k perturbations
                            _, indices = torch.topk(pert_abs, sparsity)
                            
                            # Create a mask to keep only top-k perturbations
                            mask = torch.zeros_like(pert_abs, dtype=torch.bool)
                            mask[indices] = True
                            
                            # Apply mask
                            for c in range(channels):
                                new_channel = torch.zeros_like(pert_flat[c])
                                new_channel[mask] = pert_flat[c][mask]
                                curr_pert[i, c] = new_channel.view(height, width)
                        
                        # Update best perturbation
                        adv_x = x_orig + curr_pert
                        pred, _ = self.model.predict_from_tensor(adv_x)
                        
                        if not self.targeted:
                            # For untargeted attacks, maximize distance
                            distance = haversine_distance(pred, y_target)
                            new_loss = -torch.sub(2500.0, distance)
                        else:
                            # For targeted attacks, minimize distance
                            distance = haversine_distance(pred, y_target)
                            new_loss = torch.sub(distance, 100.0)
                        
                        # Update if new perturbation is better
                        improved = new_loss < best_loss
                        if improved.any():
                            best_pert[improved] = curr_pert[improved]
                            best_loss[improved] = new_loss[improved]
        
        # Apply best perturbation to get final adversarial examples
        adv_x = torch.clamp(x_orig + best_pert, 0, 1)
        
        return adv_x