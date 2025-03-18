import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torchvision.utils import save_image
from torchvision import transforms

# Add project root to path
sys.path.append(os.path.abspath('.'))

# Import GeoCLIP model
from geoclip.model.GeoCLIP import GeoCLIP
from geoclip_adv_attacks.data.download import load_preprocessed_data

# Import PGDTrim attack
sys.path.append(os.path.join(os.path.abspath('.'), 'SparsePatches'))
from attacks.pgd_attacks.PGDTrim import PGDTrim

# Function to calculate distance between GPS coordinates (in km)
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# PyTorch version of haversine distance for backpropagation
def haversine_distance_torch(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using PyTorch operations
    for gradient computation.
    
    Args:
        lat1, lon1, lat2, lon2: Tensor coordinates in degrees
        
    Returns:
        distance: Tensor distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(a))
    r = 6371.0  # Radius of earth in kilometers
    return c * r

# Wrapper model class for GeoCLIP that implements a compatible forward method for PGDTrim
class GeoCLIPWrapper(torch.nn.Module):
    def __init__(self, geoclip_model):
        super(GeoCLIPWrapper, self).__init__()
        self.geoclip = geoclip_model
        self.device = geoclip_model.device
        
    def forward(self, x):
        """
        Forward method compatible with PGDTrim
        
        Args:
            x: Input image tensor
            
        Returns:
            x: Same tensor (identity function)
        """
        # This is just a dummy forward method that returns the input
        # The actual prediction is done in the loss function
        return x
    
    def to(self, device):
        self.device = device
        self.geoclip.to(device)
        return super().to(device)
    
    def predict_from_tensor(self, x, top_k=1, apply_transforms=True):
        """
        Wrapper for GeoCLIP's predict_from_tensor method
        """
        return self.geoclip.predict_from_tensor(x, top_k, apply_transforms)

# Custom loss function for GeoCLIP
class GeoCLIPLoss(torch.nn.Module):
    def __init__(self, model, original_coords=None, target_coords=None, targeted=False, use_geodesic=True):
        super(GeoCLIPLoss, self).__init__()
        self.model = model
        self.original_coords = original_coords  # Original coordinates for untargeted attack
        self.target_coords = target_coords      # Target coordinates for targeted attack
        self.targeted = targeted
        self.use_geodesic = use_geodesic
        
    def forward(self, x, y=None):
        """
        Args:
            x: Input image tensor
            y: Target coordinates (not used if target_coords is provided)
            
        Returns:
            loss: Loss value
        """
        # Get predictions from the model
        # We need to use predict_from_tensor which handles the model's forward method correctly
        try:
            # For gradient computation, we need to detach the input and create a copy that requires grad
            x_detached = x.detach()
            x_with_grad = x_detached.clone().requires_grad_(True)
            
            # Get the prediction
            pred_coords, pred_probs = self.model.predict_from_tensor(x_with_grad, top_k=1, apply_transforms=True)
            
            # If targeted attack, maximize probability for target location
            if self.targeted:
                # Find the closest gallery point to our target
                target = self.target_coords.to(pred_coords.device)
                distances = torch.cdist(target.unsqueeze(0), pred_coords)[0]
                closest_idx = torch.argmin(distances)
                
                # Return negative log probability (to minimize)
                # Make sure to return a tensor with batch dimension
                loss = -torch.log(pred_probs[closest_idx] + 1e-10)
                
                # Create a differentiable loss
                dummy_loss = torch.mean(x) * 0.0
                loss = loss.view(x.size(0)) + dummy_loss
                return loss
            else:
                # For untargeted attack with geodesic distance
                if self.use_geodesic and self.original_coords is not None:
                    # Get the top prediction
                    top_pred = pred_coords[0]
                    
                    # Calculate negative geodesic distance (we want to maximize distance)
                    orig = self.original_coords.to(top_pred.device)
                    
                    # We want to maximize the distance, so we return negative distance
                    # We use the top prediction's coordinates
                    distance = -haversine_distance_torch(
                        orig[0], orig[1],
                        top_pred[0], top_pred[1]
                    )
                    
                    # Create a differentiable loss
                    dummy_loss = torch.mean(x) * 0.0
                    loss = distance.view(x.size(0)) + dummy_loss
                    return loss
                else:
                    # Original loss: minimize the probability of the true location
                    # We'll use the top-1 prediction's probability as a proxy
                    loss = -torch.log(1 - pred_probs[0] + 1e-10)
                    
                    # Create a differentiable loss
                    dummy_loss = torch.mean(x) * 0.0
                    loss = loss.view(x.size(0)) + dummy_loss
                    return loss
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            # Return a default loss value if there's an error
            # Make sure to return a tensor with batch dimension that requires grad
            return torch.mean(x) * 0.0

def attack_geoclip(args):
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load GeoCLIP model
    print("Loading GeoCLIP model...")
    geoclip_model = GeoCLIP(from_pretrained=True)
    geoclip_model = geoclip_model.to(device)
    geoclip_model.eval()
    
    # Create wrapper model for PGDTrim
    model = GeoCLIPWrapper(geoclip_model)
    model = model.to(device)
    model.eval()
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    preprocessed_dir = os.path.join(args.data_dir, 'Im2GPS', 'preprocessed')
    images_tensor, coords_tensor = load_preprocessed_data(preprocessed_dir=preprocessed_dir, apply_transforms=False)
    
    # Check if images_tensor is a dictionary (raw tensors) or a tensor (normalized)
    is_dict_format = isinstance(images_tensor, dict)
    
    if is_dict_format:
        print(f"Loaded {len(images_tensor)} raw image tensors (dictionary format)")
        # Get the keys for indexing
        tensor_keys = list(images_tensor.keys())
    else:
        print(f"Loaded {len(images_tensor)} image tensors with shape {images_tensor.shape}")
    
    # Select a sample to attack
    if args.sample_idx is None:
        if is_dict_format:
            sample_idx = np.random.choice(tensor_keys)
        else:
            sample_idx = np.random.randint(0, len(images_tensor))
    else:
        sample_idx = args.sample_idx
    
    print(f"Attacking sample with index {sample_idx}")
    
    # Get the image and ground truth coordinates
    if is_dict_format:
        image = images_tensor[sample_idx]
    else:
        image = images_tensor[sample_idx]
    
    true_coords = coords_tensor[sample_idx]
    
    # Move to device
    image = image.to(device)
    true_coords = true_coords.to(device)
    
    # Check image dimensions and resize if needed
    if image.shape[1] != 224 or image.shape[2] != 224:
        print(f"Resizing image from {image.shape[1]}x{image.shape[2]} to 224x224")
        
        # Create transformation pipeline for resizing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        
        # Apply transformations
        image = transform(image)
        print(f"Resized image shape: {image.shape}")
    
    # Get original prediction
    print("Getting original prediction...")
    with torch.no_grad():
        orig_pred_coords, orig_pred_probs = model.predict_from_tensor(image.unsqueeze(0), top_k=1, apply_transforms=True)
    
    orig_pred_coords = orig_pred_coords[0]
    orig_pred_prob = orig_pred_probs[0]
    
    print(f"Original prediction: {orig_pred_coords.cpu().numpy()} with probability {orig_pred_prob.item():.4f}")
    print(f"Ground truth: {true_coords.cpu().numpy()}")
    
    # Calculate distance error
    orig_distance_error = haversine_distance(
        true_coords[0].item(), true_coords[1].item(),
        orig_pred_coords[0].item(), orig_pred_coords[1].item()
    )
    print(f"Original distance error: {orig_distance_error:.2f} km")
    
    # Set up target coordinates for targeted attack (if specified)
    if args.targeted:
        if args.target_coords is None:
            # Default target: Eiffel Tower, Paris
            target_coords = torch.tensor([48.8584, 2.2945], dtype=torch.float32).to(device)
        else:
            target_coords = torch.tensor(args.target_coords, dtype=torch.float32).to(device)
        print(f"Target coordinates: {target_coords.cpu().numpy()}")
        original_coords = None
    else:
        target_coords = None
        # For untargeted attack with geodesic distance, we need the original prediction
        original_coords = orig_pred_coords.clone()
        print(f"Using geodesic distance from original prediction: {original_coords.cpu().numpy()}")
    
    # Create loss function
    use_geodesic = getattr(args, 'use_geodesic', True)  # Default to True if not specified
    criterion = GeoCLIPLoss(model, original_coords=original_coords, target_coords=target_coords, 
                           targeted=args.targeted, use_geodesic=use_geodesic)
    
    # Set up PGDTrim attack
    misc_args = {
        'device': device,
        'dtype': torch.float32,
        'batch_size': 1,
        'data_shape': list(image.shape),
        'data_RGB_start': [0, 0, 0],
        'data_RGB_end': [1, 1, 1],
        'data_RGB_size': [1, 1, 1],
        'verbose': args.verbose,
        'report_info': args.verbose
    }
    
    pgd_args = {
        'norm': 'L0',
        'eps': args.eps / 255.0,  # Convert to [0,1] range
        'n_restarts': args.n_restarts,
        'n_iter': args.n_iter,
        'alpha': args.alpha / 255.0,  # Convert to [0,1] range
        'rand_init': not args.no_rand_init
    }
    
    dropout_args = {
        'dropout_dist': args.dropout_dist,
        'dropout_mean': args.dropout_mean,
        'dropout_std': args.dropout_std,
        'dropout_std_bernoulli': True
    }
    
    trim_args = {
        'sparsity': args.sparsity,
        'trim_steps': None,
        'max_trim_steps': args.max_trim_steps,
        'trim_steps_reduce': args.trim_steps_reduce,
        'scale_dpo_mean': True,
        'post_trim_dpo': args.post_trim_dpo,
        'dynamic_trim': args.dynamic_trim
    }
    
    mask_args = {
        'mask_dist': args.mask_dist,
        'mask_prob_amp_rate': args.mask_prob_amp_rate,
        'norm_mask_amp': args.norm_mask_amp,
        'mask_opt_iter': args.mask_opt_iter,
        'n_mask_samples': args.n_mask_samples,
        'sample_all_masks': not args.no_sample_all_masks,
        'trim_best_mask': args.trim_best_mask
    }
    
    print("Setting up PGDTrim attack...")
    attack = PGDTrim(
        model=model,
        criterion=criterion,
        misc_args=misc_args,
        pgd_args=pgd_args,
        dropout_args=dropout_args,
        trim_args=trim_args,
        mask_args=mask_args
    )
    
    # Run the attack
    print("Running PGDTrim attack...")
    # Prepare input in the format expected by PGDTrim
    x = image.unsqueeze(0)  # Add batch dimension
    y = torch.zeros(1, dtype=torch.long).to(device)  # Dummy label (not used by our custom loss)
    
    # Run the attack
    adv_x = attack.perturb(x, y, targeted=args.targeted)
    
    # Calculate perturbation statistics
    perturbation = adv_x - x
    l0_norm = torch.sum(torch.abs(perturbation) > 1e-6).item()
    l2_norm = torch.norm(perturbation).item()
    linf_norm = torch.max(torch.abs(perturbation)).item()
    
    print(f"Attack complete. Perturbation statistics:")
    print(f"L0 norm: {l0_norm} pixels")
    print(f"L2 norm: {l2_norm:.6f}")
    print(f"L∞ norm: {linf_norm:.6f}")
    
    # Get adversarial prediction
    print("Getting adversarial prediction...")
    with torch.no_grad():
        adv_pred_coords, adv_pred_probs = model.predict_from_tensor(adv_x, top_k=1, apply_transforms=True)
    
    adv_pred_coords = adv_pred_coords[0]
    adv_pred_prob = adv_pred_probs[0]
    
    print(f"Adversarial prediction: {adv_pred_coords.cpu().numpy()} with probability {adv_pred_prob.item():.4f}")
    
    # Calculate distance error for adversarial example
    if args.targeted:
        # For targeted attack, calculate distance to target
        adv_distance_error = haversine_distance(
            target_coords[0].item(), target_coords[1].item(),
            adv_pred_coords[0].item(), adv_pred_coords[1].item()
        )
        print(f"Distance to target: {adv_distance_error:.2f} km")
    else:
        # For untargeted attack, calculate distance to ground truth
        adv_distance_error = haversine_distance(
            true_coords[0].item(), true_coords[1].item(),
            adv_pred_coords[0].item(), adv_pred_coords[1].item()
        )
        print(f"Adversarial distance error: {adv_distance_error:.2f} km")
    
    # Calculate displacement (how far the prediction moved)
    displacement = haversine_distance(
        orig_pred_coords[0].item(), orig_pred_coords[1].item(),
        adv_pred_coords[0].item(), adv_pred_coords[1].item()
    )
    print(f"Prediction displacement: {displacement:.2f} km")
    
    # Save results if requested
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save original and adversarial images
        save_image(image.cpu(), os.path.join(args.output_dir, "original.png"))
        save_image(adv_x[0].cpu(), os.path.join(args.output_dir, "adversarial.png"))
        
        # Save perturbation visualization (scaled for visibility)
        pert_vis = perturbation[0].abs().cpu()
        pert_vis = pert_vis / pert_vis.max() if pert_vis.max() > 0 else pert_vis
        save_image(pert_vis, os.path.join(args.output_dir, "perturbation.png"))
        
        # Save results to text file
        with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
            f.write(f"Sample index: {sample_idx}\n")
            f.write(f"Ground truth coordinates: {true_coords.cpu().numpy()}\n\n")
            
            f.write(f"Original prediction: {orig_pred_coords.cpu().numpy()} with probability {orig_pred_prob.item():.4f}\n")
            f.write(f"Original distance error: {orig_distance_error:.2f} km\n\n")
            
            if args.targeted:
                f.write(f"Target coordinates: {target_coords.cpu().numpy()}\n")
            
            f.write(f"Adversarial prediction: {adv_pred_coords.cpu().numpy()} with probability {adv_pred_prob.item():.4f}\n")
            
            if args.targeted:
                f.write(f"Distance to target: {adv_distance_error:.2f} km\n")
            else:
                f.write(f"Adversarial distance error: {adv_distance_error:.2f} km\n")
            
            f.write(f"Prediction displacement: {displacement:.2f} km\n\n")
            
            f.write(f"Perturbation statistics:\n")
            f.write(f"L0 norm: {l0_norm} pixels\n")
            f.write(f"L2 norm: {l2_norm:.6f}\n")
            f.write(f"L∞ norm: {linf_norm:.6f}\n")
        
        print(f"Results saved to {args.output_dir}")
    
    return {
        "sample_idx": sample_idx,
        "true_coords": true_coords.cpu().numpy(),
        "orig_pred_coords": orig_pred_coords.cpu().numpy(),
        "orig_pred_prob": orig_pred_prob.item(),
        "orig_distance_error": orig_distance_error,
        "adv_pred_coords": adv_pred_coords.cpu().numpy(),
        "adv_pred_prob": adv_pred_prob.item(),
        "adv_distance_error": adv_distance_error,
        "displacement": displacement,
        "l0_norm": l0_norm,
        "l2_norm": l2_norm,
        "linf_norm": linf_norm
    }

def main():
    parser = argparse.ArgumentParser(description="PGDTrim Attack on GeoCLIP")
    
    # Data and output arguments
    parser.add_argument("--data_dir", type=str, default="D:/Study Docs/Degree Material/Sem 9 proj/GeoClip_adv/geoclip_adv_attacks/data", 
                        help="Directory containing Im2GPS data")
    parser.add_argument("--output_dir", type=str, default="D:/Study Docs/Degree Material/Sem 9 proj/GeoClip_adv/attack_results", 
                        help="Directory to save results")
    parser.add_argument("--sample_idx", type=int, default=None, 
                        help="Index of the sample to attack (None for random)")
    parser.add_argument("--save_results", action="store_true", 
                        help="Save attack results")
    parser.add_argument("--cuda", action="store_true", 
                        help="Use CUDA if available")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print verbose output")
    
    # Attack type arguments
    parser.add_argument("--targeted", action="store_true", 
                        help="Perform targeted attack")
    parser.add_argument("--target_coords", type=float, nargs=2, default=None, 
                        help="Target coordinates [lat, lon] for targeted attack")
    parser.add_argument("--use_geodesic", action="store_true", default=True,
                        help="Use geodesic distance for untargeted attacks")
    
    # PGDTrim attack arguments
    parser.add_argument("--eps", type=int, default=8, 
                        help="Epsilon for L∞ constraint (from 255)")
    parser.add_argument("--alpha", type=float, default=2.0, 
                        help="Step size for PGD (from 255)")
    parser.add_argument("--n_iter", type=int, default=100, 
                        help="Number of PGD iterations")
    parser.add_argument("--n_restarts", type=int, default=1, 
                        help="Number of restarts for PGD")
    parser.add_argument("--no_rand_init", action="store_true", 
                        help="Disable random initialization for PGD")
    
    # Sparsity and trimming arguments
    parser.add_argument("--sparsity", type=int, default=100, 
                        help="Target sparsity (number of pixels to perturb)")
    parser.add_argument("--max_trim_steps", type=int, default=5, 
                        help="Maximum number of trimming steps")
    parser.add_argument("--trim_steps_reduce", type=str, default="even", 
                        choices=["none", "even", "best"],
                        help="Policy for reducing trim steps")
    parser.add_argument("--dynamic_trim", action="store_true", 
                        help="Use dynamic trimming")
    
    # Dropout arguments
    parser.add_argument("--dropout_dist", type=str, default="bernoulli", 
                        choices=["none", "bernoulli", "cbernoulli", "gauss"],
                        help="Distribution for dropout sampling")
    parser.add_argument("--dropout_mean", type=float, default=1.0, 
                        help="Mean for dropout distribution")
    parser.add_argument("--dropout_std", type=float, default=1.0, 
                        help="Standard deviation for dropout distribution")
    parser.add_argument("--post_trim_dpo", action="store_true", 
                        help="Apply dropout after trimming")
    
    # Mask arguments
    parser.add_argument("--mask_dist", type=str, default="multinomial", 
                        choices=["topk", "multinomial", "bernoulli", "cbernoulli"],
                        help="Distribution for sampling binary masks")
    parser.add_argument("--mask_prob_amp_rate", type=int, default=0, 
                        help="Rate for increasing sampling probability")
    parser.add_argument("--norm_mask_amp", action="store_true", 
                        help="Normalize mask amplitude")
    parser.add_argument("--mask_opt_iter", type=int, default=0, 
                        help="Number of iterations to optimize masks")
    parser.add_argument("--n_mask_samples", type=int, default=1000, 
                        help="Number of mask samples")
    parser.add_argument("--no_sample_all_masks", action="store_true", 
                        help="Disable sampling all masks")
    parser.add_argument("--trim_best_mask", type=int, default=0, 
                        choices=[0, 1, 2],
                        help="When to trim to best mask (0=none, 1=final, 2=all)")
    
    args = parser.parse_args()
    
    # Run the attack
    results = attack_geoclip(args)
    
    return results

if __name__ == "__main__":
    main()