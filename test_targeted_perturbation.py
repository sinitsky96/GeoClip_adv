import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from geoclip.model.GeoCLIP import GeoCLIP
from sparse_rs.util import haversine_distance, CONTINENT_R, STREET_R, CITY_R, REGION_R, COUNTRY_R

def visualize_image(img_tensor, title=None, save_path=None):
    """Visualize a tensor as an image"""
    # Convert tensor to numpy
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Clip values to valid range
    img_np = np.clip(img_np, 0, 1)
    
    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    # Save with custom path if provided, otherwise use title
    if save_path:
        plt.savefig(save_path)
    elif title:
        plt.savefig(f"{title.replace(' ', '_').replace(':', '_')}.png")
    else:
        plt.savefig("adversarial_image.png")
    
    plt.close()

def print_location_info(coords, description=""):
    """Print information about coordinates in a readable format"""
    lat, lon = coords
    print(f"{description} Location: Latitude {lat:.6f}, Longitude {lon:.6f}")

def visualize_perturbation(img_tensor, title=None, save_path=None, threshold=0.05):
    """Visualize the perturbation in the image by highlighting pixels that were modified"""
    # Convert tensor to numpy
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Create a mask that highlights potentially perturbed pixels
    # Look for unusual values that might indicate perturbation
    # For a proper sparse attack, we'd expect only a few pixels to be abnormal
    
    # Calculate mean and std per channel
    mean_per_channel = np.mean(img_np, axis=(0, 1))
    std_per_channel = np.std(img_np, axis=(0, 1))
    
    # Create binary mask for pixels that deviate significantly from the mean
    mask = np.zeros_like(img_np[:,:,0])
    for c in range(3):  # For each channel
        channel_mask = np.abs(img_np[:,:,c] - mean_per_channel[c]) > threshold + 2*std_per_channel[c]
        mask = np.logical_or(mask, channel_mask)
    
    # Create an RGB visualization with perturbed pixels highlighted in red
    vis = np.clip(img_np, 0, 1).copy()
    vis[mask, 0] = 1.0  # Red channel
    vis[mask, 1] = 0.0  # Green channel
    vis[mask, 2] = 0.0  # Blue channel
    
    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(vis)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    # Save with custom path if provided, otherwise use title
    if save_path:
        plt.savefig(save_path)
    elif title:
        plt.savefig(f"{title.replace(' ', '_').replace(':', '_')}_perturbation.png")
    else:
        plt.savefig("adversarial_perturbation.png")
    
    plt.close()
    
    # Return count of perturbed pixels
    return np.sum(mask)

def main():
    parser = argparse.ArgumentParser(description="Visualize and evaluate targeted adversarial examples")
    parser.add_argument("--adv_file", type=str, 
                       default="/home/sinitsky96/project/GeoClip_adv/results/sparse_patches_L0/adv_complete_sparse_patches_L0_geoclip_1_1_iter_100_eps_l_inf_0.10_loss_ce_sparsity_16_targeted_True_targetclass_(37.090924, 25.370521)_seed_42.pt",
                       help="Path to the adversarial examples file")
    parser.add_argument("--output_dir", type=str, default="./adv_visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--target_coords", type=str, default="(37.090924, 25.370521)",
                       help="Target coordinates in format '(lat, lon)'")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run models on")
    
    args = parser.parse_args()
    device = torch.device(args.device)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse target coordinates from string
    target_coords_str = args.target_coords.strip("()").split(",")
    target_coords = torch.tensor([float(target_coords_str[0]), float(target_coords_str[1])], 
                                 dtype=torch.float32).to(device)
    
    print("Loading adversarial examples from", args.adv_file)
    
    # First, inspect the contents of the file
    data = torch.load(args.adv_file)
    
    # Print information about the loaded data
    print("\nFile content information:")
    if isinstance(data, torch.Tensor):
        print(f"- Loaded a tensor with shape: {data.shape}")
        print(f"- Tensor data type: {data.dtype}")
        print(f"- Value range: min={data.min().item():.4f}, max={data.max().item():.4f}")
        adv_complete = data
    elif isinstance(data, dict):
        print(f"- Loaded a dictionary with {len(data)} keys")
        print(f"- Dictionary keys: {list(data.keys())}")
        
        # Try to find tensors in the dictionary
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"  - Key '{key}' contains a tensor with shape: {value.shape}")
                print(f"    - Tensor data type: {value.dtype}")
                print(f"    - Value range: min={value.min().item():.4f}, max={value.max().item():.4f}")
        
        # Try to extract adversarial examples
        if 'adv' in data:
            adv_complete = data['adv']
            print(f"\nUsing 'adv' key with shape {adv_complete.shape}")
        else:
            # Find the first tensor with 4 dimensions (likely images)
            for key, value in data.items():
                if isinstance(value, torch.Tensor) and len(value.shape) == 4:
                    adv_complete = value
                    print(f"\nUsing '{key}' key with shape {adv_complete.shape}")
                    break
            else:
                raise ValueError("Could not find a tensor that looks like images in the file")
    else:
        print(f"- Loaded a {type(data)} type, which is unexpected")
        raise ValueError("Could not load adversarial examples from file")
    
    # Load GeoCLIP model
    print("\nLoading GeoCLIP model...")
    geoclip_model = GeoCLIP()
    geoclip_model.to(device)
    geoclip_model.eval()
    print("GeoCLIP model loaded")
    
    # Print target location
    print_location_info(target_coords.cpu().numpy(), "Target")
    
    # Select a few examples to test
    num_examples = min(5, adv_complete.shape[0])
    indices = torch.arange(num_examples)
    
    # Process each example
    for i in indices:
        print(f"\n===== Example {i+1}/{num_examples} =====")
        adv_img = adv_complete[i].to(device)
        
        # Check if file has original images for comparison
        original_img = None
        if isinstance(data, dict) and 'orig' in data:
            if i < data['orig'].shape[0]:
                original_img = data['orig'][i].to(device)
                
                # Calculate perturbation statistics
                diff = (adv_img - original_img).abs()
                perturbed_pixels = (diff.sum(dim=0) > 0.01).sum().item()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                
                # Print perturbation statistics
                print(f"Perturbation statistics:")
                print(f"- Number of pixels changed: {perturbed_pixels} / {adv_img.shape[1] * adv_img.shape[2]}")
                print(f"- Maximum change: {max_diff:.4f}")
                print(f"- Average change: {mean_diff:.4f}")
                
                # Visualize both images side by side
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                
                # Original image
                orig_np = original_img.permute(1, 2, 0).cpu().numpy()
                axes[0].imshow(np.clip(orig_np, 0, 1))
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                # Adversarial image
                adv_np = adv_img.permute(1, 2, 0).cpu().numpy()
                axes[1].imshow(np.clip(adv_np, 0, 1))
                axes[1].set_title("Adversarial Image")
                axes[1].axis('off')
                
                plt.tight_layout()
                comparison_path = os.path.join(args.output_dir, f"comparison_{i+1}.png")
                plt.savefig(comparison_path)
                plt.close()
                print(f"Saved comparison to {comparison_path}")
        
        # Visualize the adversarial image
        save_path = os.path.join(args.output_dir, f"adv_example_{i+1}.png")
        visualize_image(adv_img, f"Example {i+1} - Adversarial Image", save_path)
        print(f"Saved visualization to {save_path}")
        
        # Visualize potential perturbations
        perturbation_path = os.path.join(args.output_dir, f"perturbation_{i+1}.png")
        perturbed_pixels = visualize_perturbation(adv_img, f"Example {i+1} - Potential Perturbations", perturbation_path)
        print(f"Saved perturbation visualization to {perturbation_path}")
        print(f"Detected approximately {perturbed_pixels} potentially perturbed pixels")
        
        print("\nGeoCLIP Prediction:")
        # Get GeoCLIP prediction for adversarial image
        with torch.no_grad():
            pred_output, _ = geoclip_model.predict_from_tensor(adv_img.unsqueeze(0))
        
        # Print predicted location
        print_location_info(pred_output[0].cpu().numpy(), "Predicted")
        
        # Calculate distance to target
        distance = haversine_distance(pred_output, target_coords.unsqueeze(0)).item()
        print(f"Distance to target: {distance:.2f} km")
        
        # Categorize prediction accuracy based on distance thresholds
        for threshold, category in zip(
            [STREET_R, CITY_R, REGION_R, COUNTRY_R, CONTINENT_R],
            ["Street", "City", "Region", "Country", "Continent"]
        ):
            if distance <= threshold:
                print(f"Prediction is within {category} level accuracy ({threshold} km)")
                break
        else:
            print("Prediction is beyond continent level accuracy")
    
    print("\nAnalysis complete - visualizations saved to", args.output_dir)

if __name__ == "__main__":
    main() 