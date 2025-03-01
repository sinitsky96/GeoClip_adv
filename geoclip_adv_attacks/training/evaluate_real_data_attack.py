import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import geoclip
from geoclip.model import GeoCLIP
from train_real_data_attack import RealGeoDataset, get_transforms
from CONFIG import CONFIG, create_required_directories

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def visualize_attack(model, original_img, perturbed_img, location, save_path=None):
    """
    Visualize the attack results
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert tensors to numpy arrays for visualization
    original_img_np = original_img.cpu().numpy().transpose(1, 2, 0)
    perturbed_img_np = perturbed_img.cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize images (CLIP normalization)
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    original_img_np = original_img_np * std + mean
    perturbed_img_np = perturbed_img_np * std + mean
    
    # Clip to [0, 1]
    original_img_np = np.clip(original_img_np, 0, 1)
    perturbed_img_np = np.clip(perturbed_img_np, 0, 1)
    
    # Get predictions for original and perturbed images
    with torch.no_grad():
        # Prepare inputs
        original_img_batch = original_img.unsqueeze(0).to(device)
        perturbed_img_batch = perturbed_img.unsqueeze(0).to(device)
        location_batch = location.unsqueeze(0).to(device)
        
        # Load MP16 locations for predictions
        mp16_data = pd.read_csv(CONFIG["paths"]["mp16_metadata"])
        mp16_locations = torch.tensor(mp16_data[['LAT', 'LON']].values, dtype=torch.float).to(device)
        
        # Get logits for original and perturbed images
        original_logits = model(original_img_batch, mp16_locations)
        perturbed_logits = model(perturbed_img_batch, mp16_locations)
        
        # Get top prediction for original and perturbed images
        original_pred_idx = original_logits.argmax(dim=1)[0]
        perturbed_pred_idx = perturbed_logits.argmax(dim=1)[0]
        
        # Get the GPS coordinates for the top predictions
        original_pred_gps = mp16_locations[original_pred_idx].cpu().numpy()
        perturbed_pred_gps = mp16_locations[perturbed_pred_idx].cpu().numpy()
        
        # Get the true GPS coordinates
        true_gps = location.cpu().numpy()
    
    # Plot original image
    axs[0].imshow(original_img_np)
    axs[0].set_title(f"Original Image\nTrue: {true_gps}\nPred: {original_pred_gps}")
    axs[0].axis('off')
    
    # Plot perturbed image
    axs[1].imshow(perturbed_img_np)
    axs[1].set_title(f"Perturbed Image\nTrue: {true_gps}\nPred: {perturbed_pred_gps}")
    axs[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def evaluate_real_data_attack():
    """
    Evaluate an adversarial attack on real-world geolocated images
    """
    # Create required directories
    create_required_directories()
    
    # Load GeoCLIP model
    print("Loading GeoCLIP model...")
    model = GeoCLIP()
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    transform = get_transforms()
    dataset = RealGeoDataset(
        CONFIG["paths"]["mp16_metadata"],
        transform=transform,
        cache_dir=CONFIG["paths"]["cache_dir"],
        mp16_dir=CONFIG["paths"]["mp16_dir"],
        max_real_images=CONFIG["data"]["max_real_images"]
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG["data"]["batch_size"], shuffle=False)
    
    # Load all MP16 locations to use as prediction targets
    print("Loading MP16 locations...")
    mp16_data = pd.read_csv(CONFIG["paths"]["mp16_metadata"])
    mp16_locations = torch.tensor(mp16_data[['LAT', 'LON']].values, dtype=torch.float).to(device)
    
    # Load perturbation
    perturbation_path = os.path.join(CONFIG["paths"]["attack_output_dir"], 
                                    f"final_{CONFIG['attack']['type']}_attack.pt")
    print(f"Loading perturbation from {perturbation_path}...")
    loaded_data = torch.load(perturbation_path, map_location=device)
    
    # Check if this is a patch attack
    is_patch_attack = isinstance(loaded_data, dict) and 'patch' in loaded_data and 'location' in loaded_data
    if is_patch_attack:
        patch = loaded_data['patch']
        patch_location = loaded_data['location']
        print(f"Patch location: {patch_location}")
        print(f"Patch shape: {patch.shape}")
        print(f"Patch device: {patch.device}")
        
        # Create a small patch of the correct size (16x16 as per CONFIG)
        patch_h, patch_w = CONFIG["attack"]["patch"]["size"]
        if patch.shape[2:] != (patch_h, patch_w):
            print(f"Resizing patch from {patch.shape[2:]} to {(patch_h, patch_w)}")
            # Create a new patch of the correct size
            small_patch = torch.zeros((1, 3, patch_h, patch_w), device=device)
            small_patch[:, :, :patch_h, :patch_w] = patch[:1, :, :patch_h, :patch_w]
            patch = small_patch
        
        # Ensure patch is on the correct device
        patch = patch.to(device)
    else:
        perturbation = loaded_data
    
    # Initialize metrics
    total_distance_km = 0.0
    total_images = 0
    success_count = 0
    
    # For saving all predictions
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch['image'].to(device)
            locations = batch['location'].to(device)
            image_names = batch['image_name']
            
            # Apply perturbation
            if is_patch_attack:
                # For patch attacks, we need to apply the patch at the specified location
                ph, pw = patch.shape[2:]  # Get patch height and width
                H, W = images.shape[2:]   # Get image height and width
                

                
                # Initialize perturbed images as a copy of the original images
                perturbed_images = images.clone()
                
                # Handle each image in the batch separately
                for i in range(images.shape[0]):
                    if patch_location == 'random':
                        # Generate random location for each image
                        y = torch.randint(0, H - ph + 1, (1,)).item()
                        x = torch.randint(0, W - pw + 1, (1,)).item()
                    else:
                        y, x = patch_location
                    
                    
                    # Create mask for the patch (1 where patch should be, 0 elsewhere)
                    mask = torch.zeros((3, ph, pw), device=device)  # Create mask of patch size
                    mask.fill_(1)  # Fill with ones where patch should be
                    
                    # Apply the patch to this image
                    patch_to_apply = patch[0]  # Use first patch if multiple are provided
                    perturbed_images[i, :, y:y+ph, x:x+pw] = images[i, :, y:y+ph, x:x+pw] * (1 - mask) + patch_to_apply * mask
                    
                    # Verify the patch was applied correctly
                    patch_region = perturbed_images[i, :, y:y+ph, x:x+pw]
            else:
                perturbed_images = torch.clamp(
                    images + perturbation, 
                    CONFIG["model"]["data_RGB_start"][0], 
                    CONFIG["model"]["data_RGB_end"][0]
                )
            
            # Visualize each image in batch
            for i in range(images.shape[0]):
                vis_path = os.path.join(CONFIG["paths"]["eval_output_dir"], f"attack_vis_{total_images + i}.png")
                visualize_attack(
                    model,
                    images[i], 
                    perturbed_images[i], 
                    locations[i],
                    save_path=vis_path
                )
            
            # Get logits for original and perturbed images using MP16 locations
            original_logits = model(images, mp16_locations)
            perturbed_logits = model(perturbed_images, mp16_locations)
            
            # Get top prediction for original and perturbed images
            original_pred_idx = original_logits.argmax(dim=1)
            perturbed_pred_idx = perturbed_logits.argmax(dim=1)
            
            # Get the GPS coordinates for the top predictions
            original_pred_gps = mp16_locations[original_pred_idx]
            perturbed_pred_gps = mp16_locations[perturbed_pred_idx]
            
            # Calculate distances in kilometers using Haversine formula
            distances_km = np.zeros(images.shape[0])
            for i in range(images.shape[0]):
                orig_lat, orig_lon = original_pred_gps[i].cpu().numpy()
                pert_lat, pert_lon = perturbed_pred_gps[i].cpu().numpy()
                distances_km[i] = haversine_distance(orig_lat, orig_lon, pert_lat, pert_lon)
            
            # Count successful attacks (distance > threshold)
            successes = distances_km > CONFIG["evaluation"]["success_threshold"]
            success_count += np.sum(successes)
            
            # Update metrics
            total_distance_km += np.sum(distances_km)
            total_images += images.shape[0]
            
            # Save all predictions if requested
            if CONFIG["evaluation"]["save_predictions"]:
                for i in range(images.shape[0]):
                    true_lat, true_lon = locations[i].cpu().numpy()
                    orig_lat, orig_lon = original_pred_gps[i].cpu().numpy()
                    pert_lat, pert_lon = perturbed_pred_gps[i].cpu().numpy()
                    
                    all_predictions.append({
                        'image_name': image_names[i],
                        'true_lat': true_lat,
                        'true_lon': true_lon,
                        'original_pred_lat': orig_lat,
                        'original_pred_lon': orig_lon,
                        'perturbed_pred_lat': pert_lat,
                        'perturbed_pred_lon': pert_lon,
                        'distance_km': distances_km[i],
                        'success': successes[i]
                    })
    
    # Calculate average metrics
    print("total_images", total_images)
    avg_distance_km = total_distance_km / total_images
    success_rate = success_count / total_images
    
    print(f"Evaluation Results:")
    print(f"Average Distance: {avg_distance_km:.2f} km")
    print(f"Success Rate: {success_rate:.4f} ({success_count}/{total_images})")
    
    # Save all predictions to CSV if requested
    if CONFIG["evaluation"]["save_predictions"] and all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_path = os.path.join(CONFIG["paths"]["eval_output_dir"], "all_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Saved all predictions to {predictions_path}")
    
    # Save results
    results_path = os.path.join(CONFIG["paths"]["eval_output_dir"], "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Attack Evaluation Results\n")
        f.write(f"========================\n")
        f.write(f"Attack Type: {CONFIG['attack']['type']}\n")
        f.write(f"Data: Real-world geolocated images\n")
        f.write(f"Number of locations: {CONFIG['data']['num_locations']}\n")
        f.write(f"Success Threshold: {CONFIG['evaluation']['success_threshold']} km\n")
        f.write(f"========================\n")
        f.write(f"Average Distance: {avg_distance_km:.2f} km\n")
        f.write(f"Success Rate: {success_rate:.4f} ({success_count}/{total_images})\n")
    
    print(f"Results saved to {results_path}")
    
    return {
        'avg_distance_km': avg_distance_km,
        'success_rate': success_rate,
        'total_images': total_images,
        'success_count': success_count
    }

if __name__ == "__main__":
    evaluate_real_data_attack() 