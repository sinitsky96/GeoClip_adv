import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import geoclip
from geoclip.model import GeoCLIP
from geoclip_adv_attacks.training.train_real_data_attack import RealGeoDataset, get_transforms, visualize_attack

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration parameters
CONFIG = {
    # Paths
    "gps_gallery_path": "geoclip/model/gps_gallery/coordinates_100K.csv",
    "perturbation_path": "geoclip_adv_attacks/results/real_data_attack/final_perturbation.pt",
    "output_dir": "geoclip_adv_attacks/results/real_data_evaluation",
    "cache_dir": "geoclip_adv_attacks/data/real_data_cache",
    "mp16_dir": "geoclip_adv_attacks/data/mp16_pro",
    "mp16_metadata": "geoclip_adv_attacks/data/mp16_pro/metadata/mp16_subset.csv",
    
    # Data options
    "num_locations": 50,  # Number of locations to sample from the GPS gallery
    "batch_size": 8,
    "use_real_images": True,  # Whether to use real images from MP16-Pro
    "max_real_images": 1000,  # Maximum number of real images to use
    
    # Evaluation parameters
    "success_threshold": 100.0,  # Distance threshold in km for considering an attack successful
    "num_vis": 5,  # Number of images to visualize
    
    # Output options
    "save_predictions": True,  # Whether to save all predictions to a CSV file
}

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

def evaluate_real_data_attack(config=None):
    """
    Evaluate an adversarial attack on real-world geolocated images
    """
    if config is None:
        config = CONFIG
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Load GeoCLIP model
    print("Loading GeoCLIP model...")
    model = GeoCLIP()
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    transform = get_transforms()
    dataset = RealGeoDataset(
        config["mp16_metadata"],
        transform=transform,
        cache_dir=config["cache_dir"],
        mp16_dir=config["mp16_dir"],
        max_real_images=config["max_real_images"]
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Load all MP16 locations to use as prediction targets
    print("Loading MP16 locations...")
    mp16_data = pd.read_csv(config["mp16_metadata"])
    mp16_locations = torch.tensor(mp16_data[['LAT', 'LON']].values, dtype=torch.float).to(device)
    
    # Load perturbation
    print(f"Loading perturbation from {config['perturbation_path']}...")
    perturbation = torch.load(config["perturbation_path"], map_location=device, weights_only=True)
    
    # Initialize metrics
    total_distance_km = 0.0
    total_images = 0
    success_count = 0
    
    # For visualization
    vis_images = []
    vis_perturbed_images = []
    vis_locations = []
    
    # For saving all predictions
    all_predictions = []
    
    # If perturbation is universal (single tensor), expand it for each batch
    is_universal = len(perturbation.shape) == 4 and perturbation.shape[0] == 1
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch['image'].to(device)
            locations = batch['location'].to(device)
            image_names = batch['image_name']
            
            # Apply perturbation
            if is_universal:
                current_pert = perturbation.expand(images.shape[0], -1, -1, -1)
            else:
                current_pert = perturbation
            
            perturbed_images = torch.clamp(images + current_pert, -2.0, 2.0)  # CLIP normalization range
            
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
            
            # Count successful attacks (distance > threshold in km)
            successes = distances_km > config["success_threshold"]
            success_count += np.sum(successes)
            
            # Update metrics
            total_distance_km += np.sum(distances_km)
            total_images += images.shape[0]
            
            # Save images for visualization
            if len(vis_images) < total_images:  # Only collect if we haven't collected all images yet
                vis_images.extend(images.cpu())
                vis_perturbed_images.extend(perturbed_images.cpu())
                vis_locations.extend(locations.cpu())
            
            # Save all predictions if requested
            if config["save_predictions"]:
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
    
    # Create visualizations for all images
    print(f"\nCreating visualizations for all {len(vis_images)} images...")
    for i in range(len(vis_images)):
        vis_path = os.path.join(config["output_dir"], f"attack_vis_{i}.png")
        visualize_attack(
            model,
            vis_images[i],
            vis_perturbed_images[i],
            vis_locations[i],
            save_path=vis_path
        )
    
    # Save all predictions to CSV if requested
    if config["save_predictions"] and all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_path = os.path.join(config["output_dir"], "all_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Saved all predictions to {predictions_path}")
    
    # Save results
    results_path = os.path.join(config["output_dir"], "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Attack Evaluation Results\n")
        f.write(f"========================\n")
        f.write(f"Perturbation: {config['perturbation_path']}\n")
        f.write(f"Data: Real-world geolocated images\n")
        f.write(f"Number of locations: {config['num_locations']}\n")
        f.write(f"Success Threshold: {config['success_threshold']} km\n")
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
    # You can modify the CONFIG dictionary here before calling evaluate_real_data_attack
    # For example:
    # CONFIG["perturbation_path"] = "geoclip_adv_attacks/results/real_data_attack/final_universal_perturbation.pt"
    # CONFIG["success_threshold"] = 50.0  # 50 km threshold
    
    evaluate_real_data_attack() 