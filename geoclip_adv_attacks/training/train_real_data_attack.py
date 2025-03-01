import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import requests
from io import BytesIO
import time
import random
import geoclip
from geoclip.model import GeoCLIP
from geoclip_adv_attacks.attacks.pgd_attacks.pgd import PGD
from geoclip_adv_attacks.attacks.pgd_attacks.universal import UPGD
from torchvision import transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration parameters
CONFIG = {
    # Paths
    "gps_gallery_path": "geoclip/model/gps_gallery/coordinates_100K.csv",
    "output_dir": "geoclip_adv_attacks/results/real_data_attack",
    "cache_dir": "geoclip_adv_attacks/data/real_data_cache",
    "mp16_dir": "geoclip_adv_attacks/data/mp16_pro",
    "mp16_metadata": "geoclip_adv_attacks/data/mp16_pro/metadata/mp16_subset.csv",
    
    # Data options
    "num_locations": 100,  # Number of locations to sample from the GPS gallery
    "batch_size": 8,
    "use_real_images": True,  # Whether to use real images from MP16-Pro
    "max_real_images": 1000,  # Maximum number of real images to use
    
    # Attack parameters
    "attack_type": "pgd",  # Options: "pgd" or "universal"
    "epsilon": 0.03,
    "alpha": 0.01,
    "n_iter": 1,
    "n_restarts": 1,
    
    # Output options
    "vis_freq": 5,  # Frequency of visualization (in batches)
    "save_freq": 10,  # Frequency of saving perturbations (in batches)
    
    # Attack target options
    "targeted": False,  # Whether to perform a targeted attack
    "target_location": [0.0, 0.0],  # Target location for targeted attacks [lat, lon]
}

def get_transforms():
    """
    Get image transforms for GeoCLIP
    This should match the preprocessing used by CLIP
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

def geoclip_criterion(model, images, locations, targeted=False, target_location=None):
    """
    Custom criterion for GeoCLIP attacks
    
    For untargeted attacks: Maximize the distance between predicted and true locations
    For targeted attacks: Minimize the distance between predicted and target locations
    """
    def loss_fn(perturbed_images):
        # Get predictions from GeoCLIP
        # For untargeted attacks, we want to maximize distance from true location
        if not targeted:
            logits = model(perturbed_images, locations)
            # Negative cosine similarity (we want to minimize this)
            loss = -logits.diagonal()
        # For targeted attacks, we want to minimize distance to target location
        else:
            if target_location is None:
                raise ValueError("Target location must be provided for targeted attacks")
            
            target_loc = torch.tensor([target_location], dtype=torch.float).to(perturbed_images.device)
            target_loc = target_loc.expand(perturbed_images.shape[0], -1)
            logits = model(perturbed_images, target_loc)
            # Positive cosine similarity (we want to maximize this)
            loss = logits.diagonal()
        
        return loss
    
    return loss_fn

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
        mp16_data = pd.read_csv(CONFIG["mp16_metadata"])
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

class RealGeoDataset(Dataset):
    """
    Dataset for real-world geolocated images using the MP16 dataset
    """
    def __init__(self, mp16_metadata, transform=None, cache_dir=None, mp16_dir=None, max_real_images=1000):
        self.transform = transform
        self.cache_dir = cache_dir
        self.mp16_dir = mp16_dir
        
        # Create cache directory if needed
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load MP16 metadata
        print(f"Loading MP16 metadata from {mp16_metadata}")
        try:
            mp16_data = pd.read_csv(mp16_metadata)
            
            # Limit to max_real_images
            if len(mp16_data) > max_real_images:
                mp16_data = mp16_data.sample(max_real_images)
            
            # Store image paths and coordinates
            self.real_images = []
            for _, row in mp16_data.iterrows():
                img_path = os.path.join(self.mp16_dir, 'images', row['IMG_FILE'])
                self.real_images.append({
                    'path': img_path,
                    'lat': row['LAT'],
                    'lon': row['LON']
                })
            
            print(f"Loaded {len(self.real_images)} images from MP16 dataset")
        except Exception as e:
            print(f"Error loading MP16 metadata: {e}")
            raise e
    
    def __len__(self):
        return len(self.real_images)
    
    def __getitem__(self, idx):
        # Get image info
        img_info = self.real_images[idx]
        lat, lon = img_info['lat'], img_info['lon']
        location = np.array([lat, lon])
        
        # Load image
        if os.path.exists(img_info['path']):
            try:
                image = Image.open(img_info['path']).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_info['path']}: {e}")
                raise e
        else:
            print(f"Image {img_info['path']} not found")
            raise FileNotFoundError(f"Image {img_info['path']} not found")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'location': torch.tensor(location, dtype=torch.float),
            'image_path': img_info['path'],
            'image_name': os.path.basename(img_info['path'])
        }

def train_real_data_attack(config=None):
    """
    Train an adversarial attack on real-world geolocated images
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
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # Load all MP16 locations to use as prediction targets
    print("Loading MP16 locations...")
    mp16_data = pd.read_csv(config["mp16_metadata"])
    mp16_locations = torch.tensor(mp16_data[['LAT', 'LON']].values, dtype=torch.float).to(device)
    
    # Setup attack parameters
    misc_args = {
        'device': device,
        'dtype': torch.float32,
        'batch_size': config["batch_size"],
        'data_shape': [3, 224, 224],  # [C, H, W]
        'data_RGB_start': [-2.0, -2.0, -2.0],  # Min pixel values after CLIP normalization
        'data_RGB_end': [2.0, 2.0, 2.0],    # Max pixel values after CLIP normalization
        'data_RGB_size': [4.0, 4.0, 4.0],   # Range of pixel values
        'verbose': True,
        'report_info': True
    }
    
    pgd_args = {
        'norm': 'Linf',
        'eps': config["epsilon"],
        'n_restarts': config["n_restarts"],
        'n_iter': config["n_iter"],
        'alpha': config["alpha"],
        'rand_init': True
    }
    
    # Initialize attack based on type
    if config["attack_type"].lower() == 'universal':
        attack = UPGD(model, None, misc_args, pgd_args)
        print("Using Universal PGD Attack")
    else:
        attack = PGD(model, None, misc_args, pgd_args)
        print("Using Standard PGD Attack")
    
    attack.report_schematics()
    
    # For universal attack, we'll accumulate gradients across batches
    if config["attack_type"].lower() == 'universal':
        # Initialize universal perturbation
        universal_pert = torch.zeros((1, 3, 224, 224), device=device)
        universal_pert.requires_grad_()
        
        # Optimizer for universal perturbation
        optimizer = torch.optim.Adam([universal_pert], lr=config["alpha"])
    
    # Train attack
    print("Starting attack training...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        images = batch['image'].to(device)
        locations = batch['location'].to(device)
        
        # Create criterion for this batch
        criterion = geoclip_criterion(
            model, 
            images, 
            mp16_locations,  # Use all MP16 locations as potential targets
            targeted=config["targeted"], 
            target_location=config["target_location"]
        )
        
        if config["attack_type"].lower() == 'universal':
            # Apply universal perturbation
            perturbed_images = images + universal_pert
            
            # Ensure perturbed images are within valid range
            perturbed_images = torch.clamp(perturbed_images, -2.0, 2.0)  # CLIP normalization range
            
            # Compute loss for universal perturbation
            loss = criterion(perturbed_images).mean()
            
            # Update universal perturbation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Project perturbation to epsilon ball
            with torch.no_grad():
                universal_pert.data = torch.clamp(universal_pert.data, -config["epsilon"], config["epsilon"])
            
            # For visualization and metrics
            perturbation = universal_pert.expand_as(images)
            perturbed_images = torch.clamp(images + perturbation, -2.0, 2.0)
            
        else:
            # Standard PGD attack
            attack.criterion = criterion
            perturbed_images = attack.perturb(images, locations)
            perturbation = perturbed_images - images
        
        # Calculate distance between original and perturbed predictions
        with torch.no_grad():
            # Get logits for original and perturbed images using MP16 locations
            original_logits = model(images, mp16_locations)
            perturbed_logits = model(perturbed_images, mp16_locations)
            
            # Get top prediction for original and perturbed images
            original_pred_idx = original_logits.argmax(dim=1)
            perturbed_pred_idx = perturbed_logits.argmax(dim=1)
            
            # Get the GPS coordinates for the top predictions
            original_pred_gps = mp16_locations[original_pred_idx]
            perturbed_pred_gps = mp16_locations[perturbed_pred_idx]
            
            # Calculate distance between predictions
            distances = torch.norm(original_pred_gps - perturbed_pred_gps, dim=1)
            
            # Calculate success rate (prediction changed)
            success_rate = (original_pred_idx != perturbed_pred_idx).float().mean()
        
        print(f"Batch {batch_idx+1}/{len(dataloader)}")
        print(f"Average distance: {distances.mean().item():.4f}")
        print(f"Success rate: {success_rate.item():.4f}")
        
        # Visualize first image in batch
        if batch_idx % config["vis_freq"] == 0:
            vis_path = os.path.join(config["output_dir"], f"attack_vis_batch_{batch_idx}.png")
            visualize_attack(
                model,
                images[0], 
                perturbed_images[0], 
                locations[0],
                save_path=vis_path
            )
        
        # Save perturbations
        if batch_idx % config["save_freq"] == 0:
            if config["attack_type"].lower() == 'universal':
                torch.save(universal_pert, os.path.join(config["output_dir"], f"universal_perturbation.pt"))
            else:
                torch.save(perturbation, os.path.join(config["output_dir"], f"perturbation_batch_{batch_idx}.pt"))
    
    # Save final perturbation
    if config["attack_type"].lower() == 'universal':
        torch.save(universal_pert, os.path.join(config["output_dir"], "final_universal_perturbation.pt"))
        print(f"Saved final universal perturbation to {os.path.join(config['output_dir'], 'final_universal_perturbation.pt')}")
    else:
        torch.save(perturbation, os.path.join(config["output_dir"], "final_perturbation.pt"))
        print(f"Saved final perturbation to {os.path.join(config['output_dir'], 'final_perturbation.pt')}")
    
    print("Attack training completed!")
    
    return {
        "perturbation_path": os.path.join(config["output_dir"], 
                                         "final_universal_perturbation.pt" if config["attack_type"].lower() == 'universal' else "final_perturbation.pt")
    }

if __name__ == "__main__":
    # You can modify the CONFIG dictionary here before calling train_real_data_attack
    # For example:
    # CONFIG["attack_type"] = "universal"
    # CONFIG["epsilon"] = 0.05
    # CONFIG["targeted"] = True
    # CONFIG["target_location"] = [40.7128, -74.0060]  # New York City
    
    train_real_data_attack() 