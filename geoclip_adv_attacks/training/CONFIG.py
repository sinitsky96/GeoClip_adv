"""
Centralized configuration file for GeoClip adversarial attacks.
This file contains all configuration parameters used across the training pipeline.
"""

import os
import torch
# Base paths
BASE_DIR = "geoclip_adv_attacks"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Configuration dictionary
CONFIG = {
    # Directory paths
    "paths": {
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "results_dir": RESULTS_DIR,
        "gps_gallery_path": "geoclip/model/gps_gallery/coordinates_100K.csv",
        "cache_dir": os.path.join(DATA_DIR, "real_data_cache"),
        "mp16_dir": os.path.join(DATA_DIR, "mp16_pro"),
        "mp16_metadata": os.path.join(DATA_DIR, "mp16_pro/metadata/mp16_subset.csv"),
        "attack_output_dir": os.path.join(RESULTS_DIR, "real_data_attack"),
        "eval_output_dir": os.path.join(RESULTS_DIR, "real_data_evaluation"),
    },
    
    # Data settings
    "data": {
        "num_locations": 100,  # Number of locations to sample from the GPS gallery
        "batch_size": 8,
        "use_real_images": True,  # Whether to use real images from MP16-Pro
        "max_real_images": 1000,  # Maximum number of real images to use
        "download_mp16": True,  # Whether to download MP16 dataset if not present
        "mp16_sample_size": 10,  # Number of images to sample from MP16 dataset
    },
    
    # Model settings
    "model": {
        "image_size": [224, 224],  # [H, W]
        "data_shape": [3, 224, 224],  # [C, H, W]
        "data_RGB_start": [-2.0, -2.0, -2.0],  # Min pixel values after CLIP normalization
        "data_RGB_end": [2.0, 2.0, 2.0],  # Max pixel values after CLIP normalization
        "data_RGB_size": [4.0, 4.0, 4.0],  # Range of pixel values
    },
    
    # Attack settings
    "attack": {
        "type": "patch",  # Options: "pgd", "universal", or "patch"
        "epsilon": 0.03,  # Maximum perturbation size
        "alpha": 0.01,  # Step size
        "n_iter": 1,  # Number of iterations
        "n_restarts": 1,  # Number of random restarts
        "targeted": False,  # Whether to perform a targeted attack
        "target_location": [0.0, 0.0],  # Target location for targeted attacks [lat, lon]
        
        # Patch attack specific parameters
        "patch": {
            "size": (16, 16),  # Size of the patch (height, width) - reduced from (32, 32) to fit in 224x224 image
            "location": "random",  # "random" or tuple of (y, x) coordinates
            "l0_limit": 256,  # Maximum number of non-zero pixels in patch (16*16 = 256)
        },
    },
    
    # Training settings
    "training": {
        "vis_freq": 5,  # Frequency of visualization (in batches)
        "save_freq": 10,  # Frequency of saving perturbations (in batches)
        "verbose": True,  # Whether to print verbose information
        "report_info": True,  # Whether to report attack information
    },
    
    # Evaluation settings
    "evaluation": {
        "num_locations": 50,  # Number of locations to evaluate on
        "success_threshold": 100.0,  # Distance threshold in km for considering an attack successful
        "num_vis": 5,  # Number of images to visualize
        "save_predictions": True,  # Whether to save all predictions to a CSV file
    },
}

def get_attack_args():
    """
    Get attack-specific arguments based on the attack type.
    Returns a tuple of (misc_args, attack_specific_args)
    """
    misc_args = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': torch.float32,
        'batch_size': CONFIG["data"]["batch_size"],
        'data_shape': CONFIG["model"]["data_shape"],
        'data_RGB_start': CONFIG["model"]["data_RGB_start"],
        'data_RGB_end': CONFIG["model"]["data_RGB_end"],
        'data_RGB_size': CONFIG["model"]["data_RGB_size"],
        'verbose': CONFIG["training"]["verbose"],
        'report_info': CONFIG["training"]["report_info"]
    }
    
    if CONFIG["attack"]["type"] == "patch":
        attack_args = {
            'norm': 'L0',  # Change from 'patch' to 'L0' since patch attack is a form of L0 attack
            'eps': CONFIG["attack"]["epsilon"],  # Add epsilon parameter
            'patch_size': CONFIG["attack"]["patch"]["size"],
            'patch_location': CONFIG["attack"]["patch"]["location"],
            'n_restarts': CONFIG["attack"]["n_restarts"],
            'n_iter': CONFIG["attack"]["n_iter"],
            'alpha': CONFIG["attack"]["alpha"],
            'rand_init': True,
            'l0_limit': CONFIG["attack"]["patch"]["l0_limit"]
        }
    else:  # pgd or universal
        attack_args = {
            'norm': 'Linf',
            'eps': CONFIG["attack"]["epsilon"],
            'n_restarts': CONFIG["attack"]["n_restarts"],
            'n_iter': CONFIG["attack"]["n_iter"],
            'alpha': CONFIG["attack"]["alpha"],
            'rand_init': True
        }
    
    return misc_args, attack_args

def create_required_directories():
    """Create all required directories from the config."""
    for path in CONFIG["paths"].values():
        if isinstance(path, str) and not path.endswith('.csv'):
            os.makedirs(path, exist_ok=True)
    
    # Create MP16 subdirectories
    os.makedirs(os.path.join(CONFIG["paths"]["mp16_dir"], "images"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["paths"]["mp16_dir"], "metadata"), exist_ok=True) 