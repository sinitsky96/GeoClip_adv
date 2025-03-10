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
        "Im2GPS3k_dir": os.path.join(DATA_DIR, "Im2GPS3k"),
        "Im2GPS3k_images": os.path.join(DATA_DIR, "Im2GPS3k/images"),
        "Im2GPS3k_metadata": os.path.join(DATA_DIR, "Im2GPS3k/im2gps3k_places365.csv"),
        "attack_output_dir": os.path.join(RESULTS_DIR, "real_data_attack"),
        "eval_output_dir": os.path.join(RESULTS_DIR, "real_data_evaluation"),
    },
    
    # Data settings
    "data": {
        "name": "Im2GPS3k", # Dataset name, options "Im2GPS3k", "MP16-Pro"
        "num_locations": 100,  # Number of locations to sample from the GPS gallery
        "batch_size": 8,
        "use_real_images": True,  # Whether to use real images from MP16-Pro
        "max_real_images": 1000,  # Maximum number of real images to use
        "download_mp16": True,  # Whether to download MP16 dataset if not present
        "mp16_sample_size": 10,  # Number of images to sample from MP16 dataset
        "save_results": True,  # Whether to save results
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
            "rand_init": True,  # Whether to use random initialization
            
            # dropout
            "att_dpo_dist": "bernoulli",  # distribution for dropout sampling in pgd_attacks, options: none, bernoulli, cbernoulli, gauss
            "att_dpo_mean": 1,  # mean for the dropout distribution used in pgd_attacks
            "att_dpo_std": 0.1,  # standard deviation for the dropout distribution used in pgd_attack
            
            # trim args
            "att_trim_steps": None,  # list of L0 values for trimming the perturbation pixels, the returned perturbation will have sparsity values equal to the last entry in the list
            "att_max_trim_steps": 5,  # epsilon for trimming the patch
            "att_trim_steps_reduce": "even",  # loss function for trimming the patch
            "att_const_dpo_mean": False,  # do not scale the dropout mean during the trimming process by the trim ratio
            "att_const_dpo_std": False,  # do not scale the standard deviation of the dropout distribution as in bernoulli distribution with same mean
            "att_post_trim_dpo": False,  # apply the dropout after the trimming process as well
            "att_dynamic_trim": False,  # consider pixels wise pertubations from previous restarts in trim

            # mask args
            "att_mask_dist": 'multinomial',  # list of L0 values for trimming the perturbation pixels, the returned perturbation will have sparsity values equal to the last entry in the list
            "att_mask_prob_amp_rate": 0,  # the probability of sampling a pixel will be increased according to it's amplitude at this rate
            "att_norm_mask_amp": False,  # normalize the amplitude of the sampled masks by their number of active pixels
            "att_mask_opt_iter": 0,  # Number of iterations to optimize each mask sample before evaluation
            "att_n_mask_samples": 1000,  # number of dropout samples to take into account for estimating the per-pixel criterion
            "att_no_samples_limit": False,  # do not limit the number of masks samples
            "att_trim_best_mask": 'none',  # when all masks are sampled, trim pixels to the best mask, Options: none (default), in_final, all

            # PGDTrim Kernel args
            "att_kernel_size": 1,  # square kernel size for structured perturbation trimming (default: 1X1)
            "att_kernel_min_active": False, #Consider only fully activated kernel patches when sampling masks
            "att_kernel_group": False, # Group pixels in the mask according to kernel, defult: False, Trim the perturbation according to kernel structure
            
        },
    },
    
    # Training settings
    "training": {
        "seed": 42,  # Random seed
        "gpus": '0',  # List of GPU IDs to use
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