import os
import torch
import sys
from geoclip_adv_attacks.training.train_real_data_attack import train_real_data_attack, CONFIG as TRAIN_CONFIG
from geoclip_adv_attacks.training.evaluate_real_data_attack import evaluate_real_data_attack, CONFIG as EVAL_CONFIG
from geoclip_adv_attacks.data.mp16_pro.download.download_mp16 import main as download_mp16_data

# Main configuration for the entire pipeline
PIPELINE_CONFIG = {
    # General settings
    "base_dir": "geoclip_adv_attacks",
    
    # Data settings
    "cache_dir": "geoclip_adv_attacks/data/real_data_cache",
    "mp16_dir": "geoclip_adv_attacks/data/mp16_pro",
    "mp16_metadata": "geoclip_adv_attacks/data/mp16_pro/metadata/mp16_subset.csv",
    "num_locations": 100,  # Number of locations to sample from the GPS gallery
    "use_real_images": True,  # Whether to use real images from MP16-Pro
    "max_real_images": 1000,  # Maximum number of real images to use
    "download_mp16": True,  # Whether to download MP16 dataset if not present
    "mp16_sample_size": 10,  # Number of images to sample from MP16 dataset
    
    # Attack settings
    "attack_type": "pgd",  # Options: "pgd" or "universal"
    "epsilon": 0.03,
    "alpha": 0.01,
    "n_iter": 1,
    "n_restarts": 1,
    "batch_size": 8,
    "targeted": False,  # Whether to perform a targeted attack
    "target_location": [0.0, 0.0],  # Target location for targeted attacks [lat, lon]
    
    # Output settings
    "vis_freq": 5,  # Frequency of visualization (in batches)
    "save_freq": 10,  # Frequency of saving perturbations (in batches)
    
    # Evaluation settings
    "eval_num_locations": 50,  # Number of locations to evaluate on
    "success_threshold": 100.0,  # Distance threshold in km for considering an attack successful
    "num_vis": 5,  # Number of images to visualize
    "save_predictions": True,  # Whether to save all predictions to a CSV file
}

def check_mp16_data(config):
    """Check if MP16 dataset is downloaded and download if necessary."""
    metadata_path = config["mp16_metadata"]
    if not os.path.exists(metadata_path):
        if config["download_mp16"]:
            print("\n[Data Setup] MP16 dataset not found. Downloading...")
            try:
                download_mp16_data()
                if not os.path.exists(metadata_path):
                    raise Exception("Failed to download MP16 dataset")
                print("[Data Setup] MP16 dataset downloaded successfully.")
            except Exception as e:
                print(f"[Error] Failed to download MP16 dataset: {str(e)}")
                print("Please check your internet connection and try again.")
                sys.exit(1)
        else:
            print("[Error] MP16 dataset not found and download_mp16 is set to False.")
            print(f"Expected metadata file at: {metadata_path}")
            sys.exit(1)
    return True

def run_pipeline(config=None):
    """
    Run the complete adversarial attack pipeline:
    1. Check and download MP16 dataset if necessary
    2. Train the attack on real data
    3. Evaluate the attack on real data
    """
    if config is None:
        config = PIPELINE_CONFIG
    
    print("=" * 50)
    print("GEOCLIP REAL DATA ADVERSARIAL ATTACK PIPELINE")
    print("=" * 50)
    
    # Setup directories
    base_dir = config["base_dir"]
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    attack_dir = os.path.join(results_dir, "real_data_attack")
    eval_dir = os.path.join(results_dir, "real_data_evaluation")
    cache_dir = config["cache_dir"]
    mp16_dir = config["mp16_dir"]
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(attack_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(mp16_dir, exist_ok=True)
    os.makedirs(os.path.join(mp16_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(mp16_dir, "metadata"), exist_ok=True)
    
    # Step 1: Check and download MP16 dataset if necessary
    print("\n[Step 1] Checking MP16 dataset...")
    check_mp16_data(config)
    
    # Step 2: Train the attack on real data
    print("\n[Step 2] Training the attack on real data...")
    train_config = TRAIN_CONFIG.copy()
    train_config.update({
        "output_dir": attack_dir,
        "cache_dir": cache_dir,
        "mp16_dir": config["mp16_dir"],
        "mp16_metadata": config["mp16_metadata"],
        "num_locations": config["num_locations"],
        "use_real_images": config["use_real_images"],
        "max_real_images": config["max_real_images"],
        "attack_type": config["attack_type"],
        "epsilon": config["epsilon"],
        "alpha": config["alpha"],
        "n_iter": config["n_iter"],
        "n_restarts": config["n_restarts"],
        "batch_size": config["batch_size"],
        "targeted": config["targeted"],
        "target_location": config["target_location"],
        "vis_freq": config["vis_freq"],
        "save_freq": config["save_freq"]
    })
    
    train_result = train_real_data_attack(train_config)
    
    # Step 3: Evaluate the attack on real data
    print("\n[Step 3] Evaluating the attack on real data...")
    eval_config = EVAL_CONFIG.copy()
    eval_config.update({
        "perturbation_path": train_result["perturbation_path"],
        "output_dir": eval_dir,
        "cache_dir": cache_dir,
        "mp16_dir": config["mp16_dir"],
        "mp16_metadata": config["mp16_metadata"],
        "num_locations": config["eval_num_locations"],
        "use_real_images": config["use_real_images"],
        "max_real_images": config["max_real_images"],
        "batch_size": config["batch_size"],
        "success_threshold": config["success_threshold"],
        "num_vis": config["num_vis"],
        "save_predictions": config["save_predictions"]
    })
    
    eval_result = evaluate_real_data_attack(eval_config)
    
    # Print summary
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"Attack Type: {config['attack_type'].upper()}")
    print(f"Epsilon: {config['epsilon']}")
    print(f"Using Real Images: {config['use_real_images']}")
    if config['use_real_images']:
        print(f"MP16 Dataset: {config['mp16_metadata']}")
        print(f"Max Real Images: {config['max_real_images']}")
    if config['targeted']:
        print(f"Target Location: {config['target_location']}")
    print(f"Success Rate: {eval_result['success_rate']:.4f} ({eval_result['success_count']}/{eval_result['total_images']})")
    print(f"Average Distance: {eval_result['avg_distance_km']:.2f} km")
    print(f"Results saved to: {eval_dir}")
    print("=" * 50)
    
    return {
        "train_result": train_result,
        "eval_result": eval_result
    }

if __name__ == "__main__":
    # You can modify the PIPELINE_CONFIG dictionary here before running
    # For example:
    # PIPELINE_CONFIG["attack_type"] = "universal"
    # PIPELINE_CONFIG["epsilon"] = 0.05
    # PIPELINE_CONFIG["targeted"] = True
    # PIPELINE_CONFIG["target_location"] = [40.7128, -74.0060]  # New York City
    # PIPELINE_CONFIG["success_threshold"] = 50.0  # 50 km threshold
    # PIPELINE_CONFIG["use_real_images"] = True  # Use real images from MP16-Pro
    # PIPELINE_CONFIG["download_mp16"] = True  # Download MP16 dataset if not present
    # PIPELINE_CONFIG["mp16_sample_size"] = 10  # Number of images to sample
    
    run_pipeline() 