import os
import torch
import sys
from train_real_data_attack import train_real_data_attack
from evaluate_real_data_attack import evaluate_real_data_attack
from geoclip_adv_attacks.data.mp16_pro.download.download_mp16 import main as download_mp16_data
from CONFIG import CONFIG, create_required_directories

def check_mp16_data():
    """Check if MP16 dataset is downloaded and download if necessary."""
    metadata_path = CONFIG["paths"]["mp16_metadata"]
    if not os.path.exists(metadata_path):
        if CONFIG["data"]["download_mp16"]:
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

def run_pipeline():
    """
    Run the complete adversarial attack pipeline:
    1. Check and download MP16 dataset if necessary
    2. Train the attack on real data
    3. Evaluate the attack on real data
    """
    print("=" * 50)
    print("GEOCLIP REAL DATA ADVERSARIAL ATTACK PIPELINE")
    print("=" * 50)
    
    # Create required directories
    create_required_directories()
    
    # Step 1: Check and download MP16 dataset if necessary
    print("\n[Step 1] Checking MP16 dataset...")
    check_mp16_data()
    
    # Step 2: Train the attack on real data
    print("\n[Step 2] Training the attack on real data...")
    train_result = train_real_data_attack()
    
    # Step 3: Evaluate the attack on real data
    print("\n[Step 3] Evaluating the attack on real data...")
    eval_result = evaluate_real_data_attack()
    
    # Print summary
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"Attack Type: {CONFIG['attack']['type'].upper()}")
    if CONFIG['attack']['type'].lower() == 'patch':
        print(f"Patch Size: {CONFIG['attack']['patch']['size']}")
        print(f"Patch Location: {CONFIG['attack']['patch']['location']}")
        if CONFIG['attack']['patch']['l0_limit'] is not None:
            print(f"L0 Limit: {CONFIG['attack']['patch']['l0_limit']}")
    else:
        print(f"Epsilon: {CONFIG['attack']['epsilon']}")
    print(f"Using Real Images: {CONFIG['data']['use_real_images']}")
    if CONFIG['data']['use_real_images']:
        print(f"MP16 Dataset: {CONFIG['paths']['mp16_metadata']}")
        print(f"Max Real Images: {CONFIG['data']['max_real_images']}")
    if CONFIG['attack']['targeted']:
        print(f"Target Location: {CONFIG['attack']['target_location']}")
    print(f"Success Rate: {eval_result['success_rate']:.4f} ({eval_result['success_count']}/{eval_result['total_images']})")
    print(f"Average Distance: {eval_result['avg_distance_km']:.2f} km")
    print(f"Results saved to: {CONFIG['paths']['eval_output_dir']}")
    print("=" * 50)
    
    return {
        "train_result": train_result,
        "eval_result": eval_result
    }

if __name__ == "__main__":
    run_pipeline() 