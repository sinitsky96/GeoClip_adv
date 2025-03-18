import os
import sys
import argparse
from geoclip_pgdtrim_attack import attack_geoclip

def main():
    parser = argparse.ArgumentParser(description="Run PGDTrim Attack on GeoCLIP")
    
    # Data and output arguments
    parser.add_argument("--data_dir", type=str, 
                        default="D:/Study Docs/Degree Material/Sem 9 proj/GeoClip_adv/geoclip_adv_attacks/data", 
                        help="Directory containing Im2GPS data")
    parser.add_argument("--output_dir", type=str, 
                        default="D:/Study Docs/Degree Material/Sem 9 proj/GeoClip_adv/attack_results", 
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
    parser.add_argument("--attack_type", type=str, default="untargeted", 
                        choices=["untargeted", "targeted"],
                        help="Type of attack to perform")
    parser.add_argument("--target_coords", type=float, nargs=2, default=None, 
                        help="Target coordinates [lat, lon] for targeted attack")
    parser.add_argument("--use_geodesic", action="store_true", default=True,
                        help="Use geodesic distance for untargeted attacks")
    
    # Attack configuration presets
    parser.add_argument("--preset", type=str, default="default", 
                        choices=["default", "aggressive", "conservative", "sparse", "dense"],
                        help="Preset configuration for the attack")
    
    args = parser.parse_args()
    
    # Set up attack parameters based on preset
    attack_args = argparse.Namespace()
    
    # Copy basic arguments
    attack_args.data_dir = args.data_dir
    attack_args.output_dir = args.output_dir
    attack_args.sample_idx = args.sample_idx
    attack_args.save_results = args.save_results
    attack_args.cuda = args.cuda
    attack_args.verbose = args.verbose
    
    # Set targeted attack flag
    attack_args.targeted = (args.attack_type == "targeted")
    attack_args.target_coords = args.target_coords
    attack_args.use_geodesic = args.use_geodesic
    
    # Set preset-specific parameters
    if args.preset == "default":
        # Balanced default settings for L0 attack
        attack_args.eps = 16  # Higher epsilon for L0 attacks
        attack_args.alpha = 4.0
        attack_args.n_iter = 100
        attack_args.n_restarts = 1
        attack_args.no_rand_init = False
        attack_args.sparsity = 100  # Number of pixels to perturb
        attack_args.max_trim_steps = 5
        attack_args.trim_steps_reduce = "even"
        attack_args.dynamic_trim = False
        attack_args.dropout_dist = "bernoulli"
        attack_args.dropout_mean = 1.0
        attack_args.dropout_std = 1.0
        attack_args.post_trim_dpo = False
        attack_args.mask_dist = "multinomial"
        attack_args.mask_prob_amp_rate = 0
        attack_args.norm_mask_amp = False
        attack_args.mask_opt_iter = 0
        attack_args.n_mask_samples = 1000
        attack_args.no_sample_all_masks = False
        attack_args.trim_best_mask = 0
        
    elif args.preset == "aggressive":
        # More aggressive settings for stronger L0 attacks
        attack_args.eps = 32  # Much higher epsilon for L0 attacks
        attack_args.alpha = 8.0
        attack_args.n_iter = 200
        attack_args.n_restarts = 3
        attack_args.no_rand_init = False
        attack_args.sparsity = 200  # More pixels to perturb
        attack_args.max_trim_steps = 8
        attack_args.trim_steps_reduce = "best"
        attack_args.dynamic_trim = True
        attack_args.dropout_dist = "bernoulli"
        attack_args.dropout_mean = 1.0
        attack_args.dropout_std = 1.0
        attack_args.post_trim_dpo = True
        attack_args.mask_dist = "multinomial"
        attack_args.mask_prob_amp_rate = 2
        attack_args.norm_mask_amp = True
        attack_args.mask_opt_iter = 5
        attack_args.n_mask_samples = 2000
        attack_args.no_sample_all_masks = False
        attack_args.trim_best_mask = 1
        
    elif args.preset == "conservative":
        # More conservative settings for subtle L0 attacks
        attack_args.eps = 8
        attack_args.alpha = 2.0
        attack_args.n_iter = 50
        attack_args.n_restarts = 1
        attack_args.no_rand_init = False
        attack_args.sparsity = 50  # Fewer pixels to perturb
        attack_args.max_trim_steps = 3
        attack_args.trim_steps_reduce = "even"
        attack_args.dynamic_trim = False
        attack_args.dropout_dist = "bernoulli"
        attack_args.dropout_mean = 1.0
        attack_args.dropout_std = 1.0
        attack_args.post_trim_dpo = False
        attack_args.mask_dist = "multinomial"
        attack_args.mask_prob_amp_rate = 0
        attack_args.norm_mask_amp = False
        attack_args.mask_opt_iter = 0
        attack_args.n_mask_samples = 500
        attack_args.no_sample_all_masks = False
        attack_args.trim_best_mask = 0
        
    elif args.preset == "sparse":
        # Settings for very sparse L0 attacks (very few pixels)
        attack_args.eps = 64  # Very high epsilon for few pixels
        attack_args.alpha = 16.0
        attack_args.n_iter = 150
        attack_args.n_restarts = 2
        attack_args.no_rand_init = False
        attack_args.sparsity = 20  # Very few pixels
        attack_args.max_trim_steps = 5
        attack_args.trim_steps_reduce = "best"
        attack_args.dynamic_trim = True
        attack_args.dropout_dist = "bernoulli"
        attack_args.dropout_mean = 1.0
        attack_args.dropout_std = 1.0
        attack_args.post_trim_dpo = True
        attack_args.mask_dist = "multinomial"
        attack_args.mask_prob_amp_rate = 3
        attack_args.norm_mask_amp = True
        attack_args.mask_opt_iter = 10
        attack_args.n_mask_samples = 1500
        attack_args.no_sample_all_masks = False
        attack_args.trim_best_mask = 2
        
    elif args.preset == "dense":
        # Settings for denser L0 attacks (more pixels)
        attack_args.eps = 8  # Lower epsilon for many pixels
        attack_args.alpha = 2.0
        attack_args.n_iter = 100
        attack_args.n_restarts = 1
        attack_args.no_rand_init = False
        attack_args.sparsity = 500  # Many pixels
        attack_args.max_trim_steps = 5
        attack_args.trim_steps_reduce = "even"
        attack_args.dynamic_trim = False
        attack_args.dropout_dist = "bernoulli"
        attack_args.dropout_mean = 1.0
        attack_args.dropout_std = 1.0
        attack_args.post_trim_dpo = False
        attack_args.mask_dist = "multinomial"
        attack_args.mask_prob_amp_rate = 0
        attack_args.norm_mask_amp = False
        attack_args.mask_opt_iter = 0
        attack_args.n_mask_samples = 1000
        attack_args.no_sample_all_masks = True
        attack_args.trim_best_mask = 0
    
    # Print attack configuration
    print(f"\nRunning {args.attack_type} attack with '{args.preset}' preset")
    print(f"Parameters:")
    print(f"  - Norm: L0 (sparse attack)")
    print(f"  - Sparsity: {attack_args.sparsity} pixels")
    print(f"  - Epsilon: {attack_args.eps}/255")
    print(f"  - Alpha: {attack_args.alpha}/255")
    print(f"  - Iterations: {attack_args.n_iter}")
    print(f"  - Restarts: {attack_args.n_restarts}")
    
    if attack_args.targeted and attack_args.target_coords is not None:
        print(f"  - Target coordinates: {attack_args.target_coords}")
    elif not attack_args.targeted:
        print(f"  - Using geodesic distance: {attack_args.use_geodesic}")
    
    # Run the attack
    results = attack_geoclip(attack_args)
    
    # Print summary
    print("\nAttack Summary:")
    print(f"Original prediction: {results['orig_pred_coords']} (error: {results['orig_distance_error']:.2f} km)")
    print(f"Adversarial prediction: {results['adv_pred_coords']} (error: {results['adv_distance_error']:.2f} km)")
    print(f"Prediction displacement: {results['displacement']:.2f} km")
    print(f"L0 norm: {results['l0_norm']} pixels")
    print(f"L∞ norm: {results['linf_norm']*255:.2f}/255")
    
    return results

if __name__ == "__main__":
    main() 