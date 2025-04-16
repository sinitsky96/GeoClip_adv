import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import gc
from parser import get_args, save_img_tensors
from AdvRunner import AdvRunner
from sparse_rs.util import haversine_distance, CONTINENT_R, STREET_R, CITY_R, REGION_R, COUNTRY_R
from geoclip.model.GeoCLIP import GeoCLIP
import json
from datetime import datetime
from torchvision.utils import save_image
import numpy as np

def save_attack_results(save_dir, results_dict):
    """Save results to a JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(save_dir, f'attack_results_{timestamp}.json')
    
    # Convert any torch tensors or numpy arrays to lists
    for k, v in results_dict.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            results_dict[k] = v.tolist() if hasattr(v, 'tolist') else v
    
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"\nResults saved to: {filename}")

def save_images_and_perturbations(save_dir, original_images, perturbations, perturbed_images, batch_idx=0):
    """Save original images, perturbations, and perturbed images with enhanced visibility"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create subdirectories
    orig_dir = os.path.join(save_dir, 'original_images')
    pert_dir = os.path.join(save_dir, 'perturbations')
    adv_dir = os.path.join(save_dir, 'perturbed_images')
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(pert_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)
    
    # Save each image in the batch
    for i in range(original_images.size(0)):
        # Save original image
        save_image(original_images[i], 
                  os.path.join(orig_dir, f'original_{batch_idx}_{i}_{timestamp}.png'))
        
        # Get perturbation and count actual perturbed pixels
        pert = perturbations[i]
        # Count unique pixel locations that are perturbed (across all channels)
        perturbed_locations = (pert.abs().sum(dim=0) > 1e-6).float()
        num_perturbed = perturbed_locations.sum().item()
        print(f"\nActual number of perturbed pixel locations in perturbation {i}: {num_perturbed}")
        
        # Create binary mask for perturbed pixels
        pert_magnitude = pert.abs().sum(dim=0)  # Sum across channels
        mask = (pert_magnitude > 1e-6).float()  # Binary mask of perturbed pixels
        
        # Visualize perturbation (enhanced for visibility)
        pert_vis = pert.clone()
        # Scale perturbations to full range for better visibility
        pert_vis = 10.0 * (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
        pert_vis = torch.clamp(pert_vis, 0, 1)
        save_image(pert_vis,
                  os.path.join(pert_dir, f'perturbation_{batch_idx}_{i}_{timestamp}.png'))
        
        # Create perturbed image with enhanced perturbations
        # Method 1: Add scaled perturbation
        perturbed = original_images[i] + 5.0 * perturbations[i]
        perturbed = torch.clamp(perturbed, 0, 1)
        save_image(perturbed,
                  os.path.join(adv_dir, f'perturbed_{batch_idx}_{i}_{timestamp}.png'))
        
        # Method 2: Highlight perturbed regions with a color overlay
        perturbed_highlight = original_images[i].clone()
        # Add a red tint to perturbed pixels
        perturbed_highlight[0] = torch.where(mask > 0, 
                                           torch.clamp(perturbed_highlight[0] + 0.5, 0, 1),
                                           perturbed_highlight[0])
        save_image(perturbed_highlight,
                  os.path.join(adv_dir, f'perturbed_highlighted_{batch_idx}_{i}_{timestamp}.png'))
        
        # Print number of perturbed pixels for verification
        print(f"Number of perturbed pixels in image {i}: {mask.sum().item()}")
    
    print(f"\nImages saved in: {save_dir}")

def run_adv_attacks(args):
    print(f'Running evaluation of adversarial attacks:')
    adv_runner = AdvRunner(args.model, args.attack_obj, args.l0_hist_limits, args.data_RGB_size,
                           device=args.device, dtype=args.dtype, verbose=True, targeted=args.targeted)
    print(f'Dataset: {args.dataset}, Model: GeoClip,\n'
          f'Attack: {args.attack_name} with L0 sparsity={args.sparsity} and L_inf epsilon={args.eps_l_inf},\n'
          f'Attack iterations={args.n_iter} and restarts={args.n_restarts}')

    args.attack_obj.report_schematics()
    att_l0_norms = args.attack_obj.output_l0_norms
    att_report_info = args.attack_obj.report_info

    # Initialize lists to store distances
    all_distances_clean = []
    all_distances_adv = []

    # Get results from standard evaluation
    (init_accuracy, x_adv, y_adv, l0_norms_adv_x, l0_norms_adv_y,
     l0_norms_robust_accuracy, l0_norms_acc_steps, l0_norms_loss_steps, l0_norms_perts_info,
     adv_batch_compute_time_mean, adv_batch_compute_time_std,
     tot_adv_compute_time, tot_adv_compute_time_std, all_distances_clean) = adv_runner.run_standard_evaluation(
        args.x_test, args.y_test, args.n_examples, bs=args.batch_size)

    # No need to concatenate all_distances_clean since it's already a tensor
    distances_clean = all_distances_clean  # Already in correct shape [n_examples]

    # Unpack perturbation info
    (l0_norms_perts_max_l0, l0_norms_perts_min_l0, l0_norms_perts_mean_l0,
     l0_norms_perts_median_l0, l0_norms_perts_max_l_inf) = l0_norms_perts_info

    # Create results dictionary
    results = {
        'attack_config': {
            'dataset': args.dataset,
            'attack_name': args.attack_name,
            'sparsity': args.sparsity,
            'eps_l_inf': args.eps_l_inf,
            'n_iter': args.n_iter,
            'n_restarts': args.n_restarts,
            'n_examples': args.n_examples,
            'kernel_size': args.attack_obj.kernel_size if hasattr(args.attack_obj, 'kernel_size') else None
        },
        'metrics': {}
    }

    # Now process adversarial results for each L0 norm
    l0_norms_adv_loss = []
    for l0_norm_idx, l0_norm in enumerate(att_l0_norms):
        norm_results = {}
        
        if l0_norm < args.data_pixels:
            print("\nReporting results for sparse adversarial attack with L0 norm limitation:")
        else:
            print("\nReporting results for non-sparse adversarial attack (L0 norm limitation ineffective):")
        
        # Calculate distance-based metrics for GeoClip
        distances = haversine_distance(l0_norms_adv_y[l0_norm_idx], args.y_test)
        all_distances_adv.append(distances)
        percent_continent = (distances <= CONTINENT_R).float().mean().item() * 100
        percent_country = (distances <= COUNTRY_R).float().mean().item() * 100
        percent_region = (distances <= REGION_R).float().mean().item() * 100
        percent_city = (distances <= CITY_R).float().mean().item() * 100
        percent_street = (distances <= STREET_R).float().mean().item() * 100
        
        # Store metrics in results dictionary
        norm_results.update({
            'l0_norm': l0_norm,
            'percent_continent': percent_continent,
            'percent_country': percent_country,
            'percent_region': percent_region,
            'percent_city': percent_city,
            'percent_street': percent_street,
            'mean_distance': distances.mean().item(),
            'median_distance': distances.median().item(),
            'robust_accuracy': l0_norms_robust_accuracy[l0_norm_idx],
            'max_l0': l0_norms_perts_max_l0[l0_norm_idx],
            'min_l0': l0_norms_perts_min_l0[l0_norm_idx],
            'mean_l0': l0_norms_perts_mean_l0[l0_norm_idx],
            'median_l0': l0_norms_perts_median_l0[l0_norm_idx],
            'max_l_inf': l0_norms_perts_max_l_inf[l0_norm_idx]
        })
        
        # Print metrics
        print(f'\nMetrics for L0 norm {l0_norm}:')
        print(f'Percentage of predictions within:')
        print(f'  - Continent (2500 km): {percent_continent:.2f}%')
        print(f'  - Country (750 km): {percent_country:.2f}%')
        print(f'  - Region (200 km): {percent_region:.2f}%')
        print(f'  - City (25 km): {percent_city:.2f}%')
        print(f'  - Street (1 km): {percent_street:.2f}%')
        print(f'Distance metrics:')
        print(f'  - Mean: {distances.mean().item():.2f} km')
        print(f'  - Median: {distances.median().item():.2f} km')
        print(f'L0 norm metrics:')
        print(f'  - Robust accuracy: {l0_norms_robust_accuracy[l0_norm_idx]:.4f}')
        print(f'  - Max L0: {l0_norms_perts_max_l0[l0_norm_idx]}')
        print(f'  - Min L0: {l0_norms_perts_min_l0[l0_norm_idx]}')
        print(f'  - Mean L0: {l0_norms_perts_mean_l0[l0_norm_idx]:.2f}')
        print(f'  - Median L0: {l0_norms_perts_median_l0[l0_norm_idx]}')
        print(f'  - Max L_inf: {l0_norms_perts_max_l_inf[l0_norm_idx]:.4f}')
        
        if att_report_info:
            norm_results.update({
                'accuracy_steps': l0_norms_acc_steps[l0_norm_idx].tolist(),
                'loss_steps': l0_norms_loss_steps[l0_norm_idx].tolist()
            })
            l0_norms_adv_loss.append(l0_norms_loss_steps[l0_norm_idx, -1, -1].item())
        
        results['metrics'][f'l0_norm_{l0_norm}'] = norm_results
        
        # Save images for this L0 norm
        if args.save_results:
            attack_type = "targeted" if args.targeted else "untargeted"
            # Get kernel info from the attack object if it's a kernel-based attack
            if hasattr(args.attack_obj, 'kernel_size'):
                kernel_info = f"kernel_{args.attack_obj.kernel_size}x{args.attack_obj.kernel_size}"
            else:
                kernel_info = "no_kernel"
            
            # Only save images for the final sparsity level
            if l0_norm == args.sparsity:
                save_dir = os.path.join(args.results_dir, 
                                      f"{args.dataset}_{attack_type}_{kernel_info}_sparsity_{args.sparsity}")
                save_images_and_perturbations(
                    save_dir,
                    args.x_test.to(x_adv.device),
                    x_adv - args.x_test.to(x_adv.device),
                    x_adv
                )

    # After calculating distances_clean and distances_adv, add detailed statistics
    if isinstance(args.model, GeoCLIP):
        # Reshape adversarial distances to match clean distances
        # We need to select the distances for the final sparsity level only
        final_distances_adv = all_distances_adv[-1]  # Take the last one (corresponding to final sparsity)
        
        # Verify shapes match
        if distances_clean.shape != final_distances_adv.shape:
            print(f"Warning: Clean distances shape {distances_clean.shape} doesn't match adversarial distances shape {final_distances_adv.shape}")
            # Ensure we're using the same number of examples for both
            min_size = min(distances_clean.size(0), final_distances_adv.size(0))
            distances_clean = distances_clean[:min_size]
            final_distances_adv = final_distances_adv[:min_size]
        
        # Ensure both tensors are on the same device (GPU)
        distances_clean = distances_clean.to(args.device)
        final_distances_adv = final_distances_adv.to(args.device)
        
        # Calculate accuracy metrics
        if args.targeted:
            acc = (final_distances_adv > STREET_R).float().sum().item()
        else:
            acc = (final_distances_adv <= CONTINENT_R).float().sum().item()
        
        print("\nGeoCLIP Attack Statistics")
        print("------------------------")
        print(f'Robust accuracy: {acc / args.n_examples:.2%}')
        
        # Print statistics relative to target or true location
        target_str = f"targeted location: {args.target_class}" if args.targeted else "true location of the examples"
        print(f"\nThe following statistics are relative to the {target_str}:")
        
        # Calculate improvements
        if args.targeted:
            improvement = distances_clean - final_distances_adv
        else:
            improvement = final_distances_adv - distances_clean
            
        neg_mask = improvement <= 0
        pos_mask = improvement > 0
        final_distances_adv[neg_mask] = distances_clean[neg_mask]  # filter bad adversarial attacks
        
        # Define thresholds and labels
        thresholds = [STREET_R, CITY_R, REGION_R, COUNTRY_R, CONTINENT_R]
        labels = {
            STREET_R: "STREET_R (1 km)",
            CITY_R: "CITY_R (25 km)",
            REGION_R: "REGION_R (200 km)",
            COUNTRY_R: "COUNTRY_R (750 km)",
            CONTINENT_R: "CONTINENT_R (2500 km)"
        }
        
        # Print clean vs adversarial predictions within each threshold
        print("\nPrediction Distance Thresholds:")
        for T in thresholds:
            percent_T_clean = (distances_clean <= T).float().mean().item() * 100.0
            percent_T_adv = (final_distances_adv <= T).float().mean().item() * 100.0
            print(f"- {labels[T]}:")
            print(f"  Clean predictions within threshold: {percent_T_clean:.2f}%")
            print(f"  Adversarial predictions within threshold: {percent_T_adv:.2f}%")
            print(f"  Change: {percent_T_adv - percent_T_clean:.2f}%")
        
        # Print improvement statistics
        print("\nDistance Change Statistics:")
        percent_improved = pos_mask.float().mean().item() * 100.0
        print(f"Percentage of examples with positively changed distance: {percent_improved:.2f}%")
        
        if pos_mask.sum() > 0:
            avg_improvement = improvement[pos_mask].mean().item()
            median_improvement = improvement[pos_mask].median().item()
            print(f"For successfully perturbed examples:")
            print(f"- Average distance change: {avg_improvement:.2f} km")
            print(f"- Median distance change: {median_improvement:.2f} km")
        
        # Overall statistics
        print("\nOverall Statistics:")
        print(f"Average distance change across all examples: {improvement.mean().item():.2f} km")
        print(f"Median distance change across all examples: {improvement.median().item():.2f} km")
        
        # Add these statistics to the results dictionary
        results['attack_statistics'] = {
            'robust_accuracy': float(acc / args.n_examples),
            'thresholds': {
                labels[T]: {
                    'clean_percent': float((distances_clean <= T).float().mean().item() * 100.0),
                    'adv_percent': float((final_distances_adv <= T).float().mean().item() * 100.0),
                    'change': float(((final_distances_adv <= T).float().mean().item() - (distances_clean <= T).float().mean().item()) * 100.0)
                } for T in thresholds
            },
            'improvement_stats': {
                'percent_improved': float(percent_improved),
                'avg_improvement_successful': float(avg_improvement) if pos_mask.sum() > 0 else 0.0,
                'median_improvement_successful': float(median_improvement) if pos_mask.sum() > 0 else 0.0,
                'avg_improvement_overall': float(improvement.mean().item()),
                'median_improvement_overall': float(improvement.median().item())
            }
        }

    # Add runtime metrics
    results['runtime'] = {
        'batch_compute_time_mean': adv_batch_compute_time_mean,
        'batch_compute_time_std': adv_batch_compute_time_std,
        'total_compute_time': tot_adv_compute_time,
        'total_compute_time_std': tot_adv_compute_time_std
    }

    # Save all results
    if args.save_results:
        save_attack_results(args.results_dir, results)

    # Cleanup
    del init_accuracy, adv_runner, x_adv, l0_norms_adv_x
    del l0_norms_perts_max_l0, l0_norms_perts_min_l0, l0_norms_perts_mean_l0
    del l0_norms_perts_median_l0, l0_norms_perts_max_l_inf, l0_norms_robust_accuracy
    del l0_norms_acc_steps, l0_norms_loss_steps, adv_batch_compute_time_mean
    del adv_batch_compute_time_std, tot_adv_compute_time, tot_adv_compute_time_std
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = get_args()
    run_adv_attacks(args)