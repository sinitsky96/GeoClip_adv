import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm, trange
from sparse_rs.util import haversine_distance, CONTINENT_R, STREET_R, CITY_R, REGION_R, COUNTRY_R


class AdvRunner:
    def __init__(self, model, attack, l0_hist_limits, data_RGB_size, device, dtype, verbose=False, targeted=False):
        print("\n=== Initializing AdvRunner ===")
        print(f"Device: {device}")
        print(f"Data RGB size: {data_RGB_size}")
        print(f"Attack type: {'Targeted' if targeted else 'Untargeted'}")
        
        self.attack = attack
        self.model = model
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.targeted = targeted
        
        # Set targeted attribute in the attack object
        self.attack.targeted = targeted
        
        # Set distance thresholds
        self.continent_threshold = CONTINENT_R
        self.street_threshold = STREET_R
        self.city_threshold = CITY_R
        self.region_threshold = REGION_R
        self.country_threshold = COUNTRY_R
        
        # Initialize distance tracking
        self.clean_distances = None
        self.y_orig = None
        self.distance_changes = None
        
        self.l0_norms = self.attack.output_l0_norms
        self.n_l0_norms = len(self.l0_norms)
        
        print(f"L0 norms: {self.l0_norms}")
        print(f"Number of L0 norms: {self.n_l0_norms}")
        
        self.l0_hist_limits = l0_hist_limits
        self.data_channels = 3
        self.data_RGB_size = torch.tensor(data_RGB_size).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.attack_restarts = self.attack.n_restarts
        self.attack_iter = self.attack.n_iter
        self.attack_report_info = self.attack.report_info
        self.attack_name = self.attack.name
        
        print(f"Attack name: {self.attack_name}")
        print(f"Number of restarts: {self.attack_restarts}")
        print(f"Number of iterations: {self.attack_iter}")
        print("=== AdvRunner Initialization Complete ===\n")

    def run_clean_evaluation(self, x_orig, y_orig, n_examples, bs, n_batches, orig_device):
        print("\n=== Starting Clean Evaluation ===")
        robust_flags = torch.zeros(n_examples, dtype=torch.bool, device=orig_device)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, n_examples)
            print(f"Processing clean batch {batch_idx+1}/{n_batches} (samples {start_idx} to {end_idx})")

            x = x_orig[start_idx:end_idx, :].clone().detach().to(self.device)
            y = y_orig[start_idx:end_idx].clone().detach().to(self.device)
            
            # Get predictions from GeoClip using predict_from_tensor
            output, _ = self.model.predict_from_tensor(x)
            print(f"Clean batch predictions shape: {output.shape}")
            
            # Calculate distances between predicted and true coordinates
            distances = haversine_distance(output, y)
            print(f"Mean distance in batch: {distances.mean():.2f} km")
            
            # For untargeted attacks, we want predictions within continent radius
            # For targeted attacks, we want predictions outside street radius
            if not self.targeted:
                success = (distances > CONTINENT_R)  # Attack success means prediction is far from true location
                robust = (distances <= CONTINENT_R)  # Robust means prediction is close to true location
                print(f"Untargeted - Predictions within continent radius: {robust.float().mean()*100:.2f}%")
            else:
                success = (distances <= STREET_R)  # Attack success means prediction is close to target
                robust = (distances > STREET_R)  # Robust means prediction is far from target
                print(f"Targeted - Predictions outside street radius: {robust.float().mean()*100:.2f}%")
                
            robust_flags[start_idx:end_idx] = robust

        n_robust_examples = torch.sum(robust_flags).item()
        init_accuracy = n_robust_examples / n_examples
        print(f"=== Clean Evaluation Results ===")
        print(f"Initial accuracy: {init_accuracy:.2%}")
        print(f"Number of robust examples: {n_robust_examples}/{n_examples}")
        return robust_flags, n_robust_examples, init_accuracy

    def process_results(self, n_examples, robust_flags, l0_norms_adv_perts, perts_l0_norms, l0_norms_robust_flags):
        # Ensure all tensors are on the same device and correct dtype
        device = l0_norms_adv_perts.device
        robust_flags = robust_flags.to(device)
        perts_l0_norms = perts_l0_norms.to(device, dtype=torch.float32)
        l0_norms_robust_flags = l0_norms_robust_flags.to(device)
        data_RGB_size = self.data_RGB_size.to(device)

        # Calculate robust accuracy for each L0 norm
        l0_norms_robust_accuracy = (l0_norms_robust_flags.sum(dim=1).float() / n_examples).tolist()
        
        # Calculate L0 norm statistics
        l0_norms_perts_max_l0 = perts_l0_norms[:, robust_flags].max(dim=1)[0].tolist()
        l0_norms_perts_min_l0 = perts_l0_norms[:, robust_flags].min(dim=1)[0].tolist()
        l0_norms_perts_mean_l0 = perts_l0_norms[:, robust_flags].mean(dim=1).tolist()
        l0_norms_perts_median_l0 = perts_l0_norms[:, robust_flags].median(dim=1)[0].tolist()
        
        # Calculate L_inf norm
        l0_norms_perts_max_l_inf = (l0_norms_adv_perts[:, robust_flags].abs() / data_RGB_size).view(self.n_l0_norms, -1).max(dim=1)[0].tolist()
        
        # Store L0 norm statistics
        l0_norms_perts_info = (
            l0_norms_perts_max_l0,
            l0_norms_perts_min_l0,
            l0_norms_perts_mean_l0,
            l0_norms_perts_median_l0,
            l0_norms_perts_max_l_inf
        )

        # Calculate distance statistics for each L0 norm
        l0_norms_distances = []
        l0_norms_distance_changes = []
        
        # Move y_orig and clean_distances to the correct device
        y_orig = self.y_orig.to(device)
        clean_distances = self.clean_distances.to(device)
        
        for l0_idx in range(self.n_l0_norms):
            # Get adversarial examples for this L0 norm
            x_adv = l0_norms_adv_perts[l0_idx]
            
            # Get predictions for adversarial examples
            with torch.no_grad():
                pred_coords, _ = self.model.predict_from_tensor(x_adv)
            
            # Calculate distances (all tensors now on same device)
            distances = haversine_distance(pred_coords, y_orig)
            distance_changes = distances - clean_distances
            
            # Store statistics
            l0_norms_distances.append({
                'mean': distances.mean().item(),
                'median': distances.median().item(),
                'min': distances.min().item(),
                'max': distances.max().item()
            })
            
            l0_norms_distance_changes.append({
                'mean': distance_changes.mean().item(),
                'median': distance_changes.median().item(),
                'min': distance_changes.min().item(),
                'max': distance_changes.max().item(),
                'positive_change_ratio': (distance_changes > 0).float().mean().item()
            })
            
            # Clean up GPU memory
            del x_adv, pred_coords, distances, distance_changes
            torch.cuda.empty_cache()

        # Clean up GPU memory
        del y_orig, clean_distances
        torch.cuda.empty_cache()

        return (
            l0_norms_robust_accuracy,
            l0_norms_perts_info,
            l0_norms_distances,
            l0_norms_distance_changes
        )

    def report_attack_results(self, l0_norm, metrics):
        """Report detailed metrics for a specific L0 norm"""
        print(f"\nMetrics for L0 norm {l0_norm}:")
        
        # Distance threshold metrics
        print("Percentage of predictions within:")
        print(f"  - Continent ({self.continent_threshold} km): {metrics['continent_success']*100:.2f}%")
        print(f"  - Country ({self.country_threshold} km): {metrics['country_success']*100:.2f}%")
        print(f"  - Region ({self.region_threshold} km): {metrics['region_success']*100:.2f}%")
        print(f"  - City ({self.city_threshold} km): {metrics['city_success']*100:.2f}%")
        print(f"  - Street ({self.street_threshold} km): {metrics['street_success']*100:.2f}%")
        
        # Distance metrics
        print("Distance metrics:")
        print(f"  - Mean: {metrics['mean_distance']:.2f} km")
        print(f"  - Median: {metrics['median_distance']:.2f} km")
        
        # L0 norm metrics
        print("L0 norm metrics:")
        print(f"  - Robust accuracy: {metrics['robust_accuracy']:.4f}")
        print(f"  - Max L0: {metrics['max_l0']:.1f}")
        print(f"  - Min L0: {metrics['min_l0']:.1f}")
        print(f"  - Mean L0: {metrics['mean_l0']:.2f}")
        print(f"  - Median L0: {metrics['median_l0']:.1f}")
        print(f"  - Max L_inf: {metrics['max_linf']:.4f}")

    def run_standard_evaluation(self, x, y, n_examples=None, bs=None):
        torch.cuda.empty_cache()
        
        orig_device = x.device
        x = x.to(self.device)
        y = y.to(self.device)
        
        if n_examples is not None:
            x = x[:n_examples]
            y = y[:n_examples]
        
        # Use smaller batch size for memory efficiency
        bs = min(bs if bs is not None else x.shape[0], 2)  # Process max 2 samples at a time
        n_batches = int(np.ceil(x.shape[0] / bs))
        
        # Initialize tensors to store results (moved to CPU to save GPU memory)
        all_perts = torch.zeros((self.n_l0_norms, *x.shape), dtype=x.dtype, device='cpu')
        all_succ = torch.zeros((self.n_l0_norms, 1, 1, x.shape[0]), dtype=torch.bool, device='cpu')
        all_loss = torch.zeros((self.n_l0_norms, x.shape[0]), dtype=torch.float, device='cpu')
        all_distances = torch.zeros((self.n_l0_norms, x.shape[0]), dtype=torch.float, device='cpu')
        all_distances_clean = torch.zeros(x.shape[0], dtype=torch.float, device='cpu')
        all_success = torch.zeros(x.shape[0], dtype=torch.bool, device='cpu')
        
        # Initialize timing metrics
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_batches)]
        
        print("\n=== Starting Clean Evaluation ===")
        with torch.amp.autocast(device_type='cuda'):
            for batch_idx in range(n_batches):
                torch.cuda.empty_cache()  # Clear cache before each batch
                
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, x.shape[0])
                print(f"Processing clean batch {batch_idx + 1}/{n_batches} (samples {start_idx} to {end_idx})")
                
                x_batch = x[start_idx:end_idx].clone()
                y_batch = y[start_idx:end_idx].clone()
                
                # Get clean predictions
                with torch.no_grad():  # Add no_grad for inference
                    pred_coords, _ = self.model.predict_from_tensor(x_batch)
                print(f"Clean batch predictions shape: {pred_coords.shape}")
                
                # Calculate distances and store them
                distances = haversine_distance(pred_coords, y_batch)
                all_distances_clean[start_idx:end_idx] = distances.cpu()  # Move to CPU
                print(f"Mean distance in batch: {distances.mean().item():.2f} km")
                
                if not self.targeted:
                    success = (distances > self.continent_threshold)
                    robust = (distances <= self.continent_threshold)
                    
                    # Calculate success at different distance thresholds
                    continent_success = (distances <= self.continent_threshold)
                    country_success = (distances <= self.country_threshold)
                    region_success = (distances <= self.region_threshold)
                    city_success = (distances <= self.city_threshold)
                    street_success = (distances <= self.street_threshold)
                    
                    print(f"Clean predictions within:")
                    print(f"  - Continent ({self.continent_threshold} km): {continent_success.float().mean().item()*100:.2f}%")
                    print(f"  - Country ({self.country_threshold} km): {country_success.float().mean().item()*100:.2f}%")
                    print(f"  - Region ({self.region_threshold} km): {region_success.float().mean().item()*100:.2f}%")
                    print(f"  - City ({self.city_threshold} km): {city_success.float().mean().item()*100:.2f}%")
                    print(f"  - Street ({self.street_threshold} km): {street_success.float().mean().item()*100:.2f}%")
                else:
                    success = (distances <= self.street_threshold)
                    robust = (distances > self.street_threshold)
                    print(f"Targeted - Predictions outside street radius: {robust.float().mean().item()*100:.2f}%")
                
                all_success[start_idx:end_idx] = success.cpu()  # Move to CPU
                
                del pred_coords, distances, success, robust
                if not self.targeted:
                    del continent_success, country_success, region_success, city_success, street_success
                torch.cuda.empty_cache()
        
        init_accuracy = (~all_success).float().mean().item()
        print("=== Clean Evaluation Results ===")
        print(f"Initial accuracy: {init_accuracy*100:.2f}%")
        print(f"Number of robust examples: {(~all_success).sum().item()}/{x.shape[0]}")

        print("\n=== Starting Attack ===")
        print(f"Attack type: {'Untargeted'}")
        print(f"Number of L0 norms to test: {self.n_l0_norms}")
        print(f"L0 norms values: {self.l0_norms}")
        
        # Initialize adversarial results (on CPU)
        x_adv = x.cpu()
        y_adv = y.cpu()
        l0_norms_adv_x = torch.zeros((self.n_l0_norms, *x.shape), dtype=x.dtype, device='cpu')
        l0_norms_adv_y = torch.zeros((self.n_l0_norms, *y.shape), dtype=y.dtype, device='cpu')
        
        # Initialize accuracy and loss tracking (on CPU)
        n_steps = self.attack.n_iter + 1
        l0_norms_acc_steps = torch.ones((self.n_l0_norms, 1, n_steps), dtype=torch.float32, device='cpu')
        l0_norms_loss_steps = torch.zeros((self.n_l0_norms, 1, n_steps), dtype=torch.float32, device='cpu')
        l0_norms_acc_steps[..., 0] = init_accuracy
        
        # Store original coordinates for distance calculations
        self.y_orig = y.cpu()
        self.clean_distances = all_distances_clean
        
        # Process L0 norms in chunks to save memory
        chunk_size = 1  # Process one L0 norm at a time
        
        # Run attack
        for batch_idx in range(n_batches):
            torch.cuda.empty_cache()  # Clear cache before each batch
            
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, x.shape[0])
            print(f"\nProcessing attack batch {batch_idx + 1}/{n_batches} (samples {start_idx} to {end_idx})")
            
            x_batch = x[start_idx:end_idx].clone()
            y_batch = y[start_idx:end_idx].clone()
            
            start_events[batch_idx].record()
            print("Running attack.perturb...")
            batch_adv_perts, batch_l0_norms, batch_l_inf_norms = self.attack.perturb(x_batch, y_batch)
            end_events[batch_idx].record()
            
            # Store perturbations on CPU
            all_perts[:, start_idx:end_idx] = batch_adv_perts.cpu()
            
            # Process L0 norms in chunks
            for chunk_start in range(0, self.n_l0_norms, chunk_size):
                torch.cuda.empty_cache()  # Clear cache before each chunk
                
                chunk_end = min(chunk_start + chunk_size, self.n_l0_norms)
                print(f"Processing L0 norms {chunk_start} to {chunk_end-1}")
                
                for l0_idx in range(chunk_start, chunk_end):
                    # Move only necessary data to GPU
                    x_adv_batch = (x_batch + batch_adv_perts[l0_idx]).contiguous()
                    
                    with torch.no_grad():  # Add no_grad for inference
                        pred_coords, _ = self.model.predict_from_tensor(x_adv_batch)
                    
                    # Calculate distances and success
                    distances = haversine_distance(pred_coords, y_batch)
                    success = (distances > self.continent_threshold) if not self.targeted else (distances <= self.street_threshold)
                    
                    # Store results on CPU
                    all_succ[l0_idx, 0, 0, start_idx:end_idx] = success.cpu()
                    all_loss[l0_idx, start_idx:end_idx] = distances.cpu()
                    all_distances[l0_idx, start_idx:end_idx] = distances.cpu()
                    
                    # Store adversarial examples and predictions on CPU
                    l0_norms_adv_x[l0_idx, start_idx:end_idx] = x_adv_batch.cpu()
                    l0_norms_adv_y[l0_idx, start_idx:end_idx] = pred_coords.cpu()
                    
                    # Update accuracy and loss for this L0 norm
                    curr_acc = 1.0 - success.float().mean().item()
                    curr_loss = distances.mean().item()
                    l0_norms_acc_steps[l0_idx, 0, -1] = curr_acc
                    l0_norms_loss_steps[l0_idx, 0, -1] = curr_loss
                    
                    del x_adv_batch, pred_coords, distances, success
                    torch.cuda.empty_cache()
            
            del x_batch, y_batch, batch_adv_perts
            torch.cuda.empty_cache()
        
        # Calculate timing statistics
        torch.cuda.synchronize()
        batch_times = [s.elapsed_time(e) / 1000.0 for s, e in zip(start_events, end_events)]
        adv_batch_compute_time_mean = np.mean(batch_times)
        adv_batch_compute_time_std = np.std(batch_times)
        tot_adv_compute_time = np.sum(batch_times)
        tot_adv_compute_time_std = np.std([time * n_batches for time in batch_times])
        
        # Calculate robust accuracy and perturbation statistics
        robust_flags = ~all_success
        l0_norms_robust_flags = torch.zeros((self.n_l0_norms, x.shape[0]), dtype=torch.bool, device='cpu')
        for l0_idx in range(self.n_l0_norms):
            l0_norms_robust_flags[l0_idx] = ~all_succ[l0_idx, 0, 0]
        
        # Move necessary tensors to GPU for processing
        with torch.cuda.device(self.device):
            l0_norms_robust_accuracy, l0_norms_perts_info, l0_norms_distances, l0_norms_distance_changes = self.process_results(
                x.shape[0],
                robust_flags.to(self.device),
                all_perts.to(self.device),
                torch.tensor([self.l0_norms] * x.shape[0], device=self.device).t(),
                l0_norms_robust_flags.to(self.device)
            )
        
        # Final cleanup
        torch.cuda.empty_cache()
        
        return (init_accuracy, x_adv, y_adv, l0_norms_adv_x, l0_norms_adv_y,
                l0_norms_robust_accuracy, l0_norms_acc_steps, l0_norms_loss_steps, l0_norms_perts_info,
                adv_batch_compute_time_mean, adv_batch_compute_time_std,
                tot_adv_compute_time, tot_adv_compute_time_std, all_distances_clean)