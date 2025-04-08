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
            
            # For untargeted attacks, we want to find predictions that are close to ground truth
            # For targeted attacks, we want to find predictions that are far from target
            if not self.targeted:
                correct_batch = (distances <= CONTINENT_R).detach().to(orig_device)
                print(f"Untargeted - Predictions within continent radius: {correct_batch.float().mean()*100:.2f}%")
            else:
                correct_batch = (distances > STREET_R).detach().to(orig_device)
                print(f"Targeted - Predictions outside street radius: {correct_batch.float().mean()*100:.2f}%")
                
            robust_flags[start_idx:end_idx] = correct_batch

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
        perts_l0_norms = perts_l0_norms.to(device, dtype=torch.float32)  # Convert to float32 for mean calculation
        l0_norms_robust_flags = l0_norms_robust_flags.to(device)
        data_RGB_size = self.data_RGB_size.to(device)

        l0_norms_robust_accuracy = (l0_norms_robust_flags.sum(dim=1).float() / n_examples).tolist()
        l0_norms_perts_max_l0 = perts_l0_norms[:, robust_flags].max(dim=1)[0].tolist()
        l0_norms_perts_min_l0 = perts_l0_norms[:, robust_flags].min(dim=1)[0].tolist()
        l0_norms_perts_mean_l0 = perts_l0_norms[:, robust_flags].mean(dim=1).tolist()
        l0_norms_perts_median_l0 = perts_l0_norms[:, robust_flags].median(dim=1)[0].tolist()
        l0_norms_perts_max_l_inf = (l0_norms_adv_perts[:, robust_flags].abs() / data_RGB_size).view(self.n_l0_norms, -1).max(dim=1)[0].tolist()
        l0_norms_perts_info = l0_norms_perts_max_l0, l0_norms_perts_min_l0, l0_norms_perts_mean_l0, l0_norms_perts_median_l0, l0_norms_perts_max_l_inf

        l0_norms_hist_l0_limits = []
        l0_norms_hist_l0_ratio = []
        l0_norms_hist_l0_robust_accuracy = []
        for l0_norm_idx, l0_norm in enumerate(self.l0_norms):
            max_l0 = int(l0_norms_perts_max_l0[l0_norm_idx])
            l0_hist_limits = [l0_limit for l0_limit in self.l0_hist_limits if l0_limit < max_l0] + [max_l0]
            l0_norms_hist_l0_limits.append(l0_hist_limits)
            hist_l0_ratio = []
            hist_l0_robust_accuracy = []
            for l0_limit in l0_hist_limits:
                l0_limit_flags = perts_l0_norms[l0_norm_idx, :].le(l0_limit).logical_or(~robust_flags)
                perts_l0_limited_ratio = (l0_limit_flags.sum().float() / n_examples).item()
                l0_limit_robust_flags = l0_norms_robust_flags[l0_norm_idx, :].logical_or(~l0_limit_flags)
                l0_limit_robust_accuracy = (l0_limit_robust_flags.sum().float() / n_examples).item()
                hist_l0_ratio.append(perts_l0_limited_ratio)
                hist_l0_robust_accuracy.append(l0_limit_robust_accuracy)

            l0_norms_hist_l0_ratio.append(hist_l0_ratio)
            l0_norms_hist_l0_robust_accuracy.append(hist_l0_robust_accuracy)

        return l0_norms_robust_accuracy, l0_norms_perts_info, \
            l0_norms_hist_l0_limits, l0_norms_hist_l0_ratio, l0_norms_hist_l0_robust_accuracy

    def run_standard_evaluation(self, x, y, n_examples=None, bs=None):
        """
        Run standard evaluation of adversarial attacks
        """
        torch.cuda.empty_cache()  # Initial cleanup
        
        orig_device = x.device
        x = x.to(self.device)
        y = y.to(self.device)
        
        if n_examples is not None:
            x = x[:n_examples]
            y = y[:n_examples]
        
        n_batches = int(np.ceil(x.shape[0] / bs)) if bs else 1
        bs = x.shape[0] if bs is None else bs
        
        # Initialize tensors to store results
        all_perts = torch.zeros((self.n_l0_norms, *x.shape), dtype=x.dtype, device=orig_device)
        all_succ = torch.zeros((self.n_l0_norms, 1, 1, x.shape[0]), dtype=torch.bool, device=orig_device)
        all_loss = torch.zeros((self.n_l0_norms, x.shape[0]), dtype=torch.float, device=orig_device)
        
        # Initialize timing metrics
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_batches)]
        
        print("\n=== Starting Clean Evaluation ===")
        with torch.amp.autocast(device_type='cuda'):  # Updated to use recommended format
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, x.shape[0])
                print(f"Processing clean batch {batch_idx + 1}/{n_batches} (samples {start_idx} to {end_idx})")
                
                x_batch = x[start_idx:end_idx].clone()
                y_batch = y[start_idx:end_idx].clone()
                
                # Get clean predictions
                pred_coords, _ = self.model.predict_from_tensor(x_batch)
                print(f"Clean batch predictions shape: {pred_coords.shape}")
                
                # Calculate distances and success
                distances = haversine_distance(pred_coords, y_batch)
                print(f"Mean distance in batch: {distances.mean().item():.2f} km")
                
                # For untargeted attack, success means prediction is far from true location
                success = (distances > self.continent_threshold)
                print(f"Untargeted - Predictions within continent radius: {success.float().mean().item()*100:.2f}%")
                
                # Clean up intermediate tensors
                del pred_coords
                del distances
                torch.cuda.empty_cache()
        
        init_accuracy = (~success).float().mean().item()
        print("=== Clean Evaluation Results ===")
        print(f"Initial accuracy: {init_accuracy*100:.2f}%")
        print(f"Number of robust examples: {(~success).sum().item()}/{x.shape[0]}")
        
        print("\n=== Starting Attack ===")
        print(f"Attack type: {'Untargeted'}")
        print(f"Number of L0 norms to test: {self.n_l0_norms}")
        print(f"L0 norms values: {self.l0_norms}")
        
        # Initialize adversarial results
        x_adv = x.clone()
        y_adv = y.clone()
        l0_norms_adv_x = torch.zeros((self.n_l0_norms, *x.shape), dtype=x.dtype, device=orig_device)
        l0_norms_adv_y = torch.zeros((self.n_l0_norms, *y.shape), dtype=y.dtype, device=orig_device)
        
        # Initialize accuracy and loss tracking for each step
        n_steps = self.attack.n_iter + 1  # +1 for initial state
        l0_norms_acc_steps = torch.ones((self.n_l0_norms, 1, n_steps), dtype=torch.float32, device=orig_device)  # Initialize with 1.0 (100% accuracy)
        l0_norms_loss_steps = torch.zeros((self.n_l0_norms, 1, n_steps), dtype=torch.float32, device=orig_device)
        
        # Set initial accuracy and loss for all L0 norms
        l0_norms_acc_steps[..., 0] = init_accuracy  # Set initial accuracy
        
        for batch_idx in tqdm(range(n_batches)):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, x.shape[0])
            print(f"\nProcessing attack batch {batch_idx + 1}/{n_batches} (samples {start_idx} to {end_idx})")
            
            # Process in smaller chunks to save memory
            with torch.amp.autocast(device_type='cuda'):  # Updated to use recommended format
                x_batch = x[start_idx:end_idx].clone()
                y_batch = y[start_idx:end_idx].clone()
                
                start_events[batch_idx].record()
                print("Running attack.perturb...")
                batch_adv_perts, batch_l0_norms, batch_l_inf_norms = self.attack.perturb(x_batch, y_batch)
                end_events[batch_idx].record()
                
                # Store results and free memory
                all_perts[:, start_idx:end_idx] = batch_adv_perts.to(orig_device)
                del batch_l0_norms, batch_l_inf_norms
                
                # Get predictions for adversarial examples
                print("Getting predictions for adversarial examples...")
                batch_size = end_idx - start_idx
                
                # Process L0 norms in chunks to save memory
                chunk_size = 3  # Process 3 L0 norms at a time
                for chunk_start in range(0, self.n_l0_norms, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, self.n_l0_norms)
                    chunk_size_actual = chunk_end - chunk_start
                    
                    # Create adversarial examples for current chunk
                    x_adv_chunk = x_batch.unsqueeze(0).repeat(chunk_size_actual, 1, 1, 1, 1)
                    x_adv_chunk = x_adv_chunk + batch_adv_perts[chunk_start:chunk_end]
                    
                    # Reshape for batch prediction
                    x_adv_chunk = x_adv_chunk.view(-1, *x_batch.shape[1:])
                    pred_coords, _ = self.model.predict_from_tensor(x_adv_chunk)
                    
                    # Calculate distances and success for chunk
                    y_expanded = y_batch.unsqueeze(0).repeat(chunk_size_actual, 1, 1)
                    y_expanded = y_expanded.view(-1, y_batch.shape[1])
                    distances = haversine_distance(pred_coords, y_expanded)
                    
                    # Reshape and store results for chunk
                    chunk_succ = (distances > self.continent_threshold).view(chunk_size_actual, batch_size)
                    chunk_loss = distances.view(chunk_size_actual, batch_size)
                    
                    all_succ[chunk_start:chunk_end, 0, 0, start_idx:end_idx] = chunk_succ.to(orig_device)
                    all_loss[chunk_start:chunk_end, start_idx:end_idx] = chunk_loss.to(orig_device)
                    
                    # Store adversarial examples and predictions
                    l0_norms_adv_x[chunk_start:chunk_end, start_idx:end_idx] = x_adv_chunk.view(chunk_size_actual, batch_size, *x_batch.shape[1:]).to(orig_device)
                    l0_norms_adv_y[chunk_start:chunk_end, start_idx:end_idx] = pred_coords.view(chunk_size_actual, batch_size, -1).to(orig_device)
                    
                    # Update accuracy and loss steps for this chunk
                    for l0_idx in range(chunk_start, chunk_end):
                        curr_acc = 1.0 - chunk_succ[l0_idx - chunk_start].float().mean().item()
                        curr_loss = chunk_loss[l0_idx - chunk_start].mean().item()
                        # Store in the final step
                        l0_norms_acc_steps[l0_idx, 0, -1] = curr_acc
                        l0_norms_loss_steps[l0_idx, 0, -1] = curr_loss
                    
                    # Clean up chunk tensors
                    del x_adv_chunk, pred_coords, y_expanded, distances, chunk_succ, chunk_loss
                    torch.cuda.empty_cache()
                
                # Store best adversarial example (from smallest L0 norm that succeeded)
                best_idx = torch.where(all_succ[:, 0, 0, start_idx:end_idx])[0]
                if len(best_idx) > 0:
                    best_idx = best_idx[-1]  # Take the smallest successful L0 norm
                    x_adv[start_idx:end_idx] = l0_norms_adv_x[best_idx, start_idx:end_idx]
                    y_adv[start_idx:end_idx] = l0_norms_adv_y[best_idx, start_idx:end_idx]
                
                # Clean up batch tensors
                del x_batch, y_batch, batch_adv_perts
                torch.cuda.empty_cache()
        
        # Calculate timing statistics
        torch.cuda.synchronize()
        batch_times = [s.elapsed_time(e) / 1000.0 for s, e in zip(start_events, end_events)]  # Convert to seconds
        adv_batch_compute_time_mean = np.mean(batch_times)
        adv_batch_compute_time_std = np.std(batch_times)
        tot_adv_compute_time = np.sum(batch_times)
        tot_adv_compute_time_std = np.std([time * n_batches for time in batch_times])
        
        # Calculate robust accuracy and perturbation statistics
        robust_flags = ~success
        l0_norms_robust_flags = torch.zeros((self.n_l0_norms, x.shape[0]), dtype=torch.bool, device=orig_device)
        for l0_idx in range(self.n_l0_norms):
            l0_norms_robust_flags[l0_idx] = ~all_succ[l0_idx, 0, 0]
        
        l0_norms_robust_accuracy, l0_norms_perts_info, _, _, _ = self.process_results(
            x.shape[0], robust_flags, all_perts, 
            torch.tensor([self.l0_norms] * x.shape[0]).t(), l0_norms_robust_flags
        )
        
        # Final cleanup
        torch.cuda.empty_cache()
        
        return (init_accuracy, x_adv, y_adv, l0_norms_adv_x, l0_norms_adv_y,
                l0_norms_robust_accuracy, l0_norms_acc_steps, l0_norms_loss_steps, l0_norms_perts_info,
                adv_batch_compute_time_mean, adv_batch_compute_time_std,
                tot_adv_compute_time, tot_adv_compute_time_std)