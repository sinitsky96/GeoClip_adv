import numpy as np
import math
import itertools
import torch
from attacks.pgd_attacks.PGDTrim import PGDTrim
from sparse_rs.util import haversine_distance, CONTINENT_R


class PGDTrimKernel(PGDTrim):
    def __init__(
            self,
            model,
            criterion,
            misc_args=None,
            pgd_args=None,
            dropout_args=None,
            trim_args=None,
            mask_args=None,
            kernel_args=None):

        self.kernel_size = kernel_args['kernel_size']
        self.n_kernel_pixels = kernel_args['n_kernel_pixels']
        self.kernel_sparsity = kernel_args['kernel_sparsity']
        self.max_kernel_sparsity = kernel_args['max_kernel_sparsity']
        self.kernel_min_active = kernel_args['kernel_min_active']
        self.kernel_group = kernel_args['kernel_group']
        
        super(PGDTrimKernel, self).__init__(model, criterion, misc_args, pgd_args, dropout_args, trim_args, mask_args)
        self.l0_norms_kernels = [norm // self.n_kernel_pixels for norm in self.l0_norms[1:]]
        self.output_l0_norms_kernels = [norm // self.n_kernel_pixels for norm in self.output_l0_norms[1:]]
        self.kernel_w = None
        self.kernel_h = None
        self.n_data_kernels = None
        self.kernel_pad = None
        self.kernel_max_pool = None
        self.kernel_active_pool = None
        self.kernel_active_pool_method = None
        self.prob_pool_method = None
        self.parse_kernel_args()

    def report_schematics(self):
        
        print("Running novel PGDTrimKernel attack")
        print(f"The attack will gradually trim a dense perturbation to the specified sparsity: {self.sparsity}")
        print(f"The trimmed perturbation will be according to the kernel's structure: {self.kernel_size}")
        print(f"The perturbation will be trimmed to {self.kernel_sparsity} kernel patches of size: {self.kernel_size}X{self.kernel_size}")

        print("Perturbations will be computed for:")
        print(f"L0 norms: {self.l0_norms}")
        print(f"L0 norms kernel number: {self.l0_norms_kernels}")
        print("The best performing perturbations will be reported for:")
        print(f"L0 norms: {self.output_l0_norms}")
        print(f"L0 norms kernel number: {self.output_l0_norms_kernels}")
        print("perturbations L_inf norm limitation:")
        print(self.eps_ratio)
        print("Number of iterations for optimizing perturbations in each trim step:")
        print(self.n_iter)
        print("perturbations will be optimized with the dropout distribution:")
        print(self.dropout_str)
        print("L0 trim steps schedule for the attack:")
        
        self.report_trim_schematics()
        
        print("L0 pixel trimming will be based on masks sampled from the distribution:")
        print(self.mask_dist_str)

    # PGDTrim utilities override
    def parse_mask_args(self):
        super(PGDTrimKernel, self).parse_mask_args()
        if self.mask_dist_name == 'bernoulli' or self.mask_dist_name == 'cbernoulli':
            self.mask_dist_str = 'kernel ' + self.mask_dist_str
            return
        elif self.mask_dist_name == 'topk':
            if self.norm_mask_amp:
                self.sample_mask_pixels = self.sample_mask_pixels_normalize_amp
                self.mask_dist_str = 'kernel topk with normalized amplitude'
            else:
                self.sample_mask_pixels = self.sample_mask_pixels_unknown_count
                self.mask_dist_str = 'kernel topk'
        else:  # multinomial
            if self.norm_mask_amp:
                self.sample_mask_pixels = self.sample_mask_pixels_normalize_amp
                self.mask_dist_str = 'kernel multinomial with normalized amplitude'
            else:
                self.sample_mask_pixels = self.sample_mask_pixels_unknown_count
                self.mask_dist_str = 'kernel multinomial'
    
    def compute_l0_norms(self):
        max_kernels_log_size = int(np.log2(self.max_kernel_sparsity))
        max_trim_size = 2 ** max_kernels_log_size
        n_kerenl_trim_options = int(np.ceil(np.log2(max_trim_size / self.kernel_sparsity)))
        n_kernel_steps = [max_trim_size >> step for step in range(n_kerenl_trim_options)] + [self.kernel_sparsity]
        kernel_l0_norms = [self.n_kernel_pixels * n_kernel for n_kernel in n_kernel_steps]
        if kernel_l0_norms[0] < self.n_data_pixels:
            all_l0_norms = [self.n_data_pixels] + kernel_l0_norms
        else:
            all_l0_norms = kernel_l0_norms
        n_trim_options = len(all_l0_norms) - 2

        return n_trim_options, all_l0_norms
        
    # kernel structure utilities
    def kernel_dpo(self, pert):
        sample = self.dpo_dist.sample()
        return self.apply_mask_method(sample, pert)

    def apply_mask_kernel(self, mask, pert):
        """Apply kernel mask to perturbation"""
        batch_size, channels, h, w = mask.shape
        
        # If mask is in full resolution, downsample it to kernel grid
        if h == self.orig_h and w == self.orig_w:
            mask = self.kernel_max_pool(mask)
        
        # Verify mask dimensions match kernel grid
        assert mask.shape[2] == self.kernel_h and mask.shape[3] == self.kernel_w, \
            f"Mask dimensions {mask.shape[2]}x{mask.shape[3]} don't match kernel grid {self.kernel_h}x{self.kernel_w}"
        
        # Expand the kernel mask to full resolution
        full_mask = self.expand_kernel_mask(mask)
        
        # Ensure mask and perturbation have same dimensions
        if full_mask.size() != pert.size():
            # Expand mask to match perturbation channels
            full_mask = full_mask.expand(-1, pert.size(1), -1, -1)
        
        # Verify dimensions match before multiplication
        assert full_mask.size() == pert.size(), \
            f"Mask size {full_mask.size()} does not match perturbation size {pert.size()}"
        
        return full_mask * pert

    def expand_kernel_mask(self, mask):
        """Expand a kernel-level mask to full image resolution with proper boundary handling"""
        batch_size, channels, h, w = mask.shape
        k = self.kernel_size
        
        # Create full resolution mask
        full_mask = torch.zeros((batch_size, channels, self.orig_h, self.orig_w), device=mask.device)
        
        # Calculate padding to ensure dimensions are multiples of kernel size
        pad_h = (k - (self.orig_h % k)) % k
        pad_w = (k - (self.orig_w % k)) % k
        
        # Adjust original dimensions to be multiples of kernel size
        padded_h = self.orig_h + pad_h
        padded_w = self.orig_w + pad_w
        
        # Verify that kernel grid dimensions match expected values
        expected_h = padded_h // k
        expected_w = padded_w // k
        assert h == expected_h and w == expected_w, \
            f"Kernel grid dimensions {h}x{w} don't match expected {expected_h}x{expected_w}"
        
        # Track actual number of pixels modified
        n_pixels_modified = torch.zeros(batch_size, dtype=torch.long, device=mask.device)
        
        # Expand each kernel position
        for i in range(h):
            for j in range(w):
                if mask[:, :, i, j].sum() > 0:  # Only process active kernel positions
                    h_start = i * k
                    h_end = min((i + 1) * k, self.orig_h)  # Ensure we don't exceed original dimensions
                    w_start = j * k
                    w_end = min((j + 1) * k, self.orig_w)  # Ensure we don't exceed original dimensions
                    
                    # Calculate actual kernel size for this position
                    actual_k_h = h_end - h_start
                    actual_k_w = w_end - w_start
                    
                    # Fill the kernel region with the mask value
                    full_mask[:, :, h_start:h_end, w_start:w_end] = mask[:, :, i:i+1, j:j+1]
                    
                    # Update pixel count
                    n_pixels_modified += (actual_k_h * actual_k_w)
        
        # Store the actual number of modified pixels
        self.last_modified_pixels = n_pixels_modified
        
        return full_mask

    def get_actual_modified_pixels(self):
        """Return the actual number of pixels modified in the last mask expansion"""
        if hasattr(self, 'last_modified_pixels'):
            return self.last_modified_pixels
        return None

    def kernel_active_pool_min(self, mask):
        return - self.kernel_active_pool(-mask)
    
    def kernel_active_pool_avg(self, mask):
        return self.kernel_active_pool(mask)

    def parse_kernel_args(self):
        # Calculate kernel grid dimensions
        self.kernel_h = self.data_h // self.kernel_size
        self.kernel_w = self.data_w // self.kernel_size
        self.n_data_kernels = self.kernel_h * self.kernel_w
        
        # Store original image dimensions for mask expansion
        self.orig_h = self.data_h
        self.orig_w = self.data_w
        
        # Adjust mask shape for kernel grid - use kernel dimensions for mask_shape
        self.mask_shape = [self.batch_size, 1, self.kernel_h, self.kernel_w]  # Kernel grid size
        self.mask_shape_flat = [self.batch_size, self.n_data_kernels]
        
        # Initialize mask tensors with correct kernel grid dimensions
        self.mask_zeros = torch.zeros(self.mask_shape, dtype=torch.bool, device=self.device, requires_grad=False)
        self.mask_zeros_flat = self.mask_zeros.view(self.mask_shape_flat)
        self.mask_ones = torch.ones(self.mask_shape, dtype=torch.bool, device=self.device, requires_grad=False)
        self.mask_ones_flat = self.mask_ones.view(self.mask_shape_flat)
        
        # Set mask application method
        self.apply_mask_method = self.apply_mask_kernel
        
        if self.apply_dpo:
            self.active_dpo = self.kernel_dpo
            self.dropout_str = "kernel " + self.dropout_str

        # No padding needed for non-overlapping kernels
        self.kernel_pad = torch.nn.Identity()
        # Use stride equal to kernel size for non-overlapping kernels
        self.kernel_max_pool = torch.nn.MaxPool2d(self.kernel_size, stride=self.kernel_size)
        
        if self.kernel_min_active:
            self.kernel_active_pool = torch.nn.MaxPool2d(self.kernel_size, stride=self.kernel_size)
            self.kernel_active_pool_method = self.kernel_active_pool_min
        else:
            self.kernel_active_pool = torch.nn.AvgPool2d(self.kernel_size, stride=self.kernel_size)
            self.kernel_active_pool_method = self.kernel_active_pool_avg

        if self.kernel_group:
            self.prob_pool = torch.nn.AvgPool2d(self.kernel_size, stride=self.kernel_size)
            self.prob_pool_method = self.prob_pool_nearest_neighbors
            
            self.mask_dist = self.mask_dist_continuous_bernoulli
            self.sample_mask_pixels = self.sample_mask_pixels_dense

    # grouping kernel mask utilities
    
    def mask_prep_const(self, pixels_prob, n_trim_pixels_tensor):
        return pixels_prob

    def mask_sample_topk(self, pixels_prob, n_active_pixels):
        return self.mask_from_ind(pixels_prob.view(self.batch_size, -1).topk(n_active_pixels, dim=1, sorted=False)[1])

    def mask_sample_bernoulli(self, pixels_prob, index):
        return torch.bernoulli(pixels_prob)

    # grouping kernel mask computation

    def prob_pool_nearest_neighbors(self, pixels_prob, n_trim_pixels_tensor):
        new_prob = pixels_prob + self.prob_pool(pixels_prob)
        new_prob = new_prob * n_trim_pixels_tensor / new_prob.sum(dim=0)
        return new_prob

    def sample_mask_pixels_dense(self, mask_sample_method, mask_prep, n_trim_pixels_tensor, sample_idx):
        pixel_probs_sample = self.mask_sample_from_dist(mask_prep, sample_idx)
        dense_pixel_probs_sample = self.prob_pool_method(pixel_probs_sample, n_trim_pixels_tensor)
        return self.sample_mask_pixels_from_dense(mask_sample_method, dense_pixel_probs_sample, n_trim_pixels_tensor, sample_idx)
    
    def mask_active_pixels(self, mask):
        return self.kernel_active_pool_method(self.kernel_max_pool(self.kernel_pad(mask)))

    # mask trimming override
    
    def mask_trim_best_pixels_crit(self, x, y, dense_pert,
                                   best_pixels_crit, best_pixels_mask, best_pixels_loss,
                                   mask_sample_method, mask_prep_data, n_trim_pixels_tensor, n_mask_samples):
        if n_trim_pixels_tensor < 2:
            return super(PGDTrimKernel, self
                         ).mask_trim_best_pixels_crit(x, y, dense_pert,
                                                      best_pixels_crit, best_pixels_mask, best_pixels_loss,
                                                      mask_sample_method, mask_prep_data,
                                                      n_trim_pixels_tensor, n_mask_samples)
        with torch.no_grad():
            pixels_crit = self.compute_pixels_crit(x, y, dense_pert, best_pixels_crit,
                                                   mask_sample_method, mask_prep_data,
                                                   n_trim_pixels_tensor, n_mask_samples)
            sorted_crit_indices = pixels_crit.view(self.batch_size, -1).argsort(dim=1, descending=True)
            sorted_indices_is_distinct = torch.zeros_like(sorted_crit_indices, dtype=torch.bool)
            sorted_indices_is_distinct[:, 0] = True
            all_pixels = torch.ones_like(dense_pert)
            best_indices = sorted_crit_indices[:, 0].unsqueeze(1)
            best_indices_mask = self.mask_from_ind(best_indices)
            selected_pixels = self.apply_mask_kernel(best_indices_mask, all_pixels)
            distinct_count = torch.ones(self.batch_size, dtype=torch.int, device=self.device)
            for idx in range(1, self.n_data_kernels):
                sort_indices = sorted_crit_indices[:, idx].unsqueeze(1)
                sort_idx_mask = self.mask_from_ind(sort_indices)
                sort_idx_non_zero_count = self.apply_mask_kernel(sort_idx_mask, selected_pixels
                                                            ).view(self.batch_size, -1).count_nonzero(dim=1)
                
                is_distinct = (sort_idx_non_zero_count == 0)
                include_distinct = is_distinct * (distinct_count < n_trim_pixels_tensor)
                sorted_indices_is_distinct[include_distinct, idx] = True
                distinct_count[include_distinct] += 1
                
                idx_selected_pixels = self.apply_mask_kernel(sort_idx_mask, all_pixels)
                selected_pixels[is_distinct] += idx_selected_pixels[is_distinct]
                if (distinct_count == n_trim_pixels_tensor).all():
                    break
            if (distinct_count < n_trim_pixels_tensor).any():
                non_distinct_count = (n_trim_pixels_tensor - distinct_count).tolist()
                non_distinct_batch_count = self.n_data_kernels - distinct_count

                non_distinct_data_ind = (1 - sorted_indices_is_distinct.to(torch.int)).nonzero()
                non_distinct_batch_start_ind = ([0] + non_distinct_batch_count.cumsum(dim=0).tolist())[:-1]

                add_non_distinct_ind = [non_distinct_data_ind[batch_start_ind:batch_start_ind+batch_distinct_count]
                                    for batch_start_ind, batch_distinct_count
                                    in zip(non_distinct_batch_start_ind, non_distinct_count)]
                add_non_distinct_ind = torch.cat(add_non_distinct_ind, dim=0).transpose(0, 1)
                values = torch.ones(add_non_distinct_ind.shape[1], dtype=torch.bool, device=self.device)
                add_non_distinct_coo = torch.sparse_coo_tensor(add_non_distinct_ind, values, self.mask_shape_flat)
                sorted_indices_is_distinct += add_non_distinct_coo
            
            best_crit_mask_indices = sorted_crit_indices[sorted_indices_is_distinct].view(self.batch_size, -1)
            best_crit_mask = self.mask_from_ind(best_crit_mask_indices)
        return best_crit_mask

    def trim_pert_pixels(self, x, y, mask, dense_pert, sparse_pert,
                         best_pixels_crit, best_pixels_mask, best_pixels_loss,
                         mask_prep, mask_sample_method, mask_trim,
                         n_trim_pixels, trim_ratio, n_mask_samples):
        with torch.no_grad():
            if self.verbose:
                print(f"\nTrimming to {n_trim_pixels} pixels ({n_trim_pixels // self.n_kernel_pixels} kernels)")
                if mask is not None:
                    curr_active = (mask.sum(dim=1) > 0).sum(dim=(1,2))
                    print(f"Current mask active pixels: {curr_active.tolist()}")

            # Calculate importance scores based on kernel-wise magnitude
            # First reshape to [batch, channels, h/k, k, w/k, k]
            b, c, h, w = sparse_pert.shape
            k = self.kernel_size
            pert_reshaped = sparse_pert.view(b, c, h//k, k, w//k, k)
            
            # Sum over channels and within kernels
            kernel_magnitude = pert_reshaped.abs().sum(dim=[1, 3, 5])  # Shape: [batch, h//k, w//k]
            
            # Select top-k kernels
            n_kernels = n_trim_pixels // (self.kernel_size * self.kernel_size)
            
            # Create kernel-level mask
            kernel_mask = torch.zeros((b, 1, h//k, w//k), device=sparse_pert.device)
            
            # Get top n_kernels indices for each batch
            _, top_indices = kernel_magnitude.reshape(b, -1).topk(
                min(n_kernels, (h//k) * (w//k)), dim=1)
            
            # Convert linear indices to 2D positions
            h_indices = top_indices // (w//k)
            w_indices = top_indices % (w//k)
            
            # Set selected kernel positions to 1
            for batch_idx in range(b):
                kernel_mask[batch_idx, 0, h_indices[batch_idx], w_indices[batch_idx]] = 1.0
            
            if self.verbose:
                actual_kernels = kernel_mask.sum(dim=(1,2,3))
                print(f"Active kernels before expansion: {actual_kernels.tolist()}")
            
            # Expand kernel mask to full resolution using the existing expand_kernel_mask method
            full_mask = self.expand_kernel_mask(kernel_mask)
            
            # Expand to all channels if needed
            if full_mask.size(1) != sparse_pert.size(1):
                full_mask = full_mask.expand(-1, sparse_pert.size(1), -1, -1)
            
            if self.verbose:
                actual_pixels = (full_mask.sum(dim=1) > 0).sum(dim=(1,2))
                print(f"Active kernels: {actual_kernels.tolist()}")
                print(f"Active pixels: {actual_pixels.tolist()}")
                print(f"Expected kernels: {n_kernels}")
                print(f"Expected pixels: {n_trim_pixels}")
            
            return full_mask

    def pgd(self, x, y, mask, dense_pert, best_sparse_pert, best_loss, best_succ, dpo_mean, dpo_std, n_iter=None):
        """
        Performs PGD attack with kernel-based trimming
        """
        batch_size = x.shape[0]
        device = x.device
        n_iter = n_iter if n_iter is not None else self.n_iter

        # Initialize perturbation
        delta = torch.zeros_like(x, requires_grad=True)
        if self.rand_init:
            delta.data = torch.rand_like(delta.data) * 2 * self.eps_ratio - self.eps_ratio
            if mask is not None:
                # Ensure mask is in kernel grid dimensions before applying
                if mask.shape[2] == self.orig_h and mask.shape[3] == self.orig_w:
                    mask = self.kernel_max_pool(mask)  # Downsample to kernel grid
                delta.data = self.apply_mask_kernel(mask, delta.data)
        
        # Initialize best results
        best_sparse_pert = best_sparse_pert.clone()
        best_loss = best_loss.clone()
        success = best_succ.clone().to(dtype=torch.bool)

        # Print initial attack info
        if self.verbose:
            print(f"\nStarting PGD attack with {n_iter} iterations")
            print(f"Initial perturbation shape: {delta.shape}")
            if mask is not None:
                # Count active kernels in mask
                if mask.shape[2] == self.orig_h and mask.shape[3] == self.orig_w:
                    kernel_mask = self.kernel_max_pool(mask)
                else:
                    kernel_mask = mask
                active_kernels = (kernel_mask.sum(dim=1) > 0).sum(dim=(1,2))
                print(f"Number of active kernels in mask: {active_kernels.tolist()}")

        for i in range(n_iter):
            if self.verbose:
                print(f"\nIteration {i+1}/{n_iter}")
            
            # Forward pass
            with torch.enable_grad():
                x_adv = x + delta
                pred_coords, _ = self.model.predict_from_tensor(x_adv)
                
                # Calculate loss (negative distance for maximization)
                distances = haversine_distance(pred_coords, y)
                loss = -distances.mean()
                
                # Backward pass
                loss.backward()
            
            with torch.no_grad():
                grad = delta.grad.detach()
                
                # Update perturbation
                delta.data = delta.data - self.alpha * grad.sign()
                if mask is not None:
                    # Ensure mask is in kernel grid dimensions before applying
                    if mask.shape[2] == self.orig_h and mask.shape[3] == self.orig_w:
                        curr_mask = self.kernel_max_pool(mask)  # Downsample to kernel grid
                    else:
                        curr_mask = mask
                    delta.data = self.apply_mask_kernel(curr_mask, delta.data)
                delta.data = torch.clamp(delta.data, -self.eps_ratio, self.eps_ratio)
                
                # Calculate current metrics
                curr_success = (distances > self.continent_threshold).bool()
                
                # Update best results if current is better
                better_loss = distances > best_loss
                best_loss[better_loss] = distances[better_loss]
                success[better_loss] = curr_success[better_loss]
                if better_loss.any():
                    best_sparse_pert[better_loss] = delta.detach()[better_loss]
                
                if self.verbose:
                    # Calculate L0 norm based on actual perturbed pixels, not expanded kernel values
                    # First get the mask in kernel grid dimensions
                    if mask.shape[2] == self.orig_h and mask.shape[3] == self.orig_w:
                        curr_mask = self.kernel_max_pool(mask)
                    else:
                        curr_mask = mask
                    # Count active kernels and multiply by kernel size
                    active_kernels = (curr_mask.sum(dim=1) > 0).sum(dim=(1,2))
                    pert_l0 = active_kernels * (self.kernel_size * self.kernel_size)  # Each kernel affects kernel_size^2 pixels
                    
                    print(f"Current loss: {-loss.item():.4f}")
                    print(f"Best loss so far: {best_loss.mean().item():.4f}")
                    print(f"Success rate: {curr_success.float().mean().item()*100:.2f}%")
                    print(f"Mean distance: {distances.mean().item():.4f} km")
                    print(f"L0 norms: {pert_l0.tolist()}")
                    print(f"Gradient stats - Mean: {grad.abs().mean().item():.4f}, Max: {grad.abs().max().item():.4f}")
                
                # Clear gradients
                delta.grad.zero_()
                del grad, pred_coords

        return best_sparse_pert, best_loss, success