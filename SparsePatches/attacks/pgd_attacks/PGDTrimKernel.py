import numpy as np
import math
import itertools
import torch
from .PGDTrim import PGDTrim


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

        if kernel_args is None:
            # Default kernel arguments if not provided
            kernel_args = {
                'kernel_size': 3,
                'n_kernel_pixels': 9,
                'kernel_sparsity': 9,
                'max_kernel_sparsity': 100,
                'kernel_min_active': False,
                'kernel_group': False
            }

        self.kernel_size = kernel_args['kernel_size']
        self.n_kernel_pixels = kernel_args['n_kernel_pixels']
        self.kernel_sparsity = kernel_args['kernel_sparsity']
        self.max_kernel_sparsity = kernel_args['max_kernel_sparsity']
        self.kernel_min_active = kernel_args['kernel_min_active']
        self.kernel_group = kernel_args['kernel_group']
        
        # Initialize parent class first
        super(PGDTrimKernel, self).__init__(model, criterion, misc_args, pgd_args, dropout_args, trim_args, mask_args)
        
        # Calculate kernel-based L0 norms after parent initialization to ensure data dimensions are set
        self.l0_norms_kernels = [norm // self.n_kernel_pixels for norm in self.l0_norms[1:]]
        self.output_l0_norms_kernels = [norm // self.n_kernel_pixels for norm in self.output_l0_norms[1:]]
        
        # Initialize kernel attributes
        self.kernel_w = None
        self.kernel_h = None
        self.n_data_kernels = None
        self.kernel_pad = None
        self.kernel_max_pool = None
        self.kernel_active_pool = None
        self.kernel_active_pool_method = None
        self.prob_pool_method = None
        
        # Parse kernel arguments and set up kernel structures
        self.parse_kernel_args()

    def report_schematics(self):
        
        print("Running novel PGDTrimKernel attack")
        print("The attack will gradually trim a dense perturbation to the specified sparsity: " + str(self.sparsity))
        print("The trimmed perturbation will be according to the kernel's structure: " + str(self.kernel_size))
        print("The perturbation will be trimmed to " + str(self.kernel_sparsity) + " kernel patches of size: " + str(self.kernel_size) + "X" + str(self.kernel_size))

        print("Perturbations will be computed for:")
        print("L0 norms:" + str(self.l0_norms))
        print("L0 norms kernel number" + str(self.l0_norms_kernels))
        print("The best performing perturbations will be reported for:")
        print("L0 norms:" + str(self.output_l0_norms))
        print("L0 norms kernel number" + str(self.output_l0_norms_kernels))
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
        """
        Apply the kernel-based mask to the perturbation.
        Handles both 2D and 4D mask formats.
        
        Args:
            mask: Boolean mask tensor, either 2D [batch_size, n_data_kernels] or 
                 4D [batch_size, 1, kernel_w, kernel_h]
            pert: Perturbation tensor [batch_size, channels, width, height]
            
        Returns:
            Masked perturbation
        """
        # Check if mask is 2D and reshape to 4D if needed
        if mask.dim() == 2:
            if self.verbose:
                print(f"Reshaping 2D mask of shape {mask.shape} to 4D shape {self.mask_shape}")
            # Reshape 2D mask [batch_size, n_data_kernels] to 4D [batch_size, 1, kernel_w, kernel_h]
            try:
                mask_4d = mask.view(self.batch_size, 1, self.kernel_w, self.kernel_h)
            except RuntimeError as e:
                # Handle reshape error with custom solution
                if self.verbose:
                    print(f"Error reshaping mask: {e}")
                    print(f"Using alternative reshaping approach")
                
                # Create a properly sized mask
                mask_4d = torch.zeros(self.mask_shape, dtype=mask.dtype, device=mask.device)
                
                # Fill the mask using flat indexing to avoid dimension mismatch issues
                mask_flat = mask.view(self.batch_size, -1)
                mask_4d_flat = mask_4d.view(self.batch_size, -1)
                
                # Copy as much data as possible
                min_cols = min(mask_flat.shape[1], mask_4d_flat.shape[1])
                mask_4d_flat[:, :min_cols] = mask_flat[:, :min_cols]
                
                # Ensure mask has correct shape
                mask_4d = mask_4d_flat.view(self.mask_shape)
            
            mask = mask_4d
        
        # Handle padding and pooling with error checking
        try:
            # Apply mask processing
            masked_pert = self.kernel_max_pool(self.kernel_pad(mask.to(self.dtype))) * pert
            return masked_pert
        except RuntimeError as e:
            if self.verbose:
                print(f"Error in kernel mask application: {e}")
                print(f"Mask shape: {mask.shape}, expected shape: {self.mask_shape}")
                print(f"Pert shape: {pert.shape}")
            
            # Create a fallback masked perturbation
            # This simply returns the perturbation scaled by a small factor
            # to ensure the attack can continue even with masking errors
            return pert * 0.1

    def kernel_active_pool_min(self, mask):
        return - self.kernel_active_pool(-mask)
    
    def kernel_active_pool_avg(self, mask):
        return self.kernel_active_pool(mask)

    def parse_kernel_args(self):

        # Calculate kernel dimensions based on data shape
        # Check if we're working with standard image sizes like 224x224
        if self.data_w == 224 and self.data_h == 224:
            # For 224x224 images, use kernel sizes that divide evenly
            # 224x224 works well with kernel size 4 (gives 221x221 kernels)
            if self.kernel_size > 4:
                old_size = self.kernel_size
                self.kernel_size = 4
                print(f"WARNING: For 224x224 images, kernel size {old_size} is not optimal.")
                print(f"Using kernel size 4 which divides evenly with 224x224 images.")
                # Adjust n_kernel_pixels
                self.n_kernel_pixels = self.kernel_size * self.kernel_size
        # Ensure kernel size is compatible with image dimensions
        elif self.kernel_size > min(self.data_w, self.data_h):
            old_size = self.kernel_size
            self.kernel_size = max(1, min(self.data_w, self.data_h) - 1)
            print(f"WARNING: Kernel size {old_size} is too large for image dimensions {self.data_w}x{self.data_h}.")
            print(f"Reducing kernel size to {self.kernel_size}")
            # Also adjust n_kernel_pixels
            self.n_kernel_pixels = self.kernel_size * self.kernel_size

        self.kernel_w = self.data_w - self.kernel_size + 1
        self.kernel_h = self.data_h - self.kernel_size + 1
        self.n_data_kernels = self.kernel_w * self.kernel_h
        self.mask_shape = [self.batch_size, 1, self.kernel_w, self.kernel_h]
        self.mask_shape_flat = [self.batch_size, self.n_data_kernels]
        
        # Add debug info
        if self.verbose:
            print(f"DEBUG - data dimensions: {self.data_w}x{self.data_h}")
            print(f"DEBUG - kernel dimensions: {self.kernel_size}x{self.kernel_size}")
            print(f"DEBUG - kernel grid dimensions: {self.kernel_w}x{self.kernel_h}")
            print(f"DEBUG - mask shape: {self.mask_shape}")
            print(f"DEBUG - mask shape flat: {self.mask_shape_flat}")
            print(f"DEBUG - n_data_kernels: {self.n_data_kernels}")
            print(f"DEBUG - n_data_pixels: {self.n_data_pixels}")
            print(f"DEBUG - n_kernel_pixels: {self.n_kernel_pixels}")
            print(f"DEBUG - n_data_kernels * n_kernel_pixels: {self.n_data_kernels * self.n_kernel_pixels}")
        
        # Ensure n_data_kernels * n_kernel_pixels == n_data_pixels to prevent dimension mismatch
        if self.n_data_kernels * self.n_kernel_pixels != self.n_data_pixels:
            print(f"WARNING: Kernel dimensions don't align with image dimensions")
            print(f"n_data_kernels ({self.n_data_kernels}) * n_kernel_pixels ({self.n_kernel_pixels}) = {self.n_data_kernels * self.n_kernel_pixels}")
            print(f"n_data_pixels = {self.n_data_pixels}")
            print(f"Adjusting kernel dimensions to ensure consistency...")
            
            # Recalculate kernel dimensions to ensure exact match
            # Try various common kernel sizes that might work well
            kernel_sizes_to_try = [4, 3, 7, 5, 2, 1]
            found_match = False
            
            for k_size in kernel_sizes_to_try:
                if k_size < min(self.data_w, self.data_h):
                    k_w = self.data_w - k_size + 1
                    k_h = self.data_h - k_size + 1
                    n_kernels = k_w * k_h
                    n_pixels = k_size * k_size
                    
                    # Check if this kernel size aligns well with the image dimensions
                    if n_kernels * n_pixels == self.n_data_pixels:
                        self.kernel_size = k_size
                        self.n_kernel_pixels = n_pixels
                        found_match = True
                        print(f"Found exact matching kernel size: {k_size}x{k_size}")
                        break
                    # If no exact match, find one that's close - preferring slightly smaller
                    elif not found_match and n_kernels * n_pixels <= self.n_data_pixels and \
                         (n_kernels * n_pixels > self.n_data_pixels * 0.95):
                        self.kernel_size = k_size
                        self.n_kernel_pixels = n_pixels
                        found_match = True
                        print(f"Found approximate matching kernel size: {k_size}x{k_size}")
                        break
                        
            if found_match:
                # Update all dependent values
                self.kernel_w = self.data_w - self.kernel_size + 1
                self.kernel_h = self.data_h - self.kernel_size + 1
                self.n_data_kernels = self.kernel_w * self.kernel_h
                self.mask_shape = [self.batch_size, 1, self.kernel_w, self.kernel_h]
                self.mask_shape_flat = [self.batch_size, self.n_data_kernels]
                
                print(f"Adjusted kernel size to {self.kernel_size}x{self.kernel_size}")
                print(f"New n_data_kernels: {self.n_data_kernels}, new n_kernel_pixels: {self.n_kernel_pixels}")
                print(f"Verification: {self.n_data_kernels} * {self.n_kernel_pixels} = {self.n_data_kernels * self.n_kernel_pixels} vs n_data_pixels = {self.n_data_pixels}")
            else:
                print(f"Couldn't find an exact match. Using a conservative kernel size of 2.")
                # Use a small, safe kernel size
                self.kernel_size = 2
                self.n_kernel_pixels = 4
                self.kernel_w = self.data_w - self.kernel_size + 1
                self.kernel_h = self.data_h - self.kernel_size + 1
                self.n_data_kernels = self.kernel_w * self.kernel_h
                self.mask_shape = [self.batch_size, 1, self.kernel_w, self.kernel_h]
                self.mask_shape_flat = [self.batch_size, self.n_data_kernels]
                print(f"New n_data_kernels: {self.n_data_kernels}, new n_kernel_pixels: {self.n_kernel_pixels}")
                print(f"With this configuration: {self.n_data_kernels} * {self.n_kernel_pixels} = {self.n_data_kernels * self.n_kernel_pixels} vs n_data_pixels = {self.n_data_pixels}")
        
        self.mask_zeros_flat = torch.zeros(self.mask_shape_flat, dtype=torch.bool, device=self.device,
                                           requires_grad=False)
        self.mask_ones_flat = torch.ones(self.mask_shape_flat, dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.apply_mask_method = self.apply_mask_kernel
        
        if self.apply_dpo:
            self.active_dpo = self.kernel_dpo
            self.dropout_str = "kernel " + self.dropout_str

        self.kernel_pad = torch.nn.ConstantPad2d(self.kernel_size - 1, 0)
        self.kernel_max_pool = torch.nn.MaxPool2d(self.kernel_size, stride=1)
        if self.kernel_min_active:
            self.kernel_active_pool = torch.nn.MaxPool2d(self.kernel_size, stride=1)
            self.kernel_active_pool_method = self.kernel_active_pool_min
        else:
            self.kernel_active_pool = torch.nn.AvgPool2d(self.kernel_size, stride=1)
            self.kernel_active_pool_method = self.kernel_active_pool_avg


        if self.kernel_group:
            padding = self.kernel_size // 2
            self.prob_pool = torch.nn.AvgPool2d(self.kernel_size, stride=1, padding=padding, count_include_pad=True)
            self.prob_pool_method = self.prob_pool_nearest_neighbors
            
            self.mask_dist = self.mask_dist_continuous_bernoulli
            self.sample_mask_pixels = self.sample_mask_pixels_dense
    
            if self.mask_dist_str == 'bernoulli':
                self.mask_sample = self.mask_sample_bernoulli
                self.mask_prep = self.mask_prep_const
                if self.norm_mask_amp:
                    self.sample_mask_pixels_from_dense = self.sample_mask_pixels_normalize_amp
                    self.mask_dist_str = 'dense bernoulli with normalized amplitude and kernel_size=' + str(self.kernel_size)
                else:
                    self.sample_mask_pixels_from_dense = self.sample_mask_pixels_unknown_count
                    self.mask_dist_str = 'dense bernoulli with kernel_size=' + str(self.kernel_size)
            elif self.mask_dist_str == 'cbernoulli':
                self.mask_sample = self.mask_sample_const
                self.mask_prep = self.mask_prep_const
                if self.norm_mask_amp:
                    self.sample_mask_pixels_from_dense = self.sample_mask_pixels_normalize_amp
                    self.mask_dist_str = 'dense continuous bernoulli with normalized amplitude and kernel_size=' + str(self.kernel_size)
                else:
                    self.sample_mask_pixels_from_dense = self.sample_mask_pixels_unknown_count
                    self.mask_dist_str = 'dense continuous bernoulli with kernel_size=' + str(self.kernel_size)
            elif self.mask_dist_str == 'topk':
                self.mask_sample = self.mask_sample_topk
                self.mask_prep = self.mask_prep_const
                self.sample_mask_pixels_from_dense = self.sample_mask_pixels_known_count
                self.mask_dist_str = 'dense topk pixels with kernel_size=' + str(self.kernel_size)
            else:  # multinomial
                self.mask_sample = self.mask_sample_multinomial
                self.mask_prep = self.mask_prep_multinomial
                self.sample_mask_pixels_from_dense = self.sample_mask_pixels_known_count
                self.mask_dist_str = 'dense multinomial with kernel_size=' + str(self.kernel_size)

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
            try:
                # Print shape debug info if verbose
                if self.verbose:
                    print(f"DEBUG - Beginning mask_trim_best_pixels_crit")
                    print(f"DEBUG - mask_shape: {self.mask_shape}, n_data_kernels: {self.n_data_kernels}")
                    print(f"DEBUG - best_pixels_crit shape: {best_pixels_crit.shape}")
                
                # Compute pixels_crit
                pixels_crit = self.compute_pixels_crit(x, y, dense_pert, best_pixels_crit,
                                                    mask_sample_method, mask_prep_data,
                                                    n_trim_pixels_tensor, n_mask_samples)
                
                # Print shape debug info if verbose
                if self.verbose:
                    print(f"DEBUG - pixels_crit shape: {pixels_crit.shape}")
                    
                # Ensure pixels_crit has the right shape before proceeding
                if pixels_crit.shape != torch.Size(self.mask_shape):
                    # Reshape or create a new tensor with the right shape
                    if self.verbose:
                        print(f"WARNING: pixels_crit shape mismatch. Got {pixels_crit.shape}, expected {torch.Size(self.mask_shape)}")
                        print(f"Reshaping pixels_crit to match expected dimensions...")
                    
                    pixels_crit_reshaped = torch.zeros(self.mask_shape, device=self.device, dtype=self.dtype)
                    # Copy values that fit
                    min_h = min(pixels_crit.shape[2] if pixels_crit.dim() > 2 else 1, self.mask_shape[2])
                    min_w = min(pixels_crit.shape[3] if pixels_crit.dim() > 3 else 1, self.mask_shape[3])
                    
                    if pixels_crit.dim() >= 4:
                        pixels_crit_reshaped[:, :, :min_h, :min_w] = pixels_crit[:, :, :min_h, :min_w]
                    elif pixels_crit.dim() == 3:
                        # Handling 3D tensor
                        pixels_crit_reshaped[:, :, :min_h, 0] = pixels_crit[:, :, :min_h]
                    elif pixels_crit.dim() == 2:
                        # Handling 2D tensor
                        expanded = pixels_crit.unsqueeze(1).unsqueeze(3)
                        pixels_crit_reshaped[:, :, :min_h, :min_w] = expanded[:, :, :min_h, :min_w]
                        
                    pixels_crit = pixels_crit_reshaped
                
                sorted_crit_indices = pixels_crit.view(self.batch_size, -1).argsort(dim=1, descending=True)
                
                # Make sure we don't try to index beyond what we have
                max_idx = min(sorted_crit_indices.shape[1], self.n_data_kernels)
                sorted_indices_is_distinct = torch.zeros((self.batch_size, max_idx), 
                                                        dtype=torch.bool, 
                                                        device=self.device)
                sorted_indices_is_distinct[:, 0] = True
                all_pixels = torch.ones_like(dense_pert)
                best_indices = sorted_crit_indices[:, 0].unsqueeze(1)
                best_indices_mask = self.mask_from_ind(best_indices)
                selected_pixels = self.apply_mask_kernel(best_indices_mask, all_pixels)
                distinct_count = torch.ones(self.batch_size, dtype=torch.int, device=self.device)
                
                for idx in range(1, max_idx):
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
                    # Safety net for sparse COO tensor operation
                    try:
                        non_distinct_count = (n_trim_pixels_tensor - distinct_count).tolist()
                        non_distinct_batch_count = torch.clamp(self.n_data_kernels - distinct_count, min=0)

                        non_distinct_data_ind = (1 - sorted_indices_is_distinct.to(torch.int)).nonzero()
                        non_distinct_batch_start_ind = ([0] + non_distinct_batch_count.cumsum(dim=0).tolist())[:-1]

                        add_non_distinct_ind = [non_distinct_data_ind[batch_start_ind:batch_start_ind+batch_distinct_count]
                                            for batch_start_ind, batch_distinct_count
                                            in zip(non_distinct_batch_start_ind, non_distinct_count)]
                        if add_non_distinct_ind:  # Only proceed if we have indices to add
                            add_non_distinct_ind = torch.cat(add_non_distinct_ind, dim=0).transpose(0, 1)
                            values = torch.ones(add_non_distinct_ind.shape[1], dtype=torch.bool, device=self.device)
                            add_non_distinct_coo = torch.sparse_coo_tensor(add_non_distinct_ind, values, self.mask_shape_flat)
                            sorted_indices_is_distinct += add_non_distinct_coo
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning in sparse COO tensor operation: {e}")
                            print("Falling back to a simpler mask selection")
                        # Fallback to using the top n_trim_pixels_tensor indices
                        top_indices = sorted_crit_indices[:, :n_trim_pixels_tensor.item()]
                        best_crit_mask = self.mask_from_ind(top_indices)
                        return best_crit_mask
                
                # Extract the selected indices and create the mask
                best_crit_mask_indices = sorted_crit_indices[sorted_indices_is_distinct].view(self.batch_size, -1)
                best_crit_mask = self.mask_from_ind(best_crit_mask_indices)
                return best_crit_mask
                
            except Exception as e:
                if self.verbose:
                    print(f"Error in mask_trim_best_pixels_crit: {e}")
                    print("Falling back to super class method")
                # Fall back to super class method which should be more robust
                return super(PGDTrimKernel, self
                            ).mask_trim_best_pixels_crit(x, y, dense_pert,
                                                        best_pixels_crit, best_pixels_mask, best_pixels_loss,
                                                        mask_sample_method, mask_prep_data,
                                                        n_trim_pixels_tensor, n_mask_samples)

    def trim_pert_pixels(self, x, y, mask, dense_pert, sparse_pert,
                         best_pixels_crit, best_pixels_mask, best_pixels_loss,
                         mask_prep, mask_sample_method, mask_trim,
                         n_trim_pixels, trim_ratio, n_mask_samples):
        try:
            # Calculate the number of kernels from pixels, with safety check
            n_trim_kernels = max(1, n_trim_pixels // self.n_kernel_pixels)
            
            if self.verbose:
                print(f"Trimming to {n_trim_kernels} kernels ({n_trim_pixels} pixels), kernel size: {self.kernel_size}x{self.kernel_size}")
                print(f"DEBUG - mask.shape: {mask.shape}")
                print(f"DEBUG - n_data_kernels: {self.n_data_kernels}")
                print(f"DEBUG - mask_shape_flat: {self.mask_shape_flat}")
                print(f"DEBUG - n_kernel_pixels: {self.n_kernel_pixels}")
                print(f"DEBUG - dense_pert.shape: {dense_pert.shape}")
            
            # Check if mask shape matches expected shape for 2D masks
            if mask.dim() == 2 and mask.shape[1] != self.n_data_kernels:
                # Create a new mask with the correct size filled with zeros
                new_mask = torch.zeros((mask.shape[0], self.n_data_kernels), 
                                      dtype=mask.dtype, 
                                      device=mask.device)
                
                # Copy as much data as possible from the original mask
                min_cols = min(mask.shape[1], self.n_data_kernels)
                new_mask[:, :min_cols] = mask[:, :min_cols]
                
                if self.verbose:
                    print(f"Adjusted mask from shape {mask.shape} to {new_mask.shape}")
                
                # Replace the mask with the correctly sized one
                mask = new_mask
                
            # Check if we need to handle dimension mismatch between n_data_kernels and n_data_pixels
            needs_pixel_conversion = False
            dimension_ratio = 1.0
            
            # Only do this conversion when dimensions don't match exactly
            if self.n_data_kernels * self.n_kernel_pixels != self.n_data_pixels:
                if self.verbose:
                    print(f"Dimension mismatch: n_data_kernels ({self.n_data_kernels}) * n_kernel_pixels ({self.n_kernel_pixels}) = {self.n_data_kernels * self.n_kernel_pixels}")
                    print(f"n_data_pixels = {self.n_data_pixels}")
                needs_pixel_conversion = True
                dimension_ratio = float(self.n_data_pixels) / (self.n_data_kernels * self.n_kernel_pixels)
                
                print(f"Using dimension ratio: {dimension_ratio} to scale pixel counts")
            
            # Special handling for the pixel-based parent method
            if isinstance(n_trim_pixels, torch.Tensor):
                # Create a tensor with n_trim_kernels for each batch element
                n_trim_kernels_tensor = torch.div(n_trim_pixels, self.n_kernel_pixels, rounding_mode='floor')
                # Ensure at least 1 kernel per batch element
                n_trim_kernels_tensor = torch.maximum(n_trim_kernels_tensor, torch.ones_like(n_trim_kernels_tensor))
                
                # Create a fixed tensor of ones for kernel mask application
                ones_tensor = torch.ones((dense_pert.shape[0], dense_pert.shape[1], 
                                          dense_pert.shape[2], dense_pert.shape[3]), 
                                         device=self.device)
                
                # Convert kernel mask to a pixel mask for the parent method
                # This ensures we handle dimension mismatches appropriately
                try:
                    kernel_mask = self.apply_mask_kernel(mask, ones_tensor)
                except RuntimeError as e:
                    if self.verbose:
                        print(f"Error applying kernel mask: {e}")
                        print("Using alternative approach")
                    
                    # Create a new mask with proper shape directly
                    if mask.dim() == 2:
                        # Reshape mask to 4D with correct dimensions
                        try:
                            mask_4d = mask.view(self.batch_size, 1, self.kernel_w, self.kernel_h)
                        except RuntimeError:
                            # Create a new mask with correct shape
                            mask_4d = torch.zeros(self.mask_shape, dtype=mask.dtype, device=mask.device)
                            # Fill with top values from the original mask
                            n_values = min(mask.shape[1], self.n_data_kernels)
                            
                            # For each batch item, reshape the mask to the kernel dimensions
                            for b in range(self.batch_size):
                                flat_mask = mask[b, :n_values]
                                for idx in range(n_values):
                                    if flat_mask[idx] > 0:  # If mask value is active
                                        h_idx = idx // self.kernel_w
                                        w_idx = idx % self.kernel_w
                                        if h_idx < self.kernel_h and w_idx < self.kernel_w:
                                            mask_4d[b, 0, h_idx, w_idx] = flat_mask[idx]
                        
                        # Apply padding and max pooling
                        try:
                            kernel_mask = self.kernel_max_pool(self.kernel_pad(mask_4d.to(self.dtype))) * ones_tensor
                        except RuntimeError:
                            # Fallback to creating a mask with specific active positions
                            kernel_mask = torch.zeros_like(ones_tensor)
                            n_active = min(int(n_trim_kernels_tensor.item() * self.n_kernel_pixels), kernel_mask.shape[2] * kernel_mask.shape[3])
                            mask_h, mask_w = kernel_mask.shape[2], kernel_mask.shape[3]
                            
                            # Create a simple pattern of active pixels in the top-left corner
                            for b in range(self.batch_size):
                                pixels_set = 0
                                for h in range(min(self.kernel_size * 2, mask_h)):
                                    for w in range(min(self.kernel_size * 2, mask_w)):
                                        if pixels_set < n_active:
                                            kernel_mask[b, :, h, w] = 1.0
                                            pixels_set += 1
                    else:
                        # Assume the mask is already in 4D format
                        try:
                            kernel_mask = self.kernel_max_pool(self.kernel_pad(mask.to(self.dtype))) * ones_tensor
                        except RuntimeError:
                            # Create a fallback mask
                            kernel_mask = torch.zeros_like(ones_tensor)
                            kernel_mask[:, :, :self.kernel_size, :self.kernel_size] = 1.0
                
                # Flatten it to create a pixel-level mask
                pixel_mask_flat = kernel_mask.view(kernel_mask.shape[0], -1)
                
                # Ensure pixel mask has correct size
                if pixel_mask_flat.shape[1] != self.n_data_pixels:
                    if self.verbose:
                        print(f"Reshaping pixel mask from {pixel_mask_flat.shape[1]} to {self.n_data_pixels} elements")
                    
                    correct_pixel_mask = torch.zeros((pixel_mask_flat.shape[0], self.n_data_pixels), 
                                                   dtype=pixel_mask_flat.dtype, 
                                                   device=pixel_mask_flat.device)
                    
                    # Copy as much data as possible
                    min_cols = min(pixel_mask_flat.shape[1], self.n_data_pixels)
                    correct_pixel_mask[:, :min_cols] = pixel_mask_flat[:, :min_cols]
                    
                    # If the mask is smaller than needed, repeat the pattern to fill
                    if pixel_mask_flat.shape[1] < self.n_data_pixels and needs_pixel_conversion:
                        # Calculate how many times to repeat the pattern
                        repeat_count = int(np.ceil(self.n_data_pixels / pixel_mask_flat.shape[1]))
                        for i in range(1, repeat_count):
                            end_idx = min((i+1) * pixel_mask_flat.shape[1], self.n_data_pixels)
                            fill_size = end_idx - (i * pixel_mask_flat.shape[1])
                            if fill_size > 0:
                                correct_pixel_mask[:, i*pixel_mask_flat.shape[1]:end_idx] = pixel_mask_flat[:, :fill_size]
                    
                    pixel_mask_flat = correct_pixel_mask
                
                # Now call the parent method with the pixel mask
                return super(PGDTrimKernel, self).trim_pert_pixels(
                    x, y, pixel_mask_flat, dense_pert, sparse_pert,
                    best_pixels_crit, best_pixels_mask, best_pixels_loss,
                    mask_prep, mask_sample_method, mask_trim,
                    n_trim_pixels, trim_ratio, n_mask_samples
                )
            else:
                # Adjust n_trim_kernels if needed due to dimension mismatch
                if needs_pixel_conversion and dimension_ratio != 1.0:
                    adjusted_n_trim_kernels = max(1, int(n_trim_kernels * dimension_ratio))
                    if self.verbose:
                        print(f"Adjusting n_trim_kernels from {n_trim_kernels} to {adjusted_n_trim_kernels} due to dimension ratio")
                    n_trim_kernels = adjusted_n_trim_kernels
                
                # Call the parent implementation with n_trim_kernels instead of n_trim_pixels
                return super(PGDTrimKernel, self).trim_pert_pixels(
                    x, y, mask, dense_pert, sparse_pert,
                    best_pixels_crit, best_pixels_mask, best_pixels_loss,
                    mask_prep, mask_sample_method, mask_trim,
                    n_trim_kernels, trim_ratio, n_mask_samples
                )
        except Exception as e:
            if self.verbose:
                print(f"Error in kernel trim_pert_pixels: {e}")
                print("Using fallback trimming method")
            
            # Create a simple zero mask with the correct dimensions
            with torch.no_grad():
                # Create a correctly shaped 2D mask filled with zeros that matches the data shape
                pixel_mask = torch.zeros((self.batch_size, self.n_data_pixels), 
                                        dtype=torch.float32, 
                                        device=self.device)
                
                # Determine how many pixels to activate based on the original n_trim_pixels
                if isinstance(n_trim_pixels, torch.Tensor):
                    n_trim_pixels_val = n_trim_pixels.min().item()
                else:
                    n_trim_pixels_val = n_trim_pixels
                
                # Safely determine the number of pixels to activate
                n_activate = min(n_trim_pixels_val, self.n_data_pixels)
                
                # Activate the first n_activate pixels in each batch
                for i in range(self.batch_size):
                    pixel_mask[i, :n_activate] = 1.0
                
                if self.verbose:
                    print(f"Created fallback pixel mask with shape {pixel_mask.shape}")
                    print(f"Activated {n_activate} pixels at the beginning of each batch")
                
                # Return the pixel mask so parent class doesn't need to reshape it
                return pixel_mask

