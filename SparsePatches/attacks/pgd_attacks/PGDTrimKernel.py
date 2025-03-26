import numpy as np
import math
import itertools
import torch
from attacks.pgd_attacks.PGDTrim import PGDTrim


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

        # Initialize kernel parameters
        if kernel_args is None:
            kernel_args = {}
            
        # Set kernel defaults
        self.kernel_size = kernel_args.get('kernel_size', 3)
        self.n_kernel_pixels = kernel_args.get('n_kernel_pixels', self.kernel_size * self.kernel_size)
        self.kernel_sparsity = kernel_args.get('kernel_sparsity', self.n_kernel_pixels)
        self.max_kernel_sparsity = kernel_args.get('max_kernel_sparsity', 256)
        self.kernel_min_active = kernel_args.get('kernel_min_active', False)
        self.kernel_group = kernel_args.get('kernel_group', False)
        
        # Make sure necessary parameters exist in misc_args
        if misc_args is None:
            misc_args = {}
        
        # Extract data shape for image dimensions
        data_shape = misc_args.get('data_shape', [3, 224, 224])
        if len(data_shape) >= 2:
            self.data_h = data_shape[-2]
            self.data_w = data_shape[-1]
        else:
            # Default to standard image size
            self.data_h = 224
            self.data_w = 224
            
        # Calculate pixel counts
        self.n_data_pixels = self.data_h * self.data_w
        
        # Extract batch size
        self.batch_size = misc_args.get('batch_size', 1)
        
        # Extract device information
        self.device = misc_args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Print debug info about kernel configuration
        print(f"PGDTrimKernel initialized with:")
        print(f"  kernel_size: {self.kernel_size}")
        print(f"  n_kernel_pixels: {self.n_kernel_pixels}")
        print(f"  kernel_sparsity: {self.kernel_sparsity}")
        print(f"  max_kernel_sparsity: {self.max_kernel_sparsity}")
        print(f"  image dimensions: {self.data_w}x{self.data_h}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  device: {self.device}")
        
        # Make sure trim_args has the sparsity parameter set
        if trim_args is None:
            trim_args = {}
        if 'sparsity' not in trim_args and hasattr(self, 'kernel_sparsity'):
            trim_args['sparsity'] = self.kernel_sparsity
            
        # Call the parent class initializer
        super(PGDTrimKernel, self).__init__(model, criterion, misc_args, pgd_args, dropout_args, trim_args, mask_args)
        
        # Calculate L0 norms for kernels
        self.l0_norms_kernels = [norm // self.n_kernel_pixels for norm in self.l0_norms[1:]]
        self.output_l0_norms_kernels = [norm // self.n_kernel_pixels for norm in self.output_l0_norms[1:]]
        
        # Initialize remaining kernel parameters
        self.kernel_w = None
        self.kernel_h = None
        self.n_data_kernels = None
        self.kernel_pad = None
        self.kernel_max_pool = None
        self.kernel_active_pool = None
        self.kernel_active_pool_method = None
        self.prob_pool_method = None
        
        # Parse kernel args to set up all needed operations
        self.parse_kernel_args()
        
        # Print final configuration
        print(f"Data shape: {self.data_shape}")
        print(f"Kernel dimensions: {self.kernel_w}x{self.kernel_h}")
        print(f"L0 norms for kernels: {self.l0_norms_kernels}")

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
        """
        Compute the L0 norms for the attack, taking into account the kernel structure.
        
        This method calculates the L0 norms for the sparsity steps in the attack,
        ensuring we start with a large enough sparsity and gradually reduce it.
        """
        print(f"Computing L0 norms for kernel attack with:")
        print(f"  kernel_size: {self.kernel_size}, kernel_sparsity: {self.kernel_sparsity}")
        print(f"  max_kernel_sparsity: {self.max_kernel_sparsity}, sparsity: {self.sparsity}")
        
        # Calculate how many kernels we can have given the sparsity constraint
        max_kernels = max(1, self.sparsity // self.kernel_sparsity)
        print(f"  max_kernels: {max_kernels}")
        
        # Calculate the maximum log2 size for kernels
        max_kernels_log_size = int(np.log2(max_kernels)) + 1
        max_trim_size = 2 ** max_kernels_log_size
        print(f"  max_trim_size (kernels): {max_trim_size}")
        
        # Calculate how many trim steps we need
        n_kernel_trim_options = int(np.ceil(np.log2(max_trim_size / max(1, self.kernel_sparsity)))) + 1
        
        # Calculate the number of kernels at each step, decreasing by powers of 2
        n_kernel_steps = [max_trim_size >> step for step in range(n_kernel_trim_options)]
        
        # Add the final target number of kernels
        if n_kernel_steps[-1] != max_kernels:
            n_kernel_steps.append(max_kernels)
            
        # Remove duplicates and sort in descending order
        n_kernel_steps = sorted(list(set(n_kernel_steps)), reverse=True)
        
        # Calculate L0 norms based on number of kernels * kernel sparsity
        kernel_l0_norms = [min(self.n_data_pixels, k * self.kernel_sparsity) for k in n_kernel_steps]
        
        # Start with the full image size
        if kernel_l0_norms[0] < self.n_data_pixels:
            all_l0_norms = [self.n_data_pixels] + kernel_l0_norms
        else:
            all_l0_norms = kernel_l0_norms
            
        # Ensure the final sparsity is included
        if all_l0_norms[-1] != self.sparsity:
            all_l0_norms.append(self.sparsity)
            
        # Remove duplicates and sort in descending order
        all_l0_norms = sorted(list(set(all_l0_norms)), reverse=True)
            
        # Calculate number of trim steps
        n_trim_options = len(all_l0_norms) - 1
        
        print(f"L0 norms: {all_l0_norms}")
        print(f"Number of trim steps: {n_trim_options}")

        return n_trim_options, all_l0_norms
        
    # kernel structure utilities
    def kernel_dpo(self, pert):
        sample = self.dpo_dist.sample()
        return self.apply_mask_method(sample, pert)

    def apply_mask_kernel(self, mask, pert):
        """
        Apply a kernel-based mask to the perturbation.
        
        The mask typically has smaller dimensions (kernel_w x kernel_h) since it represents
        centers of kernels. This function expands it to match the perturbation size by:
        1. Padding the mask
        2. Applying max pooling to get the kernel effect
        3. Ensuring dimensions match
        4. Applying the result to the perturbation
        """
        # Ensure proper dimensions of mask and pert
        if mask.dim() != 4:
            raise ValueError(f"Expected mask with shape (batch_size, 1, kernel_h, kernel_w), got {mask.shape}")
        if pert.dim() != 4:
            raise ValueError(f"Expected pert with shape (batch_size, channels, height, width), got {pert.shape}")
        
        batch_size, channels, height, width = pert.shape
        
        # Print dimensions for debugging
        print(f"Debug: mask shape: {mask.shape}, pert shape: {pert.shape}")
        print(f"Debug: kernel dimensions - w: {self.kernel_w}, h: {self.kernel_h}")
        print(f"Debug: data dimensions - w: {self.data_w}, h: {self.data_h}")
        
        try:
            # Method 1: Use a custom padding approach
            # Create a new tensor of the right size filled with zeros
            expanded_mask = torch.zeros((batch_size, 1, height, width), device=mask.device, dtype=mask.dtype)
            
            # Calculate offsets to center the mask
            h_offset = (height - self.kernel_h) // 2
            w_offset = (width - self.kernel_w) // 2
            
            # Place the mask in the center
            expanded_mask[:, :, 
                         h_offset:h_offset+self.kernel_h, 
                         w_offset:w_offset+self.kernel_w] = mask
                         
            # Apply max pooling with correct padding to ensure dimensions match
            full_padding = self.kernel_size // 2
            mask_pooled = torch.nn.functional.max_pool2d(
                expanded_mask,
                kernel_size=self.kernel_size,
                stride=1,
                padding=full_padding
            )
            
            # Ensure the output has the correct dimensions
            if mask_pooled.shape[2:] != (height, width):
                print(f"Warning: mask_pooled shape {mask_pooled.shape} doesn't match expected {(batch_size, 1, height, width)}")
                mask_pooled = torch.nn.functional.interpolate(
                    mask_pooled,
                    size=(height, width),
                    mode='nearest'
                )
            
            # Apply the pooled mask to the perturbation
            return mask_pooled * pert
            
        except Exception as e:
            print(f"Error in apply_mask_kernel: {e}")
            print(f"Mask shape: {mask.shape}, Pert shape: {pert.shape}")
            
            # Fallback: Just interpolate the mask directly to the target size
            upsampled_mask = torch.nn.functional.interpolate(
                mask,
                size=(height, width),
                mode='nearest'
            )
            return upsampled_mask * pert

    def kernel_active_pool_min(self, mask):
        """
        Apply a minimum activation pooling to the mask.
        This is used when we want to find the minimum activation
        value within each kernel region.
        """
        # Ensure mask has the right dimensions
        batch_size, channels, height, width = mask.shape
        
        # Apply negation and max pooling (which is equivalent to min pooling)
        neg_mask = -mask
        pooled = self.kernel_active_pool(neg_mask)
        
        # Negate back to get the minimum values
        result = -pooled
        
        # Ensure the output has the right dimensions
        if result.shape[2:] != (self.data_h, self.data_w):
            result = torch.nn.functional.interpolate(
                result, 
                size=(self.data_h, self.data_w),
                mode='nearest'
            )
            
        return result
    
    def kernel_active_pool_avg(self, mask):
        """
        Apply average pooling to the mask.
        This is used when we want to get the average activation
        value within each kernel region.
        """
        # Ensure mask has the right dimensions
        batch_size, channels, height, width = mask.shape
        
        # Apply average pooling
        pooled = self.kernel_active_pool(mask)
        
        # Ensure the output has the right dimensions
        if pooled.shape[2:] != (self.data_h, self.data_w):
            pooled = torch.nn.functional.interpolate(
                pooled, 
                size=(self.data_h, self.data_w),
                mode='nearest'
            )
            
        return pooled

    def parse_kernel_args(self):
        """
        Initialize kernel-related parameters and operations.
        
        This method sets up the dimensions and operations for the kernel-based attack:
        - kernel_w, kernel_h: width and height of the grid of kernel centers
        - data_w, data_h: width and height of the input images
        - n_data_kernels: total number of possible kernel locations
        - kernel_max_pool: max pooling operation for kernels
        """
        # Make sure data dimensions are properly initialized
        if not hasattr(self, 'data_w') or not hasattr(self, 'data_h'):
            if hasattr(self, 'data_shape') and len(self.data_shape) >= 2:
                self.data_h = self.data_shape[-2]  # Height is second-to-last dimension
                self.data_w = self.data_shape[-1]  # Width is last dimension
            else:
                # Default to 224x224 images if not specified
                self.data_h = 224
                self.data_w = 224
        
        # Make sure batch_size is initialized
        if not hasattr(self, 'batch_size'):
            self.batch_size = 1  # Default to batch size of 1
            
        # Make sure n_data_pixels is initialized
        if not hasattr(self, 'n_data_pixels'):
            self.n_data_pixels = self.data_h * self.data_w
                
        # Kernel dimensions: how many kernel centers we can have in the image
        self.kernel_w = self.data_w - self.kernel_size + 1
        self.kernel_h = self.data_h - self.kernel_size + 1
        self.n_data_kernels = self.kernel_w * self.kernel_h
        
        # Set up mask shapes
        self.mask_shape = [self.batch_size, 1, self.kernel_h, self.kernel_w]
        self.mask_shape_flat = [self.batch_size, self.n_data_kernels]
        
        # Initialize mask tensors
        self.mask_zeros_flat = torch.zeros(self.mask_shape_flat, dtype=torch.bool, device=self.device,
                                           requires_grad=False)
        self.mask_ones_flat = torch.ones(self.mask_shape_flat, dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        
        # Set the mask application method
        self.apply_mask_method = self.apply_mask_kernel
        
        # Configure dropout if needed
        if self.apply_dpo:
            self.active_dpo = self.kernel_dpo
            self.dropout_str = "kernel " + self.dropout_str

        # We don't need to use padding/pooling operations anymore - instead we use custom functions
        # that handle dimensions correctly
        
        # Set up pooling methods for kernel activation
        if self.kernel_min_active:
            self.kernel_active_pool = torch.nn.MaxPool2d(self.kernel_size, stride=1, padding=self.kernel_size//2)
            self.kernel_active_pool_method = self.kernel_active_pool_min
        else:
            self.kernel_active_pool = torch.nn.AvgPool2d(self.kernel_size, stride=1, padding=self.kernel_size//2)
            self.kernel_active_pool_method = self.kernel_active_pool_avg

        # Configure kernel group parameters if needed
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

    # Safe max pooling that ensures dimensions match exactly
    def safe_max_pool(self, x):
        """
        Apply max pooling with guaranteed output dimensions.
        
        The issue is that when applying max_pool2d to a padded input, the output
        dimensions can be different from the original input dimensions due to padding
        and the kernel size. This function ensures the output has the exact dimensions we need.
        """
        # Get the expected output dimensions
        batch_size, channels, height, width = x.shape
        expected_height = self.data_h  
        expected_width = self.data_w
        
        # Calculate padding needed to ensure output dimensions match exactly
        # For kernels of size k and stride 1, the output size is (input_size - k + 1)
        # So to get output_size = input_size, we need padding = (k - 1) / 2 on each side
        padding = (self.kernel_size - 1) // 2
        
        # Apply max pooling with custom padding to ensure dimensions match
        pooled = torch.nn.functional.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding
        )
        
        # Double-check dimensions and fix if needed
        if pooled.shape[2] != expected_height or pooled.shape[3] != expected_width:
            print(f"Warning: pooled dimensions {pooled.shape[2:]} don't match expected {(expected_height, expected_width)}")
            # If they still don't match, resize to expected dimensions
            pooled = torch.nn.functional.interpolate(
                pooled, 
                size=(expected_height, expected_width),
                mode='nearest'
            )
        
        return pooled

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
        n_trim_kernels = n_trim_pixels // self.n_kernel_pixels
        return super(PGDTrimKernel, self).trim_pert_pixels(x, y, mask, dense_pert, sparse_pert,
                         best_pixels_crit, best_pixels_mask, best_pixels_loss,
                         mask_prep, mask_sample_method, mask_trim,
                         n_trim_kernels, trim_ratio, n_mask_samples)

    def perturb(self, x, y, targeted=False):
        # Ensure x has proper 4D shape (batch, channels, height, width)
        if x.dim() != 4:
            raise ValueError(f"Input tensor must have 4 dimensions, but got shape {x.shape}")
        
        # Call the parent class perturb method
        return super(PGDTrimKernel, self).perturb(x, y, targeted)

