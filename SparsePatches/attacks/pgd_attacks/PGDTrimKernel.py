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
        return self.kernel_max_pool(self.kernel_pad(mask.to(self.dtype))) * pert

    def kernel_active_pool_min(self, mask):
        return - self.kernel_active_pool(-mask)
    
    def kernel_active_pool_avg(self, mask):
        return self.kernel_active_pool(mask)

    def parse_kernel_args(self):

        self.kernel_w = self.data_w - self.kernel_size + 1
        self.kernel_h = self.data_h - self.kernel_size + 1
        self.n_data_kernels = self.kernel_w * self.kernel_h
        self.mask_shape = [self.batch_size, 1, self.kernel_w, self.kernel_h]
        self.mask_shape_flat = [self.batch_size, self.n_data_kernels]
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

