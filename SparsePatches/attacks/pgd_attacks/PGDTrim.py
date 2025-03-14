import numpy as np
import math
import itertools
import torch
from attacks.pgd_attacks.attack import Attack


class PGDTrim(Attack):
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

        super(PGDTrim, self).__init__(model, criterion, misc_args, pgd_args, dropout_args)

        self.name = "PGDTrim"
        self.sparsity = trim_args['sparsity']
        self.trim_steps = trim_args['trim_steps']
        self.max_trim_steps = trim_args['max_trim_steps']
        self.trim_steps_reduce_policy = trim_args['trim_steps_reduce']
        self.scale_dpo_mean = trim_args['scale_dpo_mean']
        self.enable_post_trim_dpo = trim_args['post_trim_dpo']
        self.dynamic_trim = trim_args['dynamic_trim']
        if self.trim_steps_reduce_policy == 'none':
            self.trim_steps_method = self.trim_steps_full
            self.rest_trim_steps_method = self.rest_trim_steps_unchanged
        elif self.trim_steps_reduce_policy == 'best':
            self.trim_steps_method = self.trim_steps_full
            self.rest_trim_steps_method = self.rest_trim_steps_topk
        else:
            self.trim_steps_reduce_policy = 'even'
            self.trim_steps_method = self.trim_steps_reduce_even
            self.rest_trim_steps_method = self.rest_trim_steps_unchanged
        if self.scale_dpo_mean:
            self.dropout_str += ", scaled by trim ratio"
        if self.enable_post_trim_dpo:
            self.post_trim_dpo_mean = self.dpo_mean
            self.post_trim_dpo_std = self.dpo_std
        else:
            self.post_trim_dpo_mean = torch.zeros_like(self.dpo_mean)
            self.post_trim_dpo_std = torch.zeros_like(self.dpo_std)

        self.mask_dist_name = mask_args['mask_dist']
        self.mask_dist_str = self.mask_dist_name
        self.mask_prob_amp_rate = mask_args['mask_prob_amp_rate']
        self.norm_mask_amp = mask_args['norm_mask_amp']
        self.mask_opt_iter = mask_args['mask_opt_iter']
        self.n_mask_samples = mask_args['n_mask_samples']
        self.sample_all_masks = mask_args['sample_all_masks']
        self.trim_best_mask = mask_args['trim_best_mask']
        self.apply_mask_method = self.apply_mask
        
        if self.mask_opt_iter > 0:
            self.mask_opt_method = self.mask_opt_pgd
        else:
            self.mask_opt_method = self.mask_opt_none
        self.l0_norms = None
        self.n_l0_norms = None
        self.sample_mask_pixels = None
        self.mask_prep = None
        self.mask_dist = None
        self.mask_sample = None
        self.sample_mask_pixels_from_dense = None
        self.mask_prob_method = None
        self.parse_mask_args()
        self.mask_sample_curr = None

        self.rest_n_trim_steps = None
        self.rest_trim_steps = None
        self.rest_l0_indices = None
        self.rest_l0_copy_indices = None
        self.rest_trim_steps_ratios = None
        self.rest_dpo_mean_steps = None
        self.rest_dpo_std_steps = None
        self.rest_mask_prep = None
        self.rest_mask_sample = None
        self.rest_n_mask_samples = None
        self.rest_mask_trim = None
        self.preprocess_rest_trim_steps()
        
        self.output_l0_norms = self.l0_norms
        self.l0_norms_tensor = torch.tensor(self.l0_norms, dtype=self.dtype, device=self.device)

    def report_trim_schematics(self):
        if self.trim_steps_reduce_policy != 'best':
            for rest_idx, trim_steps in enumerate(self.rest_trim_steps):
                print("Attack restart index: " + str(rest_idx) + " L0 trim steps: " + str(trim_steps))
        else:
            all_trim_steps = self.rest_trim_steps[0]
            for rest_idx, n_trim_steps in enumerate(self.rest_n_trim_steps):
                if n_trim_steps == self.max_trim_steps:
                    print("Attack restart index: " + str(rest_idx) + " L0 trim steps: " + str(all_trim_steps))
                elif n_trim_steps == 1:
                    print("Attack restart index: " + str(rest_idx) + " L0 trim steps: " + str([all_trim_steps[-1]]))
                else:
                    print("Attack restart index: " + str(rest_idx) + " number L0 trim steps: " + str(n_trim_steps))
                    print("Trim steps will be steps with top minimal decay of average loss to L0 value "
                          "over previous restarts")
                    
    def report_schematics(self):

        print("Running novel PGDTrim attack")
        print("The attack will gradually trim a dense perturbation to the specified sparsity: " + str(self.sparsity))
        
        print("Perturbations will be computed for the L0 norms:")
        print(self.l0_norms)
        print("The best performing perturbations will be reported for the L0 norms:")
        print(self.output_l0_norms)
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
    
    def parse_mask_args(self):

        if self.mask_dist_name == 'bernoulli':
            self.mask_sample = self.mask_sample_from_dist
            self.mask_prep = self.mask_prep_dist
            self.mask_dist = self.mask_dist_bernoulli
            if self.norm_mask_amp:
                self.sample_mask_pixels = self.sample_mask_pixels_normalize_amp
                self.mask_dist_str = 'bernoulli with normalized amplitude'
            else:
                self.sample_mask_pixels = self.sample_mask_pixels_unknown_count
        elif self.mask_dist_name == 'cbernoulli':
            self.mask_sample = self.mask_sample_from_dist
            self.mask_prep = self.mask_prep_dist
            self.mask_dist = self.mask_dist_continuous_bernoulli
            if self.norm_mask_amp:
                self.sample_mask_pixels = self.sample_mask_pixels_normalize_amp
                self.mask_dist_str = 'continuous bernoulli with normalized amplitude'
            else:
                self.sample_mask_pixels = self.sample_mask_pixels_unknown_count
                self.mask_dist_str = 'continuous bernoulli'
        elif self.mask_dist_name == 'topk':
            self.mask_sample = self.mask_sample_const
            self.mask_prep = self.mask_prep_topk
            self.sample_mask_pixels = self.sample_mask_pixels_known_count
        else:  # multinomial
            self.mask_sample = self.mask_sample_multinomial
            self.mask_prep = self.mask_prep_multinomial
            self.sample_mask_pixels = self.sample_mask_pixels_known_count

        if self.mask_prob_amp_rate > 0:
            self.mask_prob_method = self.mask_prob_by_amp
        else:
            self.mask_prob_method = self.mask_prob_uniform
    
    # Trim steps preprocessing utilities

    def compute_rest_skip_indices(self, rest_l0_indices):
        rest_l0_skip_indices = []
        for l0_indices in rest_l0_indices:
            l0_skip_indices = []
            for prev_idx, next_idx in zip(l0_indices, l0_indices[1:]):
                l0_skip_indices.extend(list(range(prev_idx + 1, next_idx)))
            rest_l0_skip_indices.append(l0_skip_indices)
        return rest_l0_skip_indices
    
    def compute_rest_copy_indices(self, rest_l0_skip_indices):
        rest_l0_copy_indices = []
        l0_norm_prev_computed = [0] * self.n_l0_norms
        for rest_idx, l0_indices in enumerate(self.rest_l0_indices):
            l0_skip_indices = rest_l0_skip_indices[rest_idx]
            rest_l0_copy_indices.append(
                [l0_skip_idx for l0_skip_idx in l0_skip_indices if l0_norm_prev_computed[l0_skip_idx]])
            for l0_idx in l0_indices:
                l0_norm_prev_computed[l0_idx] = True
        return rest_l0_copy_indices
    
    def compute_l0_norms(self):
        pixels_log_size = int(np.log2(self.n_data_pixels))
        max_trim_size = 2 ** pixels_log_size
        if max_trim_size < self.n_data_pixels:
            n_trim_options = int(np.ceil(np.log2(max_trim_size / self.sparsity)))
            all_l0_norms = [self.n_data_pixels] + [max_trim_size >> step for step in range(n_trim_options)] + [
                self.sparsity]
        else:
            n_trim_options = int(np.ceil(np.log2(self.n_data_pixels / self.sparsity))) - 1
            all_l0_norms = [self.n_data_pixels >> step for step in range(n_trim_options + 1)] + [self.sparsity]
        return n_trim_options, all_l0_norms
    
    def compute_trim_options(self):
        n_trim_options, all_l0_norms = self.compute_l0_norms()
        sparsity_trim_idx = len(all_l0_norms) - 1
        if self.max_trim_steps > n_trim_options + 1:
            self.max_trim_steps = n_trim_options + 1
        
        if self.n_restarts < self.max_trim_steps:
            repeat = 1
            if self.n_restarts == 1:
                step_size_list = [self.max_trim_steps]
            else:
                step_size_offset = (self.n_restarts - 1) - (self.max_trim_steps - 1) % (self.n_restarts - 1)
                step_size_list = [1] + [int((self.max_trim_steps - 1) / (self.n_restarts - 1)) + (i > step_size_offset)
                                        for i in range(1, self.n_restarts)]
            rest_n_trim_steps = list(reversed(list(itertools.accumulate(step_size_list))))
        
        else:
            repeat = int(np.ceil(self.n_restarts / self.max_trim_steps))
            rest_n_trim_steps = list(reversed(list(range(1, self.max_trim_steps + 1))))
        return all_l0_norms, rest_n_trim_steps, n_trim_options, sparsity_trim_idx, repeat
    
    def compute_even_trim_steps(self, all_l0_norms, n_trim_steps, sparsity_trim_idx):
        step_size_offset = n_trim_steps - sparsity_trim_idx % n_trim_steps
        step_size_list = [int(sparsity_trim_idx / n_trim_steps) + (i > step_size_offset) for i in
                          range(1, n_trim_steps)]
        steps_list = list(itertools.accumulate(step_size_list))
        trim_steps = [all_l0_norms[step] for step in steps_list] + [self.sparsity]
        return steps_list, trim_steps
    
    def process_trim_steps(self, trim_steps):
        curr_norm_ls = [self.n_data_pixels] + trim_steps[:-1]
        curr_norm = torch.tensor(curr_norm_ls, dtype=self.dtype, device=self.device)
        next_norm = torch.tensor(trim_steps, dtype=self.dtype, device=self.device)
        trim_steps_ratios = next_norm / curr_norm
        if self.scale_dpo_mean:
            dpo_mean_steps = self.dpo_mean * trim_steps_ratios
        else:
            dpo_mean_steps = self.dpo_mean.expand(len(trim_steps)).to(dtype=self.dtype, device=self.device)
        dpo_std_steps = self.compute_dpo_std(dpo_mean_steps)
        
        mask_prep = [self.mask_prep] * len(trim_steps)
        mask_sample = [self.mask_sample] * len(trim_steps)
        n_mask_samples = [self.n_mask_samples] * len(trim_steps)
        mask_trim = [self.mask_trim_best_pixels_crit] * len(trim_steps)
        if self.sample_all_masks:
            comb_size_ls = []
            sample_all_masks_ind = []
            for idx, curr_norm in enumerate(curr_norm_ls):
                next_norm = trim_steps[idx]
                if curr_norm <= self.n_mask_samples:
                    comb_size = math.comb(curr_norm, next_norm)
                    if comb_size <= self.n_mask_samples:
                        sample_all_masks_ind.append(idx)
                else:
                    comb_size = 0
                comb_size_ls.append(comb_size)

            for idx in sample_all_masks_ind:
                mask_prep[idx] = self.mask_prep_comb
                mask_sample[idx] = self.mask_sample_comb
                n_mask_samples[idx] = comb_size_ls[idx]
                if self.trim_best_mask > 1:
                    mask_trim[idx] = self.mask_trim_best_mask
                elif self.trim_best_mask > 0 and idx == (len(trim_steps) - 1):
                    mask_trim[idx] = self.mask_trim_best_mask
        
        return trim_steps_ratios, dpo_mean_steps, dpo_std_steps, mask_prep, mask_sample, n_mask_samples, mask_trim
    
    # Trim steps preprocessing

    def trim_steps_reduce_even(self, all_l0_norms, rest_n_trim_steps, n_trim_options, sparsity_trim_idx):
        all_steps_lists = []
        rest_trim_steps = []
        l0_norm_is_computed = [1] + [0] * n_trim_options + [1]
        for n_trim_steps in rest_n_trim_steps:
            steps_list, trim_steps = self.compute_even_trim_steps(all_l0_norms, n_trim_steps, sparsity_trim_idx)
            rest_trim_steps.append(trim_steps)
            for step in steps_list:
                l0_norm_is_computed[step] = 1
            all_steps_lists.append([0] + steps_list + [sparsity_trim_idx])
        
        steps_l0_indices = list(itertools.accumulate(l0_norm_is_computed, initial=0))
        rest_l0_indices = [[steps_l0_indices[step] for step in steps_list] for steps_list in all_steps_lists]
        l0_norms = [l0_norm for l0_norm_idx, l0_norm in enumerate(all_l0_norms) if l0_norm_is_computed[l0_norm_idx]]
        n_l0_norms = len(l0_norms)
        return l0_norms, n_l0_norms, rest_trim_steps, rest_l0_indices
    
    def trim_steps_full(self, all_l0_norms, rest_n_trim_steps, n_trim_options, sparsity_trim_idx):
        l0_norm_is_computed = [1] + [0] * n_trim_options + [1]
        steps_list, trim_steps = self.compute_even_trim_steps(all_l0_norms, rest_n_trim_steps[0], sparsity_trim_idx)
        for step in steps_list:
            l0_norm_is_computed[step] = 1
        
        l0_norms = [l0_norm for l0_norm_idx, l0_norm in enumerate(all_l0_norms) if l0_norm_is_computed[l0_norm_idx]]
        n_l0_norms = len(l0_norms)
        rest_trim_steps = [trim_steps for n_trim_steps in rest_n_trim_steps]
        rest_l0_indices = [list(range(n_l0_norms)) for n_trim_steps in rest_n_trim_steps]
        return l0_norms, n_l0_norms, rest_trim_steps, rest_l0_indices
    
    def preprocess_rest_trim_steps(self):
        if self.trim_steps is not None:
            self.sparsity = None
            self.max_trim_steps = None
            self.l0_norms = [self.n_data_pixels] + self.trim_steps
            self.n_l0_norms = len(self.l0_norms)
            self.rest_trim_steps_method = self.rest_trim_steps_unchanged

            trim_steps_ratios, dpo_mean_steps, dpo_std_steps, mask_prep, mask_sample, n_mask_samples, mask_trim\
                = self.process_trim_steps(self.trim_steps)
            self.rest_trim_steps = [self.trim_steps] * self.n_restarts
            self.rest_l0_indices = [list(range(self.n_l0_norms))] * self.n_restarts
            self.rest_l0_copy_indices = [[]] * self.n_restarts
            self.rest_trim_steps_ratios = [trim_steps_ratios] * self.n_restarts
            self.rest_dpo_mean_steps = [dpo_mean_steps] * self.n_restarts
            self.rest_dpo_std_steps = [dpo_std_steps] * self.n_restarts
            self.rest_mask_prep = [mask_prep] * self.n_restarts
            self.rest_mask_sample = [mask_sample] * self.n_restarts
            self.rest_n_mask_samples = [n_mask_samples] * self.n_restarts
            self.rest_mask_trim = [mask_trim] * self.n_restarts
            return
        elif self.max_trim_steps < 2:
            self.l0_norms = [self.n_data_pixels] + [self.sparsity]
            self.n_l0_norms = 2
            self.rest_trim_steps_method = self.rest_trim_steps_unchanged
            
            trim_steps_ratios, dpo_mean_steps, dpo_std_steps, mask_prep, mask_sample, n_mask_samples, mask_trim\
                = self.process_trim_steps([self.sparsity])
            self.rest_trim_steps = [[self.sparsity]] * self.n_restarts
            self.rest_l0_indices = [[0, 1]] * self.n_restarts
            self.rest_l0_copy_indices = [[]] * self.n_restarts
            self.rest_trim_steps_ratios = [trim_steps_ratios] * self.n_restarts
            self.rest_dpo_mean_steps = [dpo_mean_steps] * self.n_restarts
            self.rest_dpo_std_steps = [dpo_std_steps] * self.n_restarts
            self.rest_mask_prep = [mask_prep] * self.n_restarts
            self.rest_mask_sample = [mask_sample] * self.n_restarts
            self.rest_n_mask_samples = [n_mask_samples] * self.n_restarts
            self.rest_mask_trim = [mask_trim] * self.n_restarts
            return
        
        all_l0_norms, rest_n_trim_steps, n_trim_options, sparsity_trim_idx, repeat = self.compute_trim_options()
        self.l0_norms, self.n_l0_norms, rest_trim_steps, rest_l0_indices = (
            self.trim_steps_method(all_l0_norms, rest_n_trim_steps, n_trim_options, sparsity_trim_idx))
        rest_l0_skip_indices = self.compute_rest_skip_indices(rest_l0_indices)
        
        self.rest_n_trim_steps = (rest_n_trim_steps * repeat)[:self.n_restarts]
        self.rest_trim_steps = (rest_trim_steps * repeat)[:self.n_restarts]
        self.rest_l0_indices = (rest_l0_indices * repeat)[:self.n_restarts]
        rest_l0_skip_indices = (rest_l0_skip_indices * repeat)[:self.n_restarts]
        self.rest_l0_copy_indices = self.compute_rest_copy_indices(rest_l0_skip_indices)
        
        self.rest_trim_steps_ratios = []
        self.rest_dpo_mean_steps = []
        self.rest_dpo_std_steps = []
        self.rest_mask_prep = []
        self.rest_mask_sample = []
        self.rest_n_mask_samples = []
        self.rest_mask_trim = []
        for trim_steps in self.rest_trim_steps:
            trim_steps_ratios, dpo_mean_steps, dpo_std_steps, mask_prep, mask_sample, n_mask_samples, mask_trim\
                = self.process_trim_steps(trim_steps)
            self.rest_trim_steps_ratios.append(trim_steps_ratios)
            self.rest_dpo_mean_steps.append(dpo_mean_steps)
            self.rest_dpo_std_steps.append(dpo_std_steps)
            self.rest_mask_prep.append(mask_prep)
            self.rest_mask_sample.append(mask_sample)
            self.rest_n_mask_samples.append(n_mask_samples)
            self.rest_mask_trim.append(mask_trim)
        return
    
    # Trim steps update during perturb restarts

    def rest_trim_steps_unchanged(self, rest_idx, best_l0_loss):
        trim_steps = self.rest_trim_steps[rest_idx]
        l0_indices = self.rest_l0_indices[rest_idx]
        l0_copy_indices = self.rest_l0_copy_indices[rest_idx]
        trim_steps_ratios = self.rest_trim_steps_ratios[rest_idx]
        dpo_mean_steps = self.rest_dpo_mean_steps[rest_idx]
        dpo_std_steps = self.rest_dpo_std_steps[rest_idx]
        steps_mask_prep = self.rest_mask_prep[rest_idx]
        steps_mask_sample = self.rest_mask_sample[rest_idx]
        steps_n_mask_samples = self.rest_n_mask_samples[rest_idx]
        steps_mask_trim = self.rest_mask_trim[rest_idx]
        
        return (trim_steps, l0_indices, l0_copy_indices,
                trim_steps_ratios, dpo_mean_steps, dpo_std_steps,
                steps_mask_prep, steps_mask_sample, steps_n_mask_samples, steps_mask_trim)
    
    def rest_trim_steps_topk(self, rest_idx, best_l0_loss):
        n_trim_steps = self.rest_n_trim_steps[rest_idx]
        if n_trim_steps == self.n_l0_norms - 1:
            return self.rest_trim_steps_unchanged(rest_idx, best_l0_loss)
        elif n_trim_steps == 1:
            l0_indices = [0] + [self.n_l0_norms - 1]
            trim_steps = [self.l0_norms[-1]]
            l0_copy_indices = list(range(1, len(self.l0_norms) - 1))
        else:
            avg_loss_inc_to_l0 = (best_l0_loss - self.clean_loss.unsqueeze(0)).mean(dim=1) / self.l0_norms_tensor
            loss_compression_decay = -avg_loss_inc_to_l0.diff()[:-1]
            trim_indices = sorted(loss_compression_decay.topk(n_trim_steps - 1, largest=False, sorted=False)[1]
                                  .tolist())
            l0_indices = [0] + [idx + 1 for idx in trim_indices] + [self.n_l0_norms - 1]
            
            trim_steps = [self.l0_norms[idx] for idx in l0_indices[1:]]
            l0_copy_indices = [idx for idx in range(self.n_l0_norms) if idx not in l0_indices]
        
        trim_steps_ratios, dpo_mean_steps, dpo_std_steps, steps_mask_prep, steps_mask_sample, steps_n_mask_samples, steps_mask_trim \
            = self.process_trim_steps(trim_steps)
        
        self.rest_trim_steps[rest_idx] = trim_steps
        self.rest_l0_indices[rest_idx] = l0_indices
        self.rest_l0_copy_indices[rest_idx] = l0_copy_indices
        self.rest_trim_steps_ratios[rest_idx] = trim_steps_ratios
        self.rest_dpo_mean_steps[rest_idx] = dpo_mean_steps
        self.rest_dpo_std_steps[rest_idx] = dpo_std_steps
        self.rest_mask_prep[rest_idx] = steps_mask_prep
        self.rest_mask_sample[rest_idx] = steps_mask_sample
        self.rest_n_mask_samples[rest_idx] = steps_n_mask_samples
        self.rest_mask_trim[rest_idx] = steps_mask_trim

        return (trim_steps, l0_indices, l0_copy_indices,
                trim_steps_ratios, dpo_mean_steps, dpo_std_steps,
                steps_mask_prep, steps_mask_sample, steps_n_mask_samples, steps_mask_trim)
    
    # mask computation utilities

    def mask_from_ind(self, mask_indices):
        return self.mask_zeros_flat.scatter(dim=1, index=mask_indices, src=self.mask_ones_flat).view(self.mask_shape)

    def apply_mask(self, mask, pert):
        return mask * pert
    # mask probabilities computation methods

    def mask_prob_uniform(self, pert, mask, n_trim_pixels, trim_ratio):
        return trim_ratio * mask.view(self.batch_size, -1).float()
    
    def mask_prob_by_amp(self, pert, mask, n_trim_pixels, trim_ratio):
        mask_amp = torch.linalg.norm(pert.view(self.batch_size, self.data_channels, -1), ord=2, dim=1)
        mask_amp_ratio = n_trim_pixels / mask_amp.sum(dim=1, keepdims=True) * mask_amp
        prob = ((trim_ratio * mask.view(self.batch_size, -1) + self.mask_prob_amp_rate * mask_amp_ratio)
                / (self.mask_prob_amp_rate + 1))
        return prob.float()

    # mask distribution methods

    def mask_dist_multinomial(self, pixels_prob, n_trim_pixels_tensor):
        return torch.distributions.multinomial.Multinomial(total_count=n_trim_pixels_tensor, probs=pixels_prob)

    def mask_dist_bernoulli(self, pixels_prob, n_trim_pixels_tensor):
        return torch.distributions.Bernoulli(probs=pixels_prob)

    def mask_dist_continuous_bernoulli(self, pixels_prob, n_trim_pixels_tensor):
        return torch.distributions.ContinuousBernoulli(probs=pixels_prob)

    # mask prep methods

    def mask_prep_topk(self, pixels_prob, n_trim_pixels_tensor):
        return self.mask_from_ind(pixels_prob.view(self.batch_size, -1).topk(n_trim_pixels_tensor, dim=1, sorted=False)[1])

    def mask_prep_multinomial(self, pixels_prob, n_trim_pixels_tensor):
        return pixels_prob, n_trim_pixels_tensor

    def mask_prep_dist(self, pixels_prob, n_trim_pixels_tensor):
        return self.mask_dist(pixels_prob, n_trim_pixels_tensor)

    def mask_prep_comb(self, pixels_prob, n_trim_pixels_tensor):
        active_pixels_ind = pixels_prob.nonzero(as_tuple=True)[1].view(self.batch_size, -1)
        n_active_pixels = active_pixels_ind.shape[1]
        trim_ind_from_active = torch.arange(end=n_active_pixels, dtype=self.dtype, device=self.device)
        trim_comb = torch.combinations(trim_ind_from_active, r=n_trim_pixels_tensor).to(dtype=torch.int)
        return trim_comb, active_pixels_ind

    # mask sample methods

    def mask_sample_const(self, mask, n_trim_pixels_tensor):
        return mask

    def mask_sample_multinomial(self, mask_prep, index):
        pixels_prob, n_trim_pixels_tensor = mask_prep
        return self.mask_from_ind(pixels_prob.multinomial(n_trim_pixels_tensor, replacement=False))

    def mask_sample_from_dist(self, mask_dist, index):
        return mask_dist.sample()

    def mask_sample_comb(self, mask_prep, index):
        trim_comb, active_pixels_ind = mask_prep
        trim_ind_from_active = trim_comb[index]
        return self.mask_from_ind(active_pixels_ind[:, trim_ind_from_active])

    def sample_mask_pixels_known_count(self, mask_sample_method, mask_prep, n_trim_pixels_tensor, sample_idx):
        mask_sample = mask_sample_method(mask_prep, sample_idx)
        return mask_sample, n_trim_pixels_tensor

    def sample_mask_pixels_unknown_count(self, mask_sample_method, mask_prep, n_trim_pixels_tensor, sample_idx):
        mask_sample = mask_sample_method(mask_prep, sample_idx)
        sample_pixel_count = mask_sample.view(self.batch_size, -1).sum(dim=1)
        mask_sample = mask_sample.view(self.mask_shape)
        return mask_sample, sample_pixel_count

    def sample_mask_pixels_normalize_amp(self, mask_sample_method, mask_prep, n_trim_pixels_tensor, sample_idx):
        mask_sample = mask_sample_method(mask_prep, sample_idx)
        mask_sample = ((mask_sample * (n_trim_pixels_tensor / mask_sample.sum(dim=1)).unsqueeze(1))
                       .view(self.mask_shape))
        return mask_sample, n_trim_pixels_tensor

    # mask opt methods
    
    def mask_opt_none(self, x, y, mask, dense_pert):
        sparse_pert = self.apply_mask_method(mask, dense_pert)
        output, loss = self.test_pert(x, y, sparse_pert)
        return loss
    
    def mask_opt_pgd(self, x, y, mask, dense_pert):
        dense_pert_clone = dense_pert.clone().detach()
        sparse_pert = self.apply_mask_method(mask, dense_pert_clone)
        
        loss = torch.zeros_like(self.clean_loss)
        succ = torch.zeros_like(self.clean_succ)
        self.pgd(x, y, mask, dense_pert_clone, sparse_pert, loss, succ,
                 0, 0, n_iter=self.mask_opt_iter)
        return loss
    
    # mask trimming methods
    def mask_active_pixels(self, mask):
        return mask
    
    def compute_pixels_crit(self, x, y, dense_pert, best_pixels_crit,
                            mask_sample_method, mask_prep_data, n_trim_pixels_tensor, n_mask_samples):
        pixel_sample_count = torch.zeros(self.mask_shape, dtype=self.dtype, device=self.device)
        pixel_loss_sum = torch.zeros(self.mask_shape, dtype=self.dtype, device=self.device)
        
        for sample_idx in range(n_mask_samples):
            mask_sample, sample_pixel_count = self.sample_mask_pixels(mask_sample_method, mask_prep_data,
                                                                      n_trim_pixels_tensor, sample_idx)
            loss = self.mask_opt_method(x, y, mask_sample, dense_pert)
            sample_active_pixels = self.mask_active_pixels(mask_sample)
            pixel_sample_count += sample_active_pixels
            pixel_loss_sum += sample_active_pixels * (loss / sample_pixel_count).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        pixel_sample_count.clamp_(min=1)  # Avoid zero values in tensor
        pixels_crit = pixel_loss_sum / pixel_sample_count
        if self.dynamic_trim:
            pixels_crit_improve = pixels_crit.ge(best_pixels_crit)
            best_pixels_crit[pixels_crit_improve] = pixels_crit[pixels_crit_improve]
            pixels_crit = best_pixels_crit
        return pixels_crit
    
    def mask_trim_best_pixels_crit(self, x, y, dense_pert,
                                   best_pixels_crit, best_pixels_mask, best_pixels_loss,
                                   mask_sample_method, mask_prep_data, n_trim_pixels_tensor, n_mask_samples):
        with torch.no_grad():
            pixels_crit = self.compute_pixels_crit(x, y, dense_pert, best_pixels_crit,
                                                   mask_sample_method, mask_prep_data,
                                                   n_trim_pixels_tensor, n_mask_samples)

            best_crit_mask_indices = pixels_crit.view(self.batch_size, -1
                                                      ).topk(n_trim_pixels_tensor, dim=1, sorted=False)[1]
            best_crit_mask = self.mask_from_ind(best_crit_mask_indices)
        return best_crit_mask

    def mask_trim_best_mask(self, x, y, dense_pert,
                            best_pixels_crit, best_pixels_mask, best_pixels_loss,
                            mask_sample_method, mask_prep_data, n_trim_pixels_tensor, n_mask_samples):
        with torch.no_grad():
            best_mask, sample_pixel_count = self.sample_mask_pixels(mask_sample_method, mask_prep_data,
                                                                    n_trim_pixels_tensor, sample_idx=0)
            best_loss = self.mask_opt_method(x, y, best_mask, dense_pert)

            for sample_idx in range(1, n_mask_samples):
                mask_sample, sample_pixel_count = self.sample_mask_pixels(mask_sample_method, mask_prep_data,
                                                                          n_trim_pixels_tensor, sample_idx)
                loss = self.mask_opt_method(x, y, mask_sample, dense_pert)
                self.update_best(best_loss, loss,
                                 [best_mask],
                                 [mask_sample])
                
            if self.dynamic_trim:
                self.update_best(best_pixels_loss, best_loss,
                                 [best_pixels_mask],
                                 [best_mask])
                best_mask = best_pixels_mask
                
        return best_mask

    def trim_pert_pixels(self, x, y, mask, dense_pert, sparse_pert,
                         best_pixels_crit, best_pixels_mask, best_pixels_loss,
                         mask_prep, mask_sample_method, mask_trim,
                         n_trim_pixels, trim_ratio, n_mask_samples):
        with torch.no_grad():
            n_trim_pixels_tensor = torch.tensor(n_trim_pixels, dtype=torch.int, device=self.device)
            pixels_prob = self.mask_prob_method(sparse_pert, mask, n_trim_pixels_tensor, trim_ratio)
            mask_prep_data = mask_prep(pixels_prob, n_trim_pixels_tensor)
            return mask_trim(x, y, dense_pert,
                                best_pixels_crit, best_pixels_mask, best_pixels_loss,
                                mask_sample_method, mask_prep_data, n_trim_pixels_tensor, n_mask_samples)

    def pgd(self, x, y, mask, dense_pert, best_sparse_pert, best_loss, best_succ, dpo_mean, dpo_std, n_iter=None):
        with torch.no_grad():
            report_info = self.report_info
            if n_iter is None:
                n_iter = self.n_iter
            else:
                report_info = False
            sparse_pert = self.apply_mask_method(mask, dense_pert)
            loss, succ = self.eval_pert(x, y, sparse_pert)
            self.update_best(best_loss, loss,
                             [best_sparse_pert, best_succ],
                             [sparse_pert, succ])
            
            if report_info:
                all_best_succ = torch.zeros(n_iter + 1, self.batch_size, dtype=torch.bool, device=self.device)
                all_best_loss = torch.zeros(n_iter + 1, self.batch_size, dtype=self.dtype, device=self.device)
                all_best_succ[0] = best_succ
                all_best_loss[0] = best_loss

        self.set_dpo(dpo_mean, dpo_std)
        self.model.eval()
        with torch.enable_grad():
            pert = dense_pert.clone().detach()
            for k in range(1, n_iter + 1):
                pert.requires_grad_()
                mask.requires_grad_(False)
                sparse_pert = self.apply_mask_method(mask, pert)
                train_loss = self.criterion(self.model.forward(x + self.dpo(sparse_pert)), y)
                grad = torch.autograd.grad(train_loss.mean(), [pert])[0].detach()
    
                with torch.no_grad():
                    pert = self.step(pert, grad)
                    sparse_pert = self.apply_mask_method(mask, pert)
                    eval_loss, succ = self.eval_pert(x, y, sparse_pert)
                    self.update_best(best_loss, eval_loss,
                                     [dense_pert, best_sparse_pert, best_succ],
                                     [pert, sparse_pert, succ])
                    
                    if report_info:
                        all_best_succ[k] = best_succ | all_best_succ[k - 1]
                        all_best_loss[k] = best_loss

        if report_info:
            return all_best_succ, all_best_loss
        return None, None

    def perturb(self, x, y, targeted=False):
        with (torch.no_grad()):
            self.set_params(x, targeted)
            self.clean_loss, self.clean_succ = self.eval_pert(x, y, pert=torch.zeros_like(x))
            best_l0_perts = torch.zeros_like(x).unsqueeze(0).repeat(self.n_l0_norms, 1, 1, 1, 1)
            best_l0_loss = self.clean_loss.clone().detach().unsqueeze(0).repeat(self.n_l0_norms, 1)
            best_l0_succ = self.clean_succ.clone().detach().unsqueeze(0).repeat(self.n_l0_norms, 1)
            best_l0_pixels_crit = torch.zeros(self.mask_shape, dtype=self.dtype, device=self.device
                                              ).unsqueeze(0).repeat(self.n_l0_norms, 1, 1, 1, 1)
            best_l0_pixels_mask = self.mask_zeros_flat.clone().detach(
            ).view(self.mask_shape).unsqueeze(0).repeat(self.n_l0_norms, 1, 1, 1, 1)
            best_l0_pixels_loss = torch.zeros_like(best_l0_loss)

            if self.report_info:
                all_best_succ = torch.zeros(self.n_l0_norms, self.n_restarts, self.n_iter + 1, self.batch_size, dtype=torch.bool, device=self.device)
                all_best_loss = torch.zeros(self.n_l0_norms, self.n_restarts, self.n_iter + 1, self.batch_size, dtype=self.dtype, device=self.device)

        for rest in range(self.n_restarts):
            with torch.no_grad():
                mask = self.mask_ones_flat.clone().detach().view(self.mask_shape)
                (trim_steps, l0_indices, l0_copy_indices,
                 trim_steps_ratios, dpo_mean_steps, dpo_std_steps,
                 steps_mask_prep, steps_mask_sample, steps_n_mask_samples, steps_mask_trim) =\
                    self.rest_trim_steps_method(rest, best_l0_loss)

                if self.rand_init:
                    dense_pert = self.random_initialization()
                else:
                    dense_pert = torch.zeros_like(x)

            for trim_idx, n_trim_pixels in enumerate(trim_steps):
                with torch.no_grad():
                    l0_idx = l0_indices[trim_idx]
                    trim_l0_idx = l0_indices[trim_idx + 1]
                    best_sparse_pert = best_l0_perts[l0_idx]
                    best_loss = best_l0_loss[l0_idx]
                    best_succ = best_l0_succ[l0_idx]
                    best_pixels_crit = best_l0_pixels_crit[trim_l0_idx]
                    best_pixels_mask = best_l0_pixels_mask[trim_l0_idx]
                    best_pixels_loss = best_l0_pixels_loss[trim_l0_idx]
                    trim_ratio = trim_steps_ratios[trim_idx]
                    dpo_mean = dpo_mean_steps[trim_idx]
                    dpo_std = dpo_std_steps[trim_idx]
                    mask_prep = steps_mask_prep[trim_idx]
                    mask_sample = steps_mask_sample[trim_idx]
                    n_mask_samples = steps_n_mask_samples[trim_idx]
                    mask_trim = steps_mask_trim[trim_idx]
                    
                no_trim_all_best_succ, no_trim_all_best_loss\
                    = self.pgd(x, y, mask, dense_pert, best_sparse_pert, best_loss, best_succ, dpo_mean, dpo_std)
                    
                with torch.no_grad():
                    if self.report_info:
                        all_best_succ[l0_idx, rest] = no_trim_all_best_succ
                        all_best_loss[l0_idx, rest] = no_trim_all_best_loss
                    if self.verbose:
                        pert_l0 = best_sparse_pert.abs().view(self.batch_size, self.data_channels, -1).sum(dim=1).count_nonzero(1)
                        print("Finished optimizing sparse perturbation on predetermined pixels")
                        print('max L0 in perturbation: ' + str(pert_l0.max().item()))
                        pert_l_inf = (best_sparse_pert.abs() / self.data_RGB_size).view(self.batch_size, -1).max(1)[0]
                        print('max L_inf in perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                            pert_l_inf.max(), (best_sparse_pert != best_sparse_pert).sum(), best_sparse_pert.max(), best_sparse_pert.min()))
                    
                    mask = self.trim_pert_pixels(x, y, mask, dense_pert, best_sparse_pert,
                                                 best_pixels_crit, best_pixels_mask, best_pixels_loss,
                                                 mask_prep, mask_sample, mask_trim,
                                                 n_trim_pixels, trim_ratio, n_mask_samples)
            
            with torch.no_grad():
                l0_idx = l0_indices[-1]
                best_sparse_pert = best_l0_perts[l0_idx]
                best_loss = best_l0_loss[l0_idx]
                best_succ = best_l0_succ[l0_idx]
            no_trim_all_best_succ, no_trim_all_best_loss \
                = self.pgd(x, y, mask, dense_pert, best_sparse_pert, best_loss, best_succ,
                           self.post_trim_dpo_mean, self.post_trim_dpo_std)
            with torch.no_grad():
                if self.report_info:
                    all_best_succ[l0_idx, rest] = no_trim_all_best_succ
                    all_best_loss[l0_idx, rest] = no_trim_all_best_loss
                    if len(l0_copy_indices):
                        all_best_succ[l0_copy_indices, rest] = all_best_succ[l0_copy_indices, rest - 1, -1].unsqueeze(1)
                        all_best_loss[l0_copy_indices, rest] = all_best_loss[l0_copy_indices, rest - 1, -1].unsqueeze(1)

                if self.verbose:
                    pert_l0 = best_sparse_pert.abs().view(self.batch_size, self.data_channels, -1).sum(dim=1).count_nonzero(1)
                    print("Finished optimizing perturbation without pixel trimming")
                    print('max L0 in perturbation: ' + str(pert_l0.max().item()))
                    pert_l_inf = (best_sparse_pert.abs() / self.data_RGB_size).view(self.batch_size, -1).max(1)[0]
                    print('max L_inf in perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                        pert_l_inf.max(), (best_sparse_pert != best_sparse_pert).sum(), best_sparse_pert.max(),
                        best_sparse_pert.min()))

        if self.report_info:
            return best_l0_perts, all_best_succ, all_best_loss

        return best_l0_perts, None, None


