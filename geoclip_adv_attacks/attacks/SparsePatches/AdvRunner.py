import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange


class AdvRunner:
    def __init__(self, model, attack, l0_hist_limits, data_RGB_size, device, dtype, verbose=False):
        self.attack = attack
        self.model = model
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.l0_norms = self.attack.output_l0_norms
        self.n_l0_norms = len(self.l0_norms)
        self.l0_hist_limits = l0_hist_limits
        self.data_channels = 3
        self.data_RGB_size = torch.tensor(data_RGB_size).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.attack_restarts = self.attack.n_restarts
        self.attack_iter = self.attack.n_iter
        self.attack_report_info = self.attack.report_info
        self.attack_name = self.attack.name

    def run_clean_evaluation(self, x_orig, y_orig, n_examples, bs, n_batches, orig_device):
        robust_flags = torch.zeros(n_examples, dtype=torch.bool, device=orig_device)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, n_examples)

            x = x_orig[start_idx:end_idx, :].clone().detach().to(self.device)
            y = y_orig[start_idx:end_idx].clone().detach().to(self.device)
            output = self.model.forward(x)
            correct_batch = y.eq(output.max(dim=1)[1]).detach().to(orig_device)
            robust_flags[start_idx:end_idx] = correct_batch

        n_robust_examples = torch.sum(robust_flags).item()
        init_accuracy = n_robust_examples / n_examples
        if self.verbose:
            print('initial accuracy: {:.2%}'.format(init_accuracy))
        return robust_flags, n_robust_examples, init_accuracy

    def process_results(self, n_examples, robust_flags, l0_norms_adv_perts, perts_l0_norms, l0_norms_robust_flags):

        l0_norms_robust_accuracy = (l0_norms_robust_flags.sum(dim=1) / n_examples).tolist()
        l0_norms_perts_max_l0 = perts_l0_norms[:, robust_flags].max(dim=1)[0].tolist()
        l0_norms_perts_min_l0 = perts_l0_norms[:, robust_flags].min(dim=1)[0].tolist()
        l0_norms_perts_mean_l0 = perts_l0_norms[:, robust_flags].mean(dim=1).tolist()
        l0_norms_perts_median_l0 = perts_l0_norms[:, robust_flags].median(dim=1)[0].tolist()
        l0_norms_perts_max_l_inf = (l0_norms_adv_perts[:, robust_flags].abs() / self.data_RGB_size).view(self.n_l0_norms, -1).max(dim=1)[0].tolist()
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
                perts_l0_limited_ratio = (l0_limit_flags.sum() / n_examples).item()
                l0_limit_robust_flags = l0_norms_robust_flags[l0_norm_idx, :].logical_or(~l0_limit_flags)
                l0_limit_robust_accuracy = (l0_limit_robust_flags.sum() / n_examples).item()
                hist_l0_ratio.append(perts_l0_limited_ratio)
                hist_l0_robust_accuracy.append(l0_limit_robust_accuracy)

            l0_norms_hist_l0_ratio.append(hist_l0_ratio)
            l0_norms_hist_l0_robust_accuracy.append(hist_l0_robust_accuracy)


        return l0_norms_robust_accuracy, l0_norms_perts_info, \
            l0_norms_hist_l0_limits, l0_norms_hist_l0_ratio, l0_norms_hist_l0_robust_accuracy

    def run_standard_evaluation(self, x_orig, y_orig, n_examples, bs=250):
        with torch.no_grad():
            orig_device = x_orig.device
            # calculate accuracy
            n_batches = int(np.ceil(n_examples / bs))
            robust_flags, n_robust_examples, init_accuracy = self.run_clean_evaluation(x_orig, y_orig, n_examples, bs, n_batches, orig_device)
            l0_norms_robust_flags = robust_flags.detach().unsqueeze(0).repeat(self.n_l0_norms, 1)

        x_adv = x_orig.clone().detach()
        y_adv = y_orig.clone().detach()
        l0_norms_adv_perts = torch.zeros_like(x_orig).detach().unsqueeze(0).repeat(self.n_l0_norms, 1, 1, 1, 1)
        l0_norms_adv_y = y_orig.clone().detach().unsqueeze(0).repeat(self.n_l0_norms, 1)
        perts_l0_norms = torch.zeros(self.n_l0_norms, n_examples, dtype=self.dtype, device=orig_device)
        if self.attack_report_info:
            info_shape = [self.n_l0_norms, self.attack_restarts, self.attack_iter + 1, n_examples]
            all_succ = torch.zeros(info_shape, dtype=self.dtype, device=orig_device)
            all_loss = torch.zeros(info_shape, dtype=self.dtype, device=orig_device)
        else:
            all_succ = None
            all_loss = None
        with torch.cuda.device(self.device):
            start_events = [torch.cuda.Event(enable_timing=True) for batch_idx in range(n_batches)]
            end_events = [torch.cuda.Event(enable_timing=True) for batch_idx in range(n_batches)]
            for batch_idx in trange(n_batches):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, n_examples)
                batch_indices = torch.arange(start_idx, end_idx, device=orig_device)
                x = x_orig[start_idx:end_idx, :].clone().detach().to(self.device)
                y = y_orig[start_idx:end_idx].clone().detach().to(self.device)
                
                # make sure that x is a 4d tensor even if there is only a single datapoint left
                if len(x.shape) == 3:
                    x.unsqueeze_(dim=0)
                start_events[batch_idx].record()
                batch_l0_norms_adv_perts, all_batch_succ, all_batch_loss = self.attack.perturb(x, y)
                end_events[batch_idx].record()
                torch.cuda.empty_cache()
                with torch.no_grad():

                    batch_x_adv = x + batch_l0_norms_adv_perts[-1]
                    x_adv[start_idx:end_idx] = batch_x_adv.detach().to(orig_device)
                    output = self.model.forward(batch_x_adv)
                    y_adv[start_idx:end_idx] = output.max(dim=1)[1].detach().to(orig_device)
                    l0_norms_adv_perts[:, start_idx:end_idx] = batch_l0_norms_adv_perts.detach().to(orig_device)
                    batch_l0_norms = batch_l0_norms_adv_perts.abs().view(
                        self.n_l0_norms, bs, self.data_channels, -1).sum(dim=2).count_nonzero(2).unsqueeze(0)

                    perts_l0_norms[:, start_idx:end_idx] = batch_l0_norms.to(orig_device)
                    if self.attack_report_info:
                        all_succ[:, :, :, start_idx:end_idx] = all_batch_succ.to(orig_device)
                        all_loss[:, :, :, start_idx:end_idx] = all_batch_loss.to(orig_device)

                    for l0_norm_idx in range(self.n_l0_norms):
                        l0_norm_batch_adv_x = x + batch_l0_norms_adv_perts[l0_norm_idx]
                        output = self.model.forward(l0_norm_batch_adv_x)
                        l0_norm_batch_adv_y = output.max(dim=1)[1]
                        false_batch = ~y.eq(l0_norm_batch_adv_y).detach().to(orig_device)
                        non_robust_indices = batch_indices[false_batch]
                        l0_norms_robust_flags[l0_norm_idx, non_robust_indices] = False
                        l0_norms_adv_y[l0_norm_idx, start_idx:end_idx] = l0_norm_batch_adv_y.detach().to(orig_device)
            torch.cuda.synchronize()
            adv_batch_compute_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            adv_batch_compute_time_mean = np.mean(adv_batch_compute_times)
            adv_batch_compute_time_std = np.std(adv_batch_compute_times)
            tot_adv_compute_time = np.sum(adv_batch_compute_times)
            tot_adv_compute_time_std = np.std([time * n_batches for time in adv_batch_compute_times])

        with torch.no_grad():
            l0_norms_robust_accuracy, l0_norms_perts_info, \
                l0_norms_hist_l0_limits, l0_norms_hist_l0_ratio, l0_norms_hist_l0_robust_accuracy = \
                self.process_results(n_examples, robust_flags, l0_norms_adv_perts, perts_l0_norms, l0_norms_robust_flags)
            l0_norms_perts_max_l0, l0_norms_perts_min_l0, l0_norms_perts_mean_l0, l0_norms_perts_median_l0, l0_norms_perts_max_l_inf = l0_norms_perts_info

            if self.verbose:
                print("reporting results for sparse adversarial attack: " + self.attack_name)
                print("Attack batches runtime mean: " + str(adv_batch_compute_time_mean) + " s")
                print("Attack batches runtime std: " + str(tot_adv_compute_time_std) + " s")
                print("Attack total runtime: " + str(tot_adv_compute_time) + " s")
                print("Attack total runtime std over batches: " + str(tot_adv_compute_time_std) + " s")
                print("reporting results for attacks with L0 norms values:")
                print(self.l0_norms)
                print("robust accuracy for each L0 norm:")
                print(l0_norms_robust_accuracy)
                print("perturbations max L0 for each L0 norm:")
                print(l0_norms_perts_max_l0)
                print("perturbations min L0 for each L0 norm:")
                print(l0_norms_perts_min_l0)
                print("perturbations mean L0 for each L0 norm:")
                print(l0_norms_perts_mean_l0)
                print("perturbations median L0 for each L0 norm:")
                print(l0_norms_perts_median_l0)
                print("perturbations max L_inf for each L0 norm:")
                print(l0_norms_perts_max_l_inf)
                print('nan in tensors: {}, max: {:.5f}, min: {:.5f}'.format(
                    (l0_norms_adv_perts != l0_norms_adv_perts).sum(), l0_norms_adv_perts.max(),
                    l0_norms_adv_perts.min()))
                print("Reporting measured L0 histogram for each L0 norm:")
                for l0_norm_idx, l0_norm in enumerate(self.l0_norms):
                    print("Histogram for perturbations computed under L0 norm: " + str(l0_norm))
                    print("Histogram bins:")
                    print(l0_norms_hist_l0_limits[l0_norm_idx])
                    print("Ratio of perturbations with measured L0 up to each bin value:")
                    print(l0_norms_hist_l0_ratio[l0_norm_idx])
                    print("robust accuracy for perturbations with measured L0 up to each bin value:")
                    print(l0_norms_hist_l0_robust_accuracy[l0_norm_idx])
            if self.attack_report_info:
                l0_norms_acc_steps = 1 - (all_succ.sum(dim=3) / n_examples)
                l0_norms_avg_loss_steps = all_loss.sum(dim=3) / n_examples
            else:
                l0_norms_acc_steps = None
                l0_norms_avg_loss_steps = None
            l0_norms_adv_x = x_orig.unsqueeze(0) + l0_norms_adv_perts
            del all_succ
            del all_loss
            del perts_l0_norms
            del l0_norms_hist_l0_ratio
            del l0_norms_hist_l0_robust_accuracy
            del l0_norms_adv_perts
            torch.cuda.empty_cache()

            return init_accuracy, x_adv, y_adv, l0_norms_adv_x, l0_norms_adv_y, \
                l0_norms_robust_accuracy, l0_norms_acc_steps, l0_norms_avg_loss_steps, l0_norms_perts_info, \
                adv_batch_compute_time_mean, adv_batch_compute_time_std, tot_adv_compute_time, tot_adv_compute_time_std
