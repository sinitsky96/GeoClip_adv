import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import torch
from torch.nn import functional as F
from sparse_rs.util import haversine_distance, CONTINENT_R, STREET_R



class Attack:
    def __init__(self, model, criterion, misc_args, pgd_args, dropout_args=None):
        self.model = model
        self.criterion = criterion

        self.name = "PGD"
        self.device = misc_args['device']
        self.dtype = misc_args['dtype']
        self.batch_size = misc_args['batch_size']
        self.data_shape = [self.batch_size] + misc_args['data_shape']
        self.data_channels = self.data_shape[1]
        self.data_w = self.data_shape[2]
        self.data_h = self.data_shape[3]
        self.n_data_pixels = self.data_w * self.data_h
        self.data_RGB_start = (torch.tensor(misc_args['data_RGB_start'], device=self.device).
                                 unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self.data_RGB_end = (torch.tensor(misc_args['data_RGB_end'], device=self.device).
                               unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self.data_RGB_size = (torch.tensor(misc_args['data_RGB_size'], device=self.device).
                               unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self.mask_shape = [self.batch_size, 1, self.data_w, self.data_h]
        self.mask_shape_flat = [self.batch_size, self.n_data_pixels]
        self.mask_zeros_flat = torch.zeros(self.mask_shape_flat, dtype=torch.bool, device=self.device,
                                           requires_grad=False)
        self.mask_ones_flat = torch.ones(self.mask_shape_flat, dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.verbose = misc_args['verbose']
        self.report_info = misc_args['report_info']

        self.norm = pgd_args['norm']
        self.p = float(self.norm[1:])
        self.pert_lb = None
        self.pert_ub = None
        self.eps_ratio = pgd_args['eps']
        self.eps = self.eps_ratio * self.data_RGB_size
        self.n_restarts = pgd_args['n_restarts']
        self.n_iter = pgd_args['n_iter']
        self.alpha = pgd_args['alpha'] * self.data_RGB_size
        if self.alpha is None:
            self.alpha = self.eps / self.n_iter
        self.a_abs = self.alpha.abs()
        self.rand_init = pgd_args['rand_init']
        
        # Initialize targeted attribute (will be set by AdvRunner)
        self.targeted = False
        
        self.targeted_mul = None
        self.multiplier = None
        self.eval_pert = None
        self.clean_loss = None
        self.clean_succ = None

        self.active_dpo_mean = 0
        self.dpo = self.no_dpo
        self.active_dpo = self.no_dpo
        self.apply_dpo = False
        self.set_dpo_dist = self.set_dpo_dist_none
        self.set_dpo = self.set_dpo_none
        self.compute_dpo_std = self.compute_dpo_std_const
        self.dropout_str = "no dropout"
        self.dpo_dist = None
        self.dpo_mean = None
        self.dpo_std = None
        if dropout_args is not None:
            self.compute_dpo_args(dropout_args)

    def compute_dpo_args(self, dropout_args):
        dpo_mean = dropout_args['dropout_mean']
        dpo_dist_str = dropout_args['dropout_dist']
        if dpo_mean == 0 or dpo_dist_str == "none":
            self.dpo_mean = torch.tensor(0)
            self.dpo_std = torch.tensor(0)
            self.dropout_str = "no dropout"
            return
        self.set_dpo = self.set_dpo_active
        self.dpo_mean = torch.tensor(dpo_mean)
        self.dpo_std = torch.tensor(dropout_args['dropout_std'])
        self.active_dpo = self.pixel_dpo
        self.apply_dpo = True
        if dpo_dist_str == "gauss":
            self.set_dpo_dist = self.set_dpo_dist_gauss
            if dropout_args['dropout_std_bernoulli']:
                self.compute_dpo_std = self.compute_dpo_std_as_bernoulli
                self.dropout_str = ("multiplicative gaussian dropout with mean=" + str(self.dpo_mean.item()) +
                                    " and std as in bernoulli distribution")
            else:
                self.dropout_str = ("multiplicative gaussian dropout with mean=" + str(self.dpo_mean.item()) +
                                    " and std=" + str(self.dpo_std.item()))

        elif dpo_dist_str == "cbernoulli":
            self.set_dpo_dist = self.set_dpo_dist_continuous_bernoulli
            self.dropout_str = ("multiplicative continuous bernoulli dropout with p=" + str(self.dpo_mean.item()))

        else:
            self.set_dpo_dist = self.set_dpo_dist_bernoulli
            self.dropout_str = ("multiplicative bernoulli dropout with p=" + str(self.dpo_mean.item()))

    def set_params(self, x, targeted):
        self.set_batch_size(x.shape[0])
        self.set_multiplier(targeted)
        self.pert_lb = torch.maximum(self.data_RGB_start - x, -self.eps)
        self.pert_ub = torch.minimum(self.data_RGB_end - x, self.eps)

    def set_batch_size(self, batch_size):
        if self.batch_size == batch_size:
            return
        self.batch_size = batch_size
        self.data_shape[0] = self.batch_size
        self.mask_shape[0] = self.batch_size
        self.mask_shape_flat[0] = self.batch_size
        self.mask_zeros_flat = torch.zeros(self.mask_shape_flat, dtype=torch.bool, device=self.device,
                                           requires_grad=False)
        self.mask_ones_flat = torch.ones(self.mask_shape_flat, dtype=torch.bool, device=self.device,
                                         requires_grad=False)

    def set_multiplier(self, targeted):
        if targeted:
            self.targeted_mul = -1
            self.eval_pert = self.eval_pert_targeted
        else:
            self.targeted_mul = 1
            self.eval_pert = self.eval_pert_untargeted
        self.multiplier = (self.targeted_mul * self.a_abs).to(self.device)

    def project(self, perturbation):
        return perturbation.clamp_(self.pert_lb, self.pert_ub)

    def step(self, pert, grad):
        grad = self.normalize_grad(grad)
        pert += self.multiplier * grad
        return self.project(pert)

    def random_initialization(self):
        rand = torch.empty(self.data_shape, dtype=self.dtype, device=self.device).uniform_(-1, 1) * self.eps
        return self.project(rand)
    
    def normalize_grad(self, grad):
        return grad.sign()
    
    def test_pert(self, x, y, pert):
        """Test a perturbation by getting model predictions and computing loss"""
        # Get predictions from GeoClip
        output, _ = self.model.predict_from_tensor(x + pert)
        
        # Calculate distances between predicted and target coordinates
        distances = haversine_distance(output, y)
        
        # For untargeted attacks, we want to maximize distance (negative for gradient ascent)
        # For targeted attacks, we want to minimize distance
        # We use a margin-based loss that encourages distances to be either above CONTINENT_R (untargeted)
        # or below STREET_R (targeted)
        if not self.targeted:
            # For untargeted attacks, we want distances > CONTINENT_R
            # loss = max(0, CONTINENT_R - distance) per example
            margin = F.relu(torch.tensor(CONTINENT_R, device=distances.device) - distances)
            loss = -margin  # Negative because we want to maximize distance, keep per-example values
        else:
            # For targeted attacks, we want distances < STREET_R
            # loss = max(0, distance - STREET_R) per example
            margin = F.relu(distances - torch.tensor(STREET_R, device=distances.device))
            loss = margin  # Keep per-example values
        
        return output, loss

    def eval_pert_untargeted(self, x, y, pert):
        """Evaluate perturbation for untargeted attack (maximize distance)"""
        with torch.no_grad():
            output, loss = self.test_pert(x, y, pert)
            
            # Calculate distances between predicted and true coordinates
            distances = haversine_distance(output, y)
            
            # Success = prediction far from ground truth
            succ = distances > CONTINENT_R
            return loss, succ

    def eval_pert_targeted(self, x, y, pert):
        """Evaluate perturbation for targeted attack (minimize distance)"""
        with torch.no_grad():
            output, loss = self.test_pert(x, y, pert)
            
            # Calculate distances between predicted and target coordinates
            distances = haversine_distance(output, y)
            
            # Success = prediction close to target
            succ = distances <= STREET_R
            return loss, succ

    def update_best(self, best_crit, new_crit, best_ls, new_ls):
        with torch.no_grad():
            improve = new_crit.lt(best_crit)
            best_crit[improve] = new_crit[improve]
            for best, new in zip(best_ls, new_ls):
                if best.dim() > new.dim():
                    # Reshape new tensor to match best tensor dimensions
                    new_shape = [-1] + [1] * (best.dim() - new.dim())
                    new_reshaped = new.view(*new_shape)
                    # Reshape improve tensor to match broadcasting dimensions
                    improve_shape = list(improve.shape) + [1] * (best.dim() - improve.dim())
                    improve_reshaped = improve.view(*improve_shape)
                    best_updated = torch.where(improve_reshaped, new_reshaped, best)
                else:
                    # Reshape improve tensor to match the target dimensions
                    improve_expanded = improve
                    for _ in range(best.dim() - improve.dim()):
                        improve_expanded = improve_expanded.unsqueeze(-1)
                    best_updated = torch.where(improve_expanded, new.detach(), best.detach())
                best.copy_(best_updated)

    def no_dpo(self, pert):
        return pert

    def pixel_dpo(self, pert):
        return self.dpo_dist.sample() * pert

    def compute_dpo_std_const(self, dpo_mean):
        return [self.dpo_std] * len(dpo_mean)

    def compute_dpo_std_as_bernoulli(self, dpo_mean):
        std = torch.zeros_like(dpo_mean)
        var = (dpo_mean * (1 - dpo_mean))
        nonzero = var.nonzero()
        if len(nonzero):
            std[nonzero] = var[nonzero].sqrt()
        return std

    def set_dpo_dist_none(self, dpo_mean, dpo_std):
        self.dpo_dist = None

    def set_dpo_dist_bernoulli(self, dpo_mean, dpo_std):
        self.dpo_dist = torch.distributions.Bernoulli(
            probs=torch.full(self.mask_shape, dpo_mean, dtype=self.dtype, device=self.device))

    def set_dpo_dist_continuous_bernoulli(self, dpo_mean, dpo_std):
        self.dpo_dist = torch.distributions.ContinuousBernoulli(
            probs=torch.full(self.mask_shape, dpo_mean, dtype=self.dtype, device=self.device))

    def set_dpo_dist_gauss(self, dpo_mean, dpo_std):
        self.dpo_dist = torch.distributions.Normal(
            loc=torch.full(self.mask_shape, dpo_mean, dtype=self.dtype, device=self.device),
            scale=torch.full(self.mask_shape, dpo_std, dtype=self.dtype, device=self.device))

    def set_dpo_none(self, dpo_mean, dpo_std):
        return

    def set_dpo_active(self, dpo_mean, dpo_std):
        if dpo_mean:
            self.dpo = self.active_dpo
            if self.active_dpo_mean != dpo_mean:
                self.active_dpo_mean = dpo_mean
                self.set_dpo_dist(dpo_mean, dpo_std)
        else:
            self.dpo = self.no_dpo
        return

    def perturb(self, x, y, targeted=False):
        raise NotImplementedError('You need to define a perturb method!')