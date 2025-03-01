import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np


class Attack:
    """
    Base class for all attacks.
    """
    def __init__(self, model, criterion, misc_args, attack_args):
        """
        Initialize the attack.
        
        Args:
            model: The model to attack
            criterion: The loss function to optimize
            misc_args: Dictionary containing miscellaneous arguments
            attack_args: Dictionary containing attack-specific arguments
        """
        self.model = model
        self.criterion = criterion
        
        # Set device and data type
        self.device = misc_args.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dtype = misc_args.get('dtype', torch.float32)
        
        # Set batch size and data shape
        self.batch_size = misc_args.get('batch_size', 1)
        self.data_shape = misc_args.get('data_shape', [3, 224, 224])
        
        # Set data range
        self.data_RGB_start = misc_args.get('data_RGB_start', [-2.0, -2.0, -2.0])
        self.data_RGB_end = misc_args.get('data_RGB_end', [2.0, 2.0, 2.0])
        self.data_RGB_size = misc_args.get('data_RGB_size', [4.0, 4.0, 4.0])
        
        # Set verbose and report info
        self.verbose = misc_args.get('verbose', False)
        self.report_info = misc_args.get('report_info', False)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Calculate derived values
        self.data_channels = self.data_shape[0]
        self.data_w = self.data_shape[1]
        self.data_h = self.data_shape[2]
        self.n_data_pixels = self.data_w * self.data_h
        
        self.name = "PGD"
        self.data_channels = self.data_shape[0]
        self.data_w = self.data_shape[1]
        self.data_h = self.data_shape[2]
        self.n_data_pixels = self.data_w * self.data_h
        self.data_RGB_start = (torch.tensor(self.data_RGB_start, device=self.device).
                                 unsqueeze(0).unsqueeze(2).unsqueeze(3))
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
        self.verbose = misc_args['verbose']
        self.report_info = misc_args['report_info']

        self.norm = attack_args['norm']
        self.p = float(self.norm[1:])
        self.pert_lb = None
        self.pert_ub = None
        self.eps_ratio = attack_args['eps']
        self.eps = self.eps_ratio * self.data_RGB_size
        self.n_restarts = attack_args['n_restarts']
        self.n_iter = attack_args['n_iter']
        self.alpha = attack_args['alpha'] * self.data_RGB_size
        if self.alpha is None:
            self.alpha = self.eps / self.n_iter
        self.a_abs = self.alpha.abs()
        self.rand_init = attack_args['rand_init']
        self.targeted_mul = None
        self.multiplier = None
        self.eval_pert = None
        self.clean_loss = None
        self.clean_succ = None

    def set_params(self, x, targeted):
        self.batch_size = x.shape[0]
        self.data_shape[0] = x.shape[0]
        self.set_multiplier(targeted)
        self.pert_lb = self.data_RGB_start - x
        self.pert_ub = self.data_RGB_end - x

    def set_multiplier(self, targeted):
        if targeted:
            self.targeted_mul = -1
            self.eval_pert = self.eval_pert_targeted
        else:
            self.targeted_mul = 1
            self.eval_pert = self.eval_pert_untargeted
        self.multiplier = (self.targeted_mul * self.a_abs).to(self.device)

    def random_initialization(self):
        if self.norm == 'Linf':
            return torch.empty(self.data_shape, dtype=self.dtype, device=self.device).uniform_(-1, 1) * self.eps
        else:
            return torch.empty(self.data_shape, dtype=self.dtype, device=self.device).normal_(0, self.eps * self.eps)

    def project(self, perturbation):
        if self.norm == 'Linf':
            pert = torch.clamp(perturbation, -self.eps, self.eps)
        else:
            pert = F.normalize(perturbation.view(perturbation.shape[0], -1),
                               p=self.p, dim=-1).view(perturbation.shape) * self.eps
        pert.clamp_(self.pert_lb, self.pert_ub)
        return pert

    def normalize_grad(self, grad):
        if self.norm == 'Linf':
            return grad.sign()
        else:
            return F.normalize(grad.view(grad.shape[0], -1), p=self.p, dim=-1).view(grad.shape)

    def step(self, pert, grad):
        grad = self.normalize_grad(grad)
        pert += self.multiplier * grad
        return self.project(pert)

    def test_pert(self, x, y, pert):
        with torch.no_grad():
            output = self.model.forward(x + pert)
            loss = self.targeted_mul * self.criterion(output, y)
            return output, loss

    def eval_pert_untargeted(self, x, y, pert):
        with torch.no_grad():
            output, loss = self.test_pert(x, y, pert)
            # For GeoCLIP, we need to adapt this to the specific task
            # This is a placeholder and should be modified based on the specific task
            succ = torch.argmax(output, dim=1) != y
            return loss, succ

    def eval_pert_targeted(self, x, y, pert):
        with torch.no_grad():
            output, loss = self.test_pert(x, y, pert)
            # For GeoCLIP, we need to adapt this to the specific task
            # This is a placeholder and should be modified based on the specific task
            succ = torch.argmax(output, dim=1) == y
            return loss, succ

    def update_best(self, best_crit, new_crit, best_ls, new_ls):
        improve = new_crit > best_crit
        best_crit[improve] = new_crit[improve]
        for idx, best in enumerate(best_ls):
            new = new_ls[idx]
            best[improve] = new[improve]

    def perturb(self, X, y):
        """
        Perturb the input X to maximize the loss.
        
        Args:
            X: Input data
            y: Target data
            
        Returns:
            Perturbed input
        """
        raise NotImplementedError("Subclasses must implement perturb")

    def report_schematics(self):
        """
        Print information about the attack.
        """
        raise NotImplementedError("Subclasses must implement report_schematics") 