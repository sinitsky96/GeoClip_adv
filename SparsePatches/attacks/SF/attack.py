from attacks.SF.sparsefool import sparsefool
from attacks.SF.utils import valid_bounds
import torch


class SF:
    def __init__(self, model, misc_args, attack_args):

        self.model = model

        self.device = misc_args['device']
        self.batch_size = misc_args['batch_size']
        self.data_shape = [self.batch_size] + misc_args['data_shape']
        self.data_channels = self.data_shape[1]
        self.data_w = self.data_shape[2]
        self.data_h = self.data_shape[3]
        self.n_data_pixels = self.data_w * self.data_h
        self.data_RGB_start = 0
        self.data_RGB_end = 1
        self.data_RGB_size = 1
        self.verbose = misc_args['verbose']
        self.report_info = False
        self.output_l0_norms = [self.n_data_pixels]
        self.n_restarts = 1
        self.report_info = False
        self.name = "SF"

        self.eps_ratio = attack_args['eps']
        self.eps = self.eps_ratio * self.data_RGB_size
        self.eps_from_255 = attack_args['eps_from_255']
        self.n_iter = attack_args['n_iter']
        self.lambda_factor = attack_args['lambda_factor']

    def report_schematics(self):
        print("Running SF attack based on the paper: \" SparseFool: a few pixels make a big difference \" ")
        print("Attack L_inf norm limitation:")
        print(self.eps)
        print("Number of iterations for perturbation optimization:")
        print(self.n_iter)

    def perturb(self, data, target, targeted=False):
        
        lb = (data - self.eps).clamp_(self.data_RGB_start, self.data_RGB_end)
        ub = (data + self.eps).clamp_(self.data_RGB_start, self.data_RGB_end)
        
        x_adv, r, pred_label, fool_label, loops = sparsefool(data, self.model, lb, ub, lambda_=self.lambda_factor, max_iter=self.n_iter, device=self.device)
        adv_pert = x_adv - data

        return adv_pert, None, None
