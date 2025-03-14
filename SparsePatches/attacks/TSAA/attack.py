import torch
import torch.nn.functional as F

from attacks.TSAA.generators import GeneratorResnet
import time


class TSAA:
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
        self.n_iter = 1
        self.report_info = False
        self.name = "TSAA"

        self.eps_ratio = attack_args['eps']
        self.eps = self.eps_ratio * self.data_RGB_size
        self.eps_from_255 = attack_args['eps_from_255']
        self.is_inception_model = attack_args['is_inception_model']
        self.model_type = 'res50'
        if self.is_inception_model:
            self.model_type = 'incv3'

        self.checkpoint = ('./attacks/TSAA/pretrain/netG_-1_' + self.model_type +
                           '_imagenet_eps' + str(self.eps_from_255) + '.pth')
        self.netG = GeneratorResnet(inception=self.is_inception_model, eps=self.eps, evaluate=True)
        self.netG.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.netG.to(self.device)
        self.netG.eval()
        
        if self.is_inception_model:
            self.trans = self.trans_incep
        else:
            self.trans = self.trans_identity
    
    def report_schematics(self):
        print("Running TSAA attack based on the paper: \" Transferable Sparse Adversarial Attack \" ")
        print("The attack will generate an adversarial perturbation via a pretrained GAN model")
        print("Path to pretrained GAN model:")
        print(self.checkpoint)

    
    def trans_identity(self, x):
        return x

    def trans_incep(self, x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return x
    
    def perturb(self, data, target, targeted=False):
        # Adversary
        adv, _, adv_0, adv_00 = self.netG(data)
        adv_trans = self.trans(adv.clone().detach())
        adv_pert = adv_trans - data

        return adv_pert, None, None
