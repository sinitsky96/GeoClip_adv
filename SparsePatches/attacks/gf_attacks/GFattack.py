import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import attacks.gf_attacks.generators as generators


# cudnn.benchmark = True


class GFattack:
    def __init__(self, model, misc_args, attack_args):

        self.model = model
        self.criterion = self.CWLoss

        self.device = misc_args['device']
        self.batch_size = 1
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
        self.name = "GF"

        self.gen_distort = attack_args['gen_distort']
        if self.gen_distort:
            self.data_RGB_start = -1
            self.data_RGB_end = 1
            self.data_RGB_size = 2
        self.eps_ratio = attack_args['eps']
        self.eps = self.eps_ratio * self.data_RGB_size
        self.eps_from_255 = attack_args['eps_from_255']
        self.n_iter = attack_args['n_iter']
        self.n_reduce_iter = attack_args['n_reduce_iter']

        pool_kernel = 3
        self.Avg_pool = nn.AvgPool2d(pool_kernel, stride=1, padding=int(pool_kernel / 2))
        self.SIZE = self.n_data_pixels
        self.boost = (self.eps_from_255 < 128)
        step = int(max(int(self.eps_from_255 / 10.), 1))
        a = [i for i in range(0, int(self.eps_from_255 + step), step)]
        self.search_num = len(a)
        self.a = np.asarray(a) * self.data_RGB_size / 255.

        if self.gen_distort:
            ##### Generator loading for distortion map
            self.netG = generators.Res_ResnetGenerator(3, 1, 16, norm_type='batch', act_type='relu')
            self.netG = torch.nn.DataParallel(self.netG, device_ids=[self.device])
            self.netG.load_state_dict(torch.load('./attacks/gf_attacks/pretrain/G_imagenet.pth', map_location=self.device))
            self.netG.cuda()
            self.netG.eval()
            self.image_hill = self.image_hill_generator
        else:
            ##### use identity as the distortion map
            self.image_hill = self.image_hill_identity

    def report_schematics(self):
        print("Running GF attack based on the paper: \" GreedyFool: Distortion-Aware Sparse Adversarial Attack \" ")
        print("Attack L_inf norm limitation:")
        print(self.eps)
        print("Number of iterations for perturbation optimization:")
        print(self.n_iter)
        print("Number of iterations for L0 norm reduction:")
        print(self.n_reduce_iter)

    def image_hill_identity(self, real_A):
        return 1

    def image_hill_generator(self, real_A):
        self.netG.eval()
        image_hill = self.netG(real_A * 0.5 + 0.5) * 0.5 + 0.5
        pre_hill = 1 - image_hill
        pre_hill = pre_hill.view(1, 1, -1)

        np_hill = pre_hill.detach().cpu().numpy()
        percen = np.percentile(np_hill, 30)
        pre_hill = torch.max(pre_hill - percen, torch.zeros(pre_hill.size()).cuda())
        np_hill = pre_hill.detach().cpu().numpy()
        percen = np.percentile(np_hill, 75)
        pre_hill /= percen
        pre_hill = torch.clamp(pre_hill, 0, 1)
        pre_hill = self.Avg_pool(pre_hill)
        return pre_hill


    def clip(self, adv_A, real_A, eps):
        g_x = real_A - adv_A
        clip_gx = torch.clamp(g_x, min=-eps, max=eps)
        adv_x = real_A - clip_gx
        return adv_x

    def CWLoss(self, logits, target, kappa=0, tar=True):
        target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
        target_one_hot = Variable(torch.eye(1000).type(torch.cuda.FloatTensor)[target.long()].cuda())
        real = torch.sum(target_one_hot * logits, 1)
        other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10000), 1)[0]
        kappa = torch.zeros_like(other).fill_(kappa)

        if tar:
            return torch.sum(torch.max(other - real, kappa))
        else:
            return torch.sum(torch.max(real - other, kappa))

    def perturb(self, data, target, targeted=False):
        input_A = data.cuda(non_blocking=True)
        real_A = Variable(input_A, requires_grad=False)

        pre_hill = self.image_hill(real_A)

        logist_B = self.model(real_A)
        _, target = torch.max(logist_B, 1)
        adv = real_A
        ini_num = 1
        grad_num = ini_num
        mask = torch.zeros(1, 3, self.SIZE).cuda()

        ##### Increasing
        for iters in range(self.n_iter):
            temp_A = Variable(adv.data, requires_grad=True)
            logist_B = self.model(temp_A)
            _, pre = torch.max(logist_B, 1)

            if target.cpu().data.float() != pre.cpu().data.float():
                break
            Loss = self.criterion(logist_B, target, -100, False) / real_A.size(0)

            self.model.zero_grad()
            if temp_A.grad is not None:
                temp_A.grad.data.fill_(0)
            Loss.backward()

            grad = temp_A.grad
            abs_grad = torch.abs(grad).view(1, 3, -1).mean(1, keepdim=True)
            abs_grad = abs_grad * pre_hill
            if not self.boost:
                abs_grad = abs_grad * (1 - mask)
            _, grad_sort_idx = torch.sort(abs_grad)
            grad_sort_idx = grad_sort_idx.view(-1)
            grad_idx = grad_sort_idx[-grad_num:]
            mask[0, :, grad_idx] = 1.
            temp_mask = mask.view(1, 3, self.data_w, self.data_h)
            grad = temp_mask * grad

            abs_grad = torch.abs(grad)
            abs_grad = abs_grad / torch.max(abs_grad)
            normalized_grad = abs_grad * grad.sign()
            scaled_grad = normalized_grad.mul(self.eps_ratio)
            temp_A = temp_A - scaled_grad
            temp_A = self.clip(temp_A, real_A, self.eps)
            adv = torch.clamp(temp_A, self.data_RGB_start, self.data_RGB_end)
            if self.boost:
                grad_num += ini_num

        final_adv = adv
        adv_noise = real_A - final_adv
        adv = final_adv

        abs_noise = torch.abs(adv_noise).view(1, 3, -1).mean(1, keepdim=True)
        temp_mask = abs_noise != 0
        modi_num = torch.sum(temp_mask).data.clone().item()

        reduce_num = modi_num
        reduce_count = 0
        ###### Reducing
        if modi_num > 2:
            reduce_idx = 0
            while reduce_idx < reduce_num and reduce_count < self.n_reduce_iter:
                reduce_count += 1
                adv_noise = real_A - adv

                abs_noise = torch.abs(adv_noise).view(1, 3, -1).mean(1, keepdim=True)
                reduce_mask = abs_noise != 0
                reduce_mask = reduce_mask.repeat(1, 3, 1).float()
                abs_noise[abs_noise == 0] = 3.

                reduce_num = torch.sum(reduce_mask).data.clone().item() / 3
                if reduce_num == 1:
                    break

                noise_show, noise_sort_idx = torch.sort(abs_noise)
                noise_sort_idx = noise_sort_idx.view(-1)

                noise_idx = noise_sort_idx[reduce_idx]
                reduce_mask[0, :, noise_idx] = 0.
                temp_mask = reduce_mask.view(1, 3, self.data_w, self.data_h)
                noise = temp_mask * adv_noise

                abs_noise = torch.abs(noise)
                abs_noise = abs_noise / torch.max(abs_noise)
                normalized_grad = abs_noise * noise.sign()

                with torch.no_grad():
                    self.model.eval()
                    ex_temp_eps = torch.from_numpy(self.a).view(-1, 1, 1, 1).float().cuda()
                    ex_normalized_grad = normalized_grad.repeat(self.search_num, 1, 1, 1)
                    ex_scaled_grad = ex_normalized_grad.mul(ex_temp_eps)
                    ex_real_A = real_A.repeat(self.search_num, 1, 1, 1)
                    ex_temp_A = ex_real_A - ex_scaled_grad
                    ex_temp_A = self.clip(ex_temp_A, ex_real_A, self.eps)
                    ex_adv = torch.clamp(ex_temp_A, self.data_RGB_start, self.data_RGB_end)
                    ex_temp_A = Variable(ex_adv.data, requires_grad=False)
                    ex_logist_B = self.model(ex_temp_A)
                    _, pre = torch.max(ex_logist_B, 1)
                    comp = torch.eq(target.cpu().data.float(), pre.cpu().data.float())
                    top1 = torch.sum(comp).float() / pre.size(0)
                    if top1 != 1:  ##### exists at least one adversarial sample
                        found = False
                        for i in range(self.search_num):
                            if comp[i] == 0:
                                temp_adv = ex_temp_A[i:i + 1]
                                logist_B = self.model(temp_adv)
                                _, pre = torch.max(logist_B, 1)
                                new_comp = torch.eq(target.cpu().data.float(), pre.cpu().data.float())
                                if torch.sum(new_comp) != 0:
                                    continue
                                found = True
                                adv = temp_adv
                                break
                        if found == False:
                            reduce_idx += 1
                    else:
                        reduce_idx += 1

        adv_noise = adv - real_A

        return adv_noise, None, None
