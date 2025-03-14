import torch
from attacks.pgd_attacks.attack import Attack


class PGD(Attack):
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
        super(PGD, self).__init__(model, criterion, misc_args, pgd_args)

    def perturb(self, x, y, targeted=False):
        self.set_params(x, targeted)
        self.mask = torch.ones_like(x)

        if self.report_info:
            all_best_succ = torch.zeros(self.n_restarts * (self.n_iter + 1), self.batch_size, dtype=torch.bool, device=self.device)
            all_best_loss = torch.zeros(self.n_restarts * (self.n_iter + 1), self.batch_size, device=self.device)
        else:
            all_best_succ = None
            all_best_loss = None

        best_pert = torch.zeros_like(x)
        clean_output = self.model.forward(x)
        best_loss = self.criterion(clean_output, y).detach()
        best_ps = torch.ones_like(y, dtype=self.dtype, device=self.device)

        self.model.eval()
        for rest in range(self.n_restarts):
            pert = torch.zeros_like(x, requires_grad=True)

            if self.rand_init:
                pert = self.random_initialization()

            pert = self.project(pert, self.mask)
            loss, ps, succ = self.eval_pert(x, y, targeted, pert)
            self.update_best(best_loss, loss,
                             [best_pert, best_ps],
                             [pert, ps])
            if self.report_info:
                curr_iter = rest * (self.n_iter + 1)
                all_best_succ[curr_iter] = succ | all_best_succ[curr_iter - 1]
                all_best_loss[curr_iter] = best_loss

            for k in range(self.n_iter):
                pert.requires_grad_()
                x_i = x + pert
                oi = self.model.forward(x_i)
                train_loss = self.criterion(oi, y)
                grad = torch.autograd.grad(train_loss.mean(), [pert])[0].detach()

                with torch.no_grad():
                    grad = self.normalize_grad(grad)
                    pert += self.multiplier * grad
                    pert = self.project(pert, self.mask)

                eval_loss, ps, succ = self.eval_pert(x, y, targeted, pert)
                self.update_best(best_loss, eval_loss,
                                 [best_pert, best_ps],
                                 [pert, ps])
                if self.report_info:
                    curr_iter = rest * (self.n_iter + 1) + k + 1
                    all_best_succ[curr_iter] = succ | all_best_succ[curr_iter - 1]
                    all_best_loss[curr_iter] = best_loss

        best_pert.detach()
        adv_pert = best_pert[-1].clone().detach().unsqueeze(0)

        return adv_pert, all_best_succ, all_best_loss
