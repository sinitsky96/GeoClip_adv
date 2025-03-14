import torch
import gc
from parser import get_args, save_img_tensors
from AdvRunner import AdvRunner


def run_adv_attacks(args):
    print(f'Running evaluation of adversarial attacks:')
    adv_runner = AdvRunner(args.model, args.attack_obj, args.l0_hist_limits, args.data_RGB_size,
                           device=args.device, dtype=args.dtype, verbose=True)
    print(f'Dataset: {args.dataset}, Model: {args.model_name},\n'
          f'Attack: {args.attack_name} with L0 sparsity={args.sparsity} and L_inf epsilon={args.eps_l_inf},\n'
          f'Attack iterations={args.n_iter} and restarts={args.n_restarts}')
    print("Shape of input samples:")
    print(args.data_shape)
    print("Data RGB range:")
    print(list(zip(args.data_RGB_start, args.data_RGB_end)))

    args.attack_obj.report_schematics()
    att_l0_norms = args.attack_obj.output_l0_norms
    att_report_info = args.attack_obj.report_info

    init_accuracy, x_adv, y_adv, l0_norms_adv_x, l0_norms_adv_y, \
        l0_norms_robust_accuracy, l0_norms_acc_steps, l0_norms_avg_loss_steps, l0_norms_perts_info, \
        adv_batch_compute_time_mean, adv_batch_compute_time_std, tot_adv_compute_time, tot_adv_compute_time_std = \
        adv_runner.run_standard_evaluation(args.x_test, args.y_test, args.n_examples, bs=args.batch_size)
    l0_norms_perts_max_l0, l0_norms_perts_min_l0, l0_norms_perts_mean_l0, l0_norms_perts_median_l0, l0_norms_perts_max_l_inf = l0_norms_perts_info

    l0_norms_adv_loss = []
    for l0_norm_idx, l0_norm in enumerate(att_l0_norms):
        if l0_norm < args.data_pixels:
            print("reporting results for sparse adversarial attack with L0 norm limitation:")
        else:
            print("reporting results for non-sparse adversarial attack (L0 norm limitation ineffective):")
        print(f'perturbations L0 norm limitation : {l0_norm}')
        print(f'perturbations L_inf norm limitation : {args.eps_l_inf}')
        print(f'robust accuracy: {l0_norms_robust_accuracy[l0_norm_idx]}')
        print(f'max L0 in perturbations: {l0_norms_perts_max_l0[l0_norm_idx]}')
        print(f'min L0 in perturbations: {l0_norms_perts_min_l0[l0_norm_idx]}')
        print(f'mean L0 in perturbation: {l0_norms_perts_mean_l0[l0_norm_idx]}')
        print(f'median L0 in perturbation: {l0_norms_perts_median_l0[l0_norm_idx]}')
        print(f'max L_inf in perturbations: {l0_norms_perts_max_l_inf[l0_norm_idx]}')
        if att_report_info:
            print(f'Model: {args.model_name}, robust accuracy after adversarial attack iterations: {l0_norms_acc_steps[l0_norm_idx].tolist()}')
            print(f'Model: {args.model_name}, loss after adversarial attack iterations: {l0_norms_avg_loss_steps[l0_norm_idx].tolist()}')
            l0_norms_adv_loss.append(l0_norms_avg_loss_steps[l0_norm_idx, -1, -1].item())

    print("Reporting results summary for L0 norms:")
    l0_norms_adv_succ_ratio = [(init_accuracy - adv_acc) / init_accuracy for adv_acc in l0_norms_robust_accuracy]
    adv_succ_ratio_to_l0 = [adv_succ / l0_norm for adv_succ, l0_norm in zip(l0_norms_adv_succ_ratio, att_l0_norms)]
    init_accuracy_percent = init_accuracy * 100
    print("L0 results:")
    print("Tested L0 norms:")
    print(att_l0_norms)
    print("attacked Model clean accuracy percentile:")
    print(init_accuracy_percent)
    print("l0_norms_robust_accuracy")
    print(l0_norms_robust_accuracy)
    print("l0_norms_adv_succ_ratio")
    print(l0_norms_adv_succ_ratio)
    print("adv_succ_ratio_to_l0")
    print(adv_succ_ratio_to_l0)
    print("l0_norms_perts_max_l0")
    print(l0_norms_perts_max_l0)
    print("l0_norms_perts_min_l0")
    print(l0_norms_perts_min_l0)
    print("l0_norms_perts_mean_l0")
    print(l0_norms_perts_mean_l0)
    print("l0_norms_perts_median_l0")
    print(l0_norms_perts_median_l0)
    print("l0_norms_perts_max_l_inf")
    print(l0_norms_perts_max_l_inf)
    
    print("Final sparse results:")
    print("attacked Model clean accuracy percentile:")
    print(init_accuracy_percent)
    print("Sparsity:")
    print(att_l0_norms[-1])
    print("sparse_robust_accuracy")
    print(l0_norms_robust_accuracy[-1])
    print("sparse_adv_succ_ratio")
    print(l0_norms_adv_succ_ratio[-1])
    print("adv_succ_ratio_to_sparsity")
    print(adv_succ_ratio_to_l0[-1])
    print("sparse_perts_max_l0")
    print(l0_norms_perts_max_l0[-1])
    print("sparse_perts_min_l0")
    print(l0_norms_perts_min_l0[-1])
    print("sparse_perts_mean_l0")
    print(l0_norms_perts_mean_l0[-1])
    print("sparse_perts_median_l0")
    print(l0_norms_perts_median_l0[-1])
    print("sparse_perts_max_l_inf")
    print(l0_norms_perts_max_l_inf[-1])

    print("Runtime Evaluation:")
    print("adv_batch_compute_time_mean")
    print(adv_batch_compute_time_mean)
    print("adv_batch_compute_time_std")
    print(adv_batch_compute_time_std)
    print("tot_adv_compute_time")
    print(tot_adv_compute_time)
    print("tot_adv_compute_time_std")
    print(tot_adv_compute_time_std)

    if args.save_results:
        save_path = args.adv_pert_save_path + '/adv_input.pt'
        print("saving adv inputs tensors to path:")
        print(save_path)
        torch.save(x_adv, args.adv_pert_save_path + '/adv_input.pt')
        for l0_norm_idx, l0_norm in enumerate(att_l0_norms):
            norm_str = str(int(l0_norm))
            save_path = args.adv_pert_save_path + '/adv_inputs_l0_norm_' + norm_str + '.pt'
            print("saving adv inputs tensors with L0 norm: " + norm_str + " to path:")
            print(save_path)
            torch.save(l0_norms_adv_x[l0_norm_idx], save_path)
        save_path = args.imgs_save_path + '/adv_inputs'
        print("saving adv inputs images to path:")
        print(save_path)
        save_img_tensors(save_path, x_adv, args.y_test, y_adv, args.labels_str_dict)
        for l0_norm_idx, l0_norm in enumerate(att_l0_norms):
            norm_str = str(int(l0_norm))
            save_path = args.imgs_save_path + '/adv_input_l0_norm_' + norm_str
            print("saving adv inputs images with L0 norm: " + norm_str + " to path:")
            print(save_path)
            save_img_tensors(save_path, l0_norms_adv_x[l0_norm_idx], args.y_test,
                             l0_norms_adv_y[l0_norm_idx], args.labels_str_dict)




    # imgs_save_path
    del init_accuracy
    del adv_runner
    del x_adv
    del l0_norms_adv_x
    del l0_norms_perts_max_l0
    del l0_norms_perts_min_l0
    del l0_norms_perts_mean_l0
    del l0_norms_perts_median_l0
    del l0_norms_perts_max_l_inf
    del l0_norms_robust_accuracy
    del l0_norms_acc_steps
    del l0_norms_avg_loss_steps
    del adv_batch_compute_time_mean
    del adv_batch_compute_time_std
    del tot_adv_compute_time
    del tot_adv_compute_time_std
    del l0_norms_adv_succ_ratio
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = get_args()
    run_adv_attacks(args)