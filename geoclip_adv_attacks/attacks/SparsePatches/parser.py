import torch
import torch.backends.cudnn as cudnn
import argparse
from os import mkdir
from os.path import isdir
from torchvision.utils import save_image
from robustbench.utils import load_model

import PGDTrim, PGDTrimKernel


def parse_args():
    parser = argparse.ArgumentParser()
    # run args
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: random)')
    parser.add_argument('--gpus', default='0', help='List of GPUs used - e.g 0,1,3')
    parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
    parser.add_argument('--attack_verbose', action='store_true')
    # data args
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, imagenet')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--imagenet_size', type=int, default=224, help='imagenet data will be transformed to size X size (default: 224)')
    parser.add_argument('--imagenet_resize', type=int, default=256, help='imagenet data will first be resized to resize X resize (default: 256)')
    parser.add_argument('--imagenet_resize_bicubic', action='store_true', help='imagenet resize will be bicubic (default: False)')
    parser.add_argument('--normalize_data', action='store_true', help='normalize mean and std in data')
    parser.add_argument('--n_examples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--report_info', action='store_true', help='additional info and non final results will be reported as well')
    parser.add_argument('--save_results', action='store_true',
                        help='save the produced results')
    parser.add_argument('--l0_hist_limits', type=int, nargs='+',
                        default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 224, 256, 299, 512, 1024],
                        help='Limits of L0 values for processing the L0 histogram of the resulting perturbations')
    # [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    # model args
    parser.add_argument('--model_name', type=str, default='', help='model name to load from robustness (default: use pretrained ResNet18 model)')
    parser.add_argument('--model_transform_input', action='store_true', help='apply model specific data transformation')

    # attack args
    # pgd attacks args
    parser.add_argument('--attack', type=str, default='PGDTrim', help='PGDTrim, CS, PGD_L0, SF, GF, TSAA, Homotopy')
    parser.add_argument('--eps_l_inf_from_255', type=int, default=255)
    parser.add_argument('--sparsity', type=int, default=1, help='number of pixels in sparse pgd_attacks')
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--n_restarts', type=int, default=1, help='number of restart iterations for pgd_attacks')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--att_init_zeros', action='store_true', help='initialize the adversarial pertubation to zeroes (default: random initialization)')
    
    # PGDTrim dropout args
    parser.add_argument('--att_dpo_dist', type=str, default='bernoulli', help='distribution for dropout sampling in pgd_attacks, options: none, bernoulli, cbernoulli, gauss')
    parser.add_argument('--att_dpo_mean', type=float, default=1, help='mean for the dropout distribution used in pgd_attacks (default: 1)')
    parser.add_argument('--att_dpo_std', type=float, default=1, help='standard deviation for the dropout distribution used in pgd_attacks, if not determined otherwise according to the mean (default: 1)')
    # PGDTrim trim args
    parser.add_argument('--att_trim_steps', type=int, nargs='+',
                        default=None,
                        help='list of L0 values for trimming the perturbation pixels, '
                             'the returned perturbation will have sparsity values equal to the last entry in the list')
    parser.add_argument('--att_max_trim_steps', type=int, default=5, help='limit the number of pixel trimming iterations')
    parser.add_argument('--att_trim_steps_reduce', type=str, default='even', help='Policy for reducing trim steps over multiple restarts, Options: none, even, best')
    parser.add_argument('--att_const_dpo_mean', action='store_true', help='do not scale the dropout mean during the trimming process by the trim ratio (default: False)')
    parser.add_argument('--att_const_dpo_std', action='store_true', help='do not scale the standard deviation of the dropout distribution as in bernoulli distribution with same mean (default: False)')
    parser.add_argument('--att_post_trim_dpo', action='store_true', help='apply the dropout after the trimming process as well (default: False)')
    parser.add_argument('--att_dynamic_trim', action='store_true', help='consider pixels wise pertubations from previous restarts in trim (default: False)')

    # PGDTrim mask args
    parser.add_argument('--att_mask_dist', type=str, default='multinomial', help='distribution for sampling binary masks. options: topk, multinomial, bernoulli, cbernoulli')
    parser.add_argument('--att_mask_prob_amp_rate', type=int, default=0, help='the probability of sampling a pixel will be increased according to it\'s amplitude at this rate (default: 0, uniform probability)')
    parser.add_argument('--att_norm_mask_amp', action='store_true', help='normalize the amplitude of the sampled masks by their number of active pixels (default: False)')
    parser.add_argument('--att_mask_opt_iter', type=int, default=0, help='Number of iterations to optimize each mask sample before evaluation (default: 0, no optimization)')
    parser.add_argument('--att_n_mask_samples', type=int, default=1000, help='number of dropout samples to take into account for estimating the per-pixel criterion (default: 1000)')
    parser.add_argument('--att_no_samples_limit', action='store_true', help='do not limit the number of masks samples (default: limit to sampling all masks once)')
    parser.add_argument('--att_trim_best_mask', type=str, default='none', help='when all masks are sampled, trim pixels to the best mask, Options: none (default), in_final, all')

    # PGDTrim kernel args
    parser.add_argument('--att_kernel_size', type=int, default=1, help='square kernel size for structured perturbation trimming (default: 1X1)')
    # parser.add_argument('--att_kernel_dpo', action='store_true', help='Group pixels dpo according to the kernel, defult: False')
    parser.add_argument('--att_kernel_min_active', action='store_true', help='Consider only fully activated kernel patches when sampling masks, defult: False')
    parser.add_argument('--att_kernel_group', action='store_true', help='Group pixels in the mask according to kernel, defult: False, Trim the perturbation according to kernel structure')
    
    args = parser.parse_args()
    print("args")
    print(args)
    return args


def compute_run_args(args):
    if args.gpus is not None and not args.force_cpu and torch.cuda.is_available():
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.enabled = True
        cudnn.benchmark = True
        args.device = torch.device('cuda:' + str(args.gpus[0]))
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpus = []
        args.device = torch.device('cpu')
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    print('Running inference on device \"{}\"'.format(args.device))
    return args


def compute_data_args(args):
    if args.n_examples < args.batch_size:
        args.batch_size = args.n_examples
            
    args.data_RGB_start = [0, 0, 0]
    args.data_RGB_end = [1, 1, 1]
    args.data_RGB_offset = [0, 0, 0]
    args.data_RGB_size = [1, 1, 1]
    args.model_transform_input = True

    args.n_examples = args.x_test.shape[0]
    args.data_channels = args.x_test.shape[1]
    args.data_shape = list(args.x_test.shape)[1:]
    args.data_pixels = args.data_shape[1] * args.data_shape[2]
    args.dtype = args.x_test.dtype
    return args


def compute_models_args(args):

    if len(args.model_name):
        args.model = load_model(args.model_name, dataset=args.dataset, threat_model='Linf').to(args.device)
    elif args.dataset == "imagenet":
        args.model_name = 'inception_v3'
        args.model = inception_v3(pretrained=False, transform_input=args.model_transform_input)
        args.model.load_state_dict(torch.load('models/imagenet/inception_v3_google-1a9a5a14.pth'))
        args.model = args.model.to(args.device)
    elif args.dataset == "cifar100":
        args.model_name = 'Hendrycks2019Using'
        args.model = load_model(args.model_name, dataset=args.dataset, threat_model='Linf').to(args.device)
    else:
        #cifar 10
        args.model_name = 'ResNet18'
        args.model = ResNet18(args.device)
        args.model.load_state_dict(torch.load('models/cifar10/resnet18.pt', map_location=args.device))

    args.model.eval()

    return args


def compute_attack_args(args):
    args.criterion = torch.nn.CrossEntropyLoss(reduction='none')
    args.eps_l_inf = args.eps_l_inf_from_255 / 255
    args.type_attack = 'L0'
    args.l0_attacks_eps = -1
    if args.eps_l_inf < 1:
        args.type_attack = 'L0+Linf'
        args.l0_attacks_eps = args.eps_l_inf
    args.att_rand_init = not args.att_init_zeros
    args.att_sample_all_masks = not args.att_no_samples_limit
    args.att_scale_dpo_mean = not args.att_const_dpo_std
    args.att_dpo_std_bernoulli = not args.att_const_dpo_mean
    att_trim_best_mask_dict = {'none': 0, 'in_final': 1, 'all': 2}
    args.att_trim_best_mask_str = args.att_trim_best_mask
    if args.att_kernel_size > 1:
        args.att_sample_all_masks = False
        args.att_trim_best_mask_str = 'none'
        args.n_kernel_pixels = args.att_kernel_size ** 2
        if args.sparsity < args.n_kernel_pixels:
            args.sparsity = args.n_kernel_pixels
        else:
            args.sparsity = args.sparsity - args.sparsity % args.n_kernel_pixels
        args.kernel_sparsity = args.sparsity // args.n_kernel_pixels
        args.max_kernel_sparsity = (args.data_shape[1] // args.att_kernel_size) * (args.data_shape[2] // args.att_kernel_size)
        args.l0_hist_limits = [args.n_kernel_pixels * i for i in range(args.max_kernel_sparsity + 1)]
    args.att_trim_best_mask = att_trim_best_mask_dict[args.att_trim_best_mask_str]

    if args.att_dpo_dist == "none" or args.att_dpo_mean == 0:
        args.att_dpo_mean = 0
        args.att_dpo_std = 0
        args.att_dpo_dist = "none"
        args.dpo_dis_str = "no_dropout"
    else:
        args.dpo_dis_str = ("distribution_mul_" + args.att_dpo_dist +
                            "_mean_" + str(args.att_dpo_mean).replace('.', '_'))
        if args.att_dpo_dist == "gauss":
            if args.att_dpo_std_bernoulli:
                args.dpo_dis_str += "_std_as_bernoulli"
            else:
                args.dpo_dis_str += "_std_" + str(args.att_dpo_std).replace('.', '_')

        if args.att_scale_dpo_mean:
            args.dpo_dis_str += "_scaled_by_trim_steps"

        args.dpo_dis_str += "_applied_during_"
        if args.att_post_trim_dpo:
            args.dpo_dis_str += "whole_attack"
        else:
            args.dpo_dis_str += "trimming_process"
    
    args.att_trim_str = ""
    if args.att_dynamic_trim:
        args.att_trim_str += "dynamic_"
    if args.att_trim_steps is not None:
        args.sparsity = args.att_trim_steps[-1]
        args.att_max_trim_steps = len(args.att_trim_steps)
        args.att_trim_str += ("steps_" + str(args.att_trim_steps).
                              replace('[', '').replace(']', '').
                              replace(' ', '').replace(',', '_'))
    elif args.att_trim_steps_reduce == 'none':
        args.att_trim_str += "n_trim_steps_" + str(args.att_max_trim_steps)
    elif args.att_trim_steps_reduce == 'best':
        args.att_trim_str += "trim_steps_reduce_best_init_" + str(args.att_max_trim_steps)
    else:
        args.att_trim_steps_reduce = 'even'
        args.att_trim_str += "trim_steps_reduce_even_init_" + str(args.att_max_trim_steps)

    args.att_mask_dist_str = args.att_mask_dist
    
    if args.att_norm_mask_amp:
        args.att_mask_dist_str += "_normalized_amplitude"
    if args.att_mask_opt_iter > 0:
        args.att_mask_dist_str += "_opt_iter_" + str(args.att_mask_opt_iter)
    args.att_mask_dist_str += "_n_samples_" + str(args.att_n_mask_samples)
    att_sample_limit_str = ""
    if args.att_sample_all_masks:
        att_sample_limit_str = "_limited_to_sample_all_masks"
        if args.att_trim_best_mask > 1:
            att_sample_limit_str += "_and_trim_to_best"
        elif args.att_trim_best_mask > 0:
            att_sample_limit_str += "_and_trim_to_best_in_final"
    args.att_mask_dist_str += att_sample_limit_str
    
    args.attack_kernel_str = ""
    if args.att_kernel_size > 1:
        args.attack_kernel_str = "kernel"
        kernel_size_str = str(args.att_kernel_size) + "X" + str(args.att_kernel_size)
        if args.att_kernel_group:
            args.attack_kernel_str += "_grouping" + kernel_size_str
        else:
            args.attack_kernel_str += "_structure_" + kernel_size_str
            if args.att_kernel_min_active:
                args.attack_kernel_str += "_count_min_active"
            
    args.attack_name = args.attack
    args.attack_name = 'PGDTrim'
    args.attack = PGDTrim
    print("Testing models under " + args.attack_name + " attack")     
    args.att_misc_args = {'device': args.device,
                            'dtype': args.dtype,
                            'batch_size': args.batch_size,
                            'data_shape': args.data_shape,
                            'data_RGB_start': args.data_RGB_start,
                            'data_RGB_end': args.data_RGB_end,
                            'data_RGB_size': args.data_RGB_size,
                            'verbose': args.attack_verbose,
                            'report_info': args.report_info}

    args.att_pgd_args = {'norm': 'Linf',
                            'eps': args.eps_l_inf,
                            'n_restarts': args.n_restarts,
                            'n_iter': args.n_iter,
                            'alpha': args.alpha,
                            'rand_init': args.att_rand_init}

    args.att_dpo_args = {'dropout_dist': args.att_dpo_dist,
                            'dropout_mean': args.att_dpo_mean,
                            'dropout_std': args.att_dpo_std,
                            'dropout_std_bernoulli': args.att_dpo_std_bernoulli}

    args.att_trim_args = {'sparsity': args.sparsity,
                            'trim_steps': args.att_trim_steps,
                            'max_trim_steps': args.att_max_trim_steps,
                            'trim_steps_reduce': args.att_trim_steps_reduce,
                            'scale_dpo_mean': args.att_scale_dpo_mean,
                            'post_trim_dpo': args.att_post_trim_dpo,
                            'dynamic_trim': args.att_dynamic_trim}
    
    args.att_mask_args = {'mask_dist': args.att_mask_dist,
                            'mask_prob_amp_rate': args.att_mask_prob_amp_rate,
                            'norm_mask_amp': args.att_norm_mask_amp,
                            'mask_opt_iter': args.att_mask_opt_iter,
                            'n_mask_samples': args.att_n_mask_samples,
                            'sample_all_masks': args.att_sample_all_masks,
                            'trim_best_mask': args.att_trim_best_mask}
    
    if args.att_kernel_size > 1:
        args.attack_name = 'PGDTrimKernel'
        args.attack = PGDTrimKernel
        args.att_kernel_args = {'kernel_size': args.att_kernel_size,
                                'n_kernel_pixels': args.n_kernel_pixels,
                                'kernel_sparsity': args.kernel_sparsity,
                                'max_kernel_sparsity': args.max_kernel_sparsity,
                                'kernel_min_active': args.att_kernel_min_active,
                                'kernel_group': args.att_kernel_group}
    else:
        args.att_kernel_args = None
        args.attack_kernel_str = ""

    
    args.attack_obj_str = "norm_Linf_eps_from_255_" + str(args.eps_l_inf_from_255) + \
                            "_sparsity_" + str(args.sparsity) + \
                            "_iter_" + str(args.n_iter) + \
                            "_restarts_" + str(args.n_restarts) + \
                            "_alpha_" + str(args.alpha).replace('.', '_') + \
                            "_rand_init_" + str(args.att_rand_init)
    args.attack_dpo_str = args.dpo_dis_str
    args.attack_trim_str = "trim_l0_" + args.att_trim_str + \
                            "_mask_distribution_" + args.att_mask_dist_str

    args.attack_obj = args.attack(args.model, args.criterion,
                                        misc_args=args.att_misc_args,
                                        pgd_args=args.att_pgd_args,
                                        dropout_args=args.att_dpo_args,
                                        trim_args=args.att_trim_args,
                                        mask_args=args.att_mask_args,
                                        kernel_args=args.att_kernel_args)
    return args


def compute_save_path_args(args):
    if args.results_dir is None or not len(args.results_dir) or not args.save_results:
        return args
    if not isdir(args.results_dir):
        mkdir(args.results_dir)
    args.data_save_path = args.results_dir + '/' + args.dataset
    if not isdir(args.data_save_path):
        mkdir(args.data_save_path)
    args.data_save_path = args.data_save_path + '/n_examples_' + str(args.n_examples)
    if not isdir(args.data_save_path):
        mkdir(args.data_save_path)
    args.model_save_path = args.data_save_path + '/model_' + args.model_name
    if not isdir(args.model_save_path):
        mkdir(args.model_save_path)
    args.attack_class_save_path = args.model_save_path + '/attack_' + args.attack_name
    if not isdir(args.attack_class_save_path):
        mkdir(args.attack_class_save_path)
    args.attack_obj_save_path = args.attack_class_save_path + '/' + args.attack_obj_str
    if not isdir(args.attack_obj_save_path):
        mkdir(args.attack_obj_save_path)
    if len(args.attack_dpo_str):
        args.attack_obj_save_path = args.attack_obj_save_path + '/' + args.attack_dpo_str
        if not isdir(args.attack_obj_save_path):
            mkdir(args.attack_obj_save_path)
    if len(args.attack_trim_str):
        args.attack_obj_save_path = args.attack_obj_save_path + '/' + args.attack_trim_str
        if not isdir(args.attack_obj_save_path):
            mkdir(args.attack_obj_save_path)
    if len(args.attack_kernel_str):
        args.attack_obj_save_path = args.attack_obj_save_path + '/' + args.attack_kernel_str
        if not isdir(args.attack_obj_save_path):
            mkdir(args.attack_obj_save_path)
    args.results_save_path = args.attack_obj_save_path
    args.adv_pert_save_path = args.results_save_path + '/perturbations'
    if not isdir(args.adv_pert_save_path):
        mkdir(args.adv_pert_save_path)
    args.imgs_save_path = args.results_save_path + '/images'
    if not isdir(args.imgs_save_path):
        mkdir(args.imgs_save_path)
    if args.save_results:
        print("Results save path:")
        print(args.results_save_path)
    return args


def get_args():
    args = parse_args()
    args = compute_run_args(args)
    args = compute_data_args(args)
    args = compute_models_args(args)
    args = compute_attack_args(args)
    args = compute_save_path_args(args)
    return args


def save_img_tensors(path, x, gt=None, pred=None, labels_str_dict=None, save_type=".pdf"):
    if not isdir(path):
        mkdir(path)
    for idx, input in enumerate(x):
        save_path = path + '/' + str(idx)
        if not isdir(save_path):
            mkdir(save_path)
        name = "img"
        if labels_str_dict is not None:
            if gt is not None:
                input_gt = gt[idx]
                gt_label = labels_str_dict[input_gt]
                name += "_gt_label_" + gt_label
            if pred is not None:
                input_pred = pred[idx]
                pred_label = labels_str_dict[input_pred]
                name += "_pred_label_" + pred_label
        name += save_type
        save_image(input, save_path + '/' + name)
