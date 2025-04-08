import torch
import torch.backends.cudnn as cudnn
import argparse
from os import mkdir
from os.path import isdir
from torchvision.utils import save_image
import traceback


from attacks.pgd_attacks import PGDTrim
from attacks.pgd_attacks import PGDTrimKernel

from data.Im2GPS3k.download import load_im2gps_data, CLIP_load_data_tensor
from data.MP_16.download import load_mp16_data, get_transforms as get_mp16_transforms
from data.mixed_dataset.download import get_mixed_dataloader, get_transforms as get_mixed_transforms

def parse_args():
    parser = argparse.ArgumentParser()
    
    # run args
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                      help='random seed (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                      help='device to use (default: None)')
    parser.add_argument('--force_cpu', action='store_true',
                      help='force using cpu')
    parser.add_argument('--attack_verbose', action='store_true',
                      help='print attack progress')
    
    # GeoClip specific args
    parser.add_argument('--targeted', action='store_true',
                      help='perform targeted attack')
    parser.add_argument('--target_coords', type=float, nargs=2, default=None,
                      help='target coordinates (lat, lon) for targeted attack')
    parser.add_argument('--continent_threshold', type=float, default=2500,
                      help='Distance threshold for continent level (km)')
    parser.add_argument('--country_threshold', type=float, default=750,
                      help='Distance threshold for country level (km)')
    parser.add_argument('--region_threshold', type=float, default=200,
                      help='Distance threshold for region level (km)')
    parser.add_argument('--city_threshold', type=float, default=25,
                      help='Distance threshold for city level (km)')
    parser.add_argument('--street_threshold', type=float, default=1,
                      help='Distance threshold for street level (km)')
    
    # data args
    parser.add_argument('--dataset', type=str, choices=['Im2GPS', 'Im2GPS3k', 'YFCC26k', 'MP_16', 'mixed'],
                      default='Im2GPS3k', help='dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='path to data directory')
    parser.add_argument('--samples_per_dataset', type=int, default=1000,
                      help='number of samples per dataset when using mixed dataset')
    parser.add_argument('--max_images', type=int, default=None,
                      help='maximum number of images to load from MP_16 dataset')
    parser.add_argument('--n_examples', type=int, default=10000,
                      help='number of examples to attack')
    parser.add_argument('--batch_size', type=int, default=250,
                      help='batch size for attack')
    
    # results args
    parser.add_argument('--results_dir', type=str, default='./results',
                      help='directory to save results')
    parser.add_argument('--report_info', action='store_true',
                      help='report additional info and non-final results')
    parser.add_argument('--save_results', action='store_true',
                      help='save attack results')
    parser.add_argument('--l0_hist_limits', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                      help='L0 histogram limits')
    
    # model args
    parser.add_argument('--model_name', type=str, default='',
                      help='model name for robustbench models')
    parser.add_argument('--model_transform_input', action='store_true',
                      help='transform input before model forward pass')
    
    # attack args
    parser.add_argument('--attack', type=str, default='PGDTrim',
                      help='attack type')
    parser.add_argument('--eps_l_inf_from_255', type=float, default=8,
                      help='L_inf norm bound (from 255)')
    parser.add_argument('--sparsity', type=int, default=1,
                      help='L0 sparsity')
    parser.add_argument('--n_iter', type=int, default=100,
                      help='number of iterations')
    parser.add_argument('--n_restarts', type=int, default=1,
                      help='number of restarts')
    parser.add_argument('--alpha', type=float, default=None,
                      help='step size')
    
    # PGDTrim specific args
    parser.add_argument('--att_init_zeros', action='store_true',
                      help='initialize perturbation with zeros')
    parser.add_argument('--att_dpo_dist', type=str, default='none',
                      help='dropout distribution')
    parser.add_argument('--att_dpo_mean', type=float, default=0,
                      help='dropout mean')
    parser.add_argument('--att_dpo_std', type=float, default=0,
                      help='dropout standard deviation')
    parser.add_argument('--att_trim_steps', type=int, nargs='+', default=None,
                      help='trim steps')
    parser.add_argument('--att_max_trim_steps', type=int, default=None,
                      help='maximum trim steps')
    parser.add_argument('--att_trim_steps_reduce', type=str, default='none',
                      help='trim steps reduction policy')
    parser.add_argument('--att_const_dpo_mean', action='store_true',
                      help='use constant dropout mean')
    parser.add_argument('--att_const_dpo_std', action='store_true',
                      help='use constant dropout standard deviation')
    parser.add_argument('--att_post_trim_dpo', action='store_true',
                      help='apply dropout after trimming')
    parser.add_argument('--att_dynamic_trim', action='store_true',
                      help='use dynamic trimming')
    parser.add_argument('--att_mask_dist', type=str, default='bernoulli',
                      help='mask distribution')
    parser.add_argument('--att_mask_prob_amp_rate', type=float, default=0,
                      help='mask probability amplification rate')
    parser.add_argument('--att_norm_mask_amp', action='store_true',
                      help='normalize mask amplitude')
    parser.add_argument('--att_mask_opt_iter', type=int, default=0,
                      help='mask optimization iterations')
    parser.add_argument('--att_n_mask_samples', type=int, default=1,
                      help='number of mask samples')
    parser.add_argument('--att_no_samples_limit', action='store_true',
                      help='no limit on number of samples')
    parser.add_argument('--att_trim_best_mask', type=int, default=1,
                      help='trim best mask')
    parser.add_argument('--att_kernel_size', type=int, default=1,
                      help='kernel size')
    parser.add_argument('--att_kernel_min_active', action='store_true',
                      help='use minimum active kernel')
    parser.add_argument('--att_kernel_group', action='store_true',
                      help='use kernel grouping')
    
    args = parser.parse_args()
    return args

def compute_run_args(args):
    if args.force_cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return args

def compute_data_args(args):
    if args.n_examples < args.batch_size:
        args.batch_size = args.n_examples
    if args.attack == 'SF' or args.attack == 'GF' or args.attack == 'Homotopy':
        args.batch_size = 1

    # Set default data normalization parameters
    args.data_RGB_start = [0, 0, 0]
    args.data_RGB_end = [1, 1, 1]
    args.data_RGB_offset = [0, 0, 0]
    args.data_RGB_size = [1, 1, 1]
    args.model_transform_input = True

    # Handle GeoClip datasets
    if args.dataset in ['Im2GPS', 'Im2GPS3k', 'YFCC26k', 'MP_16', 'mixed']:
        transform = get_mixed_transforms()
        
        if args.dataset == 'mixed':
            dataloader = get_mixed_dataloader(
                args.data_dir,
                batch_size=args.batch_size,
                samples_per_dataset=args.samples_per_dataset,
                transform=transform,
            )
        elif args.dataset == 'MP_16':
            dataloader = load_mp16_data(
                args.data_dir,
                batch_size=args.batch_size,
                transform=transform,
                max_images=args.max_images
            )
        else:  # Im2GPS3k
            dataloader = load_im2gps_data(
                args.data_dir,
                batch_size=args.batch_size,
                transform=transform
            )
            
        # Convert dataloader to tensors
        x_tensors = []
        y_tensors = []
        for x, y in dataloader:
            x_tensors.append(x)
            y_tensors.append(y)
        
        args.x_test = torch.cat(x_tensors, dim=0)
        args.y_test = torch.cat(y_tensors, dim=0)
        
        # Only keep the first n_examples samples
        if args.n_examples is not None and args.n_examples < args.x_test.shape[0]:
            args.x_test = args.x_test[:args.n_examples]
            args.y_test = args.y_test[:args.n_examples]
            print(f"\nTruncated dataset to first {args.n_examples} examples")
        
        args.n_classes = None  # GeoClip predicts continuous coordinates, not classes
    
    args.n_examples = args.x_test.shape[0]
    args.data_channels = args.x_test.shape[1]
    args.data_shape = list(args.x_test.shape)[1:]
    args.data_pixels = args.data_shape[1] * args.data_shape[2]
    args.dtype = args.x_test.dtype
    return args

def compute_models_args(args):
    try:
        # Import and initialize GeoClip model
        from geoclip.model.GeoCLIP import GeoCLIP
        args.model = GeoCLIP(from_pretrained=True)  # Initialize GeoClip model with pretrained weights
        args.model = args.model.to(args.device)
        args.model.eval()
        print("GeoClip model loaded successfully")
    except Exception as e:
        print(f"\nError loading model: {e}")
        print(traceback.format_exc())
        raise
    return args

def compute_attack_args(args):
    # For GeoClip, we don't use a criterion since we compute distance-based loss directly in the attack
    args.criterion = None
    print("Using distance-based loss for GeoClip")
    
    # Convert trim_best_mask value to the appropriate format
    att_trim_best_mask_dict = {'none': 0, 'in_final': 1, 'all': 2}
    
    # Setup dropout string configuration
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
    
    # Force PGDTrimKernel when kernel_size > 1
    if args.att_kernel_size > 1:
        args.attack = PGDTrimKernel  # Set the actual class, not just the name
        args.attack_name = 'PGDTrimKernel'
        print(f"\nUsing kernel size: {args.att_kernel_size}x{args.att_kernel_size}")
        args.att_sample_all_masks = False
        args.att_trim_best_mask = 0  # Force none for kernel mode
        args.n_kernel_pixels = args.att_kernel_size ** 2
        if args.sparsity < args.n_kernel_pixels:
            args.sparsity = args.n_kernel_pixels
        else:
            args.sparsity = args.sparsity - args.sparsity % args.n_kernel_pixels
        args.kernel_sparsity = args.sparsity // args.n_kernel_pixels
        args.max_kernel_sparsity = (args.data_shape[1] // args.att_kernel_size) * (args.data_shape[2] // args.att_kernel_size)
        args.l0_hist_limits = [args.n_kernel_pixels * i for i in range(args.max_kernel_sparsity + 1)]
        print(f"Adjusted sparsity: {args.sparsity}")
        print(f"Kernel sparsity: {args.kernel_sparsity}")
        print(f"Max kernel sparsity: {args.max_kernel_sparsity}")
    else:
        args.attack = PGDTrim  # Set the actual class for non-kernel mode
        args.attack_name = 'PGDTrim'
        # For non-kernel mode, convert trim_best_mask value if it's a string
        if isinstance(args.att_trim_best_mask, str):
            args.att_trim_best_mask = att_trim_best_mask_dict[args.att_trim_best_mask]
    
    args.eps_l_inf = args.eps_l_inf_from_255 / 255
    args.type_attack = 'L0'
    args.l0_attacks_eps = -1
    if args.eps_l_inf < 1:
        args.type_attack = 'L0+Linf'
        args.l0_attacks_eps = args.eps_l_inf
    print(f"Attack type: {args.type_attack}")
    print(f"L∞ epsilon: {args.eps_l_inf:.4f}")
    
    args.att_rand_init = not args.att_init_zeros
    args.att_sample_all_masks = not args.att_no_samples_limit
    args.att_scale_dpo_mean = not args.att_const_dpo_std
    args.att_dpo_std_bernoulli = not args.att_const_dpo_mean
    
    print("\nAttack parameters:")
    print(f"Random initialization: {args.att_rand_init}")
    print(f"Sample all masks: {args.att_sample_all_masks}")
    print(f"Scale dropout mean: {args.att_scale_dpo_mean}")
    print(f"Bernoulli dropout std: {args.att_dpo_std_bernoulli}")
    
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
        if args.att_max_trim_steps is None:
            args.att_max_trim_steps = 10  # Set default max trim steps
        args.att_trim_str += "n_trim_steps_" + str(args.att_max_trim_steps)
    elif args.att_trim_steps_reduce == 'best':
        if args.att_max_trim_steps is None:
            args.att_max_trim_steps = 10  # Set default max trim steps
        args.att_trim_str += "trim_steps_reduce_best_init_" + str(args.att_max_trim_steps)
    else:
        args.att_trim_steps_reduce = 'even'
        if args.att_max_trim_steps is None:
            args.att_max_trim_steps = 10  # Set default max trim steps
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
        args.attack_kernel_str = "kernel_" + args.attack_name
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

    args.att_misc_args = {'device': args.device,
                         'dtype': args.dtype,
                         'batch_size': args.batch_size,
                         'data_shape': args.data_shape,
                         'data_RGB_start': args.data_RGB_start,
                         'data_RGB_end': args.data_RGB_end,
                         'data_RGB_size': args.data_RGB_size,
                         'verbose': args.attack_verbose,
                         'report_info': args.report_info}

    args.att_dpo_args = {'dropout_dist': args.att_dpo_dist,
                         'dropout_mean': args.att_dpo_mean,
                         'dropout_std': args.att_dpo_std,
                         'dropout_std_bernoulli': args.att_dpo_std_bernoulli}

    args.att_mask_args = {'mask_dist': args.att_mask_dist,
                          'mask_prob_amp_rate': args.att_mask_prob_amp_rate,
                          'norm_mask_amp': args.att_norm_mask_amp,
                          'mask_opt_iter': args.att_mask_opt_iter,
                          'n_mask_samples': args.att_n_mask_samples,
                          'sample_all_masks': args.att_sample_all_masks,
                          'trim_best_mask': args.att_trim_best_mask}

    args.att_trim_args = {'sparsity': args.sparsity,
                          'trim_steps': args.att_trim_steps,
                          'max_trim_steps': args.att_max_trim_steps,
                          'trim_steps_reduce': args.att_trim_steps_reduce,
                          'scale_dpo_mean': args.att_scale_dpo_mean,
                          'post_trim_dpo': args.att_post_trim_dpo,
                          'dynamic_trim': args.att_dynamic_trim}

    args.att_pgd_args = {'norm': 'Linf',
                         'eps': args.eps_l_inf,
                         'n_restarts': args.n_restarts,
                         'n_iter': args.n_iter,
                         'alpha': args.eps_l_inf/4 if args.alpha is None else args.alpha,
                         'rand_init': args.att_rand_init}

    # Initialize the attack object
    args.attack_obj = args.attack(args.model, args.criterion,
                              misc_args=args.att_misc_args,
                              pgd_args=args.att_pgd_args,
                              dropout_args=args.att_dpo_args,
                              trim_args=args.att_trim_args,
                              mask_args=args.att_mask_args,
                              kernel_args=args.att_kernel_args if args.att_kernel_size > 1 else None)
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

if __name__ == "__main__":
    print("\n=== Starting Main Execution ===")
    try:
        print("Getting arguments...")
        args = get_args()
        print("Arguments retrieved successfully")
        print("\nFinal configuration:")
        print(f"Device: {args.device}")
        print(f"Model: {args.model_name}")
        print(f"Dataset: {args.dataset}")
        print(f"Attack: {args.attack_name}")
    except Exception as e:
        print(f"\nError in main execution: {e}")
        print("Stack trace:", traceback.format_exc())
        raise