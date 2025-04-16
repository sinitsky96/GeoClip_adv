import time
import torch
import torchvision
import numpy as np
import torch.optim as optim
import os
import argparse
from utils import * 

from data.mixed_dataset.download import get_mixed_dataloader, get_transforms as get_mixed_transforms
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import TensorDataset, DataLoader, random_split
from sparse_rs.attack_sparse_rs import AttackGeoCLIP, ClipWrap
from sparse_rs.util import haversine_distance, CONTINENT_R, STREET_R, CITY_R, REGION_R, COUNTRY_R 
from geoclip.model.GeoCLIP import GeoCLIP



def get_aug():
    parser = argparse.ArgumentParser(description='Feature-Gathering Universal Adversarial Perturbation')
    parser.add_argument('--remark', default='', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=2, type=int)
    parser.add_argument('--model_name', default='CLIP', type=str,
        help='Choose from "alexnet, googlenet, vgg16, vgg19, resnet50, resnet152, deit_tiny, deit_small, deit_base".')
    parser.add_argument('--train_data_dir', default='path_of_train_data', type=str)
    parser.add_argument('--val_data_dir', default='path_of_validation_data', type=str)
    parser.add_argument('--result_dir', default='path_of_result_dir', type=str)
    parser.add_argument('--xi', default=0.0392, type=float)
    parser.add_argument('--p', default=np.inf, type=float)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--top_k', default=[1,3,5], nargs='+', type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--target', default=-1, type=int)
    parser.add_argument('--target_param', default=0.1, type=float)
    parser.add_argument('--val_freq', default=1, type=int)


    parser.add_argument('--data_path', type=str, default="./data")


    args = parser.parse_args()
    return args

def create_logger(args):
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    exp_time = time.strftime('%m%d_%H%M')
    log_dir = os.path.join(args.result_dir, exp_time + '_' + args.model_name + '_' + str(args.target))
    if args.remark:
        log_dir += '_' + args.remark
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = my_logger(args, os.path.join(log_dir, 'log.txt'))
    logger.info(args)
    return logger, log_dir

def main():
    start_time = time.time()
    args = get_aug()

    logger, log_dir = create_logger(args)

    torch.cuda.set_device(int(args.gpu))
    device = torch.device( args.device if torch.cuda.is_available() else "cpu" )

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #### can be cleaned up, copy pasta pipeline from RS #######

    transform = get_mixed_transforms()
    dataloader = get_mixed_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        samples_per_dataset=args.samples_per_dataset,
        transform=transform,
        clip_varient=True if args.model.lower() == 'clip' else False,
        shuffle=False,
    )

    x_tensors = []
    y_tensors = []
    labels = []
    for x, y, label in dataloader:
        x_tensors.append(x)
        y_tensors.append(y)
        labels.append(label)
    
    x_test = torch.cat(x_tensors, dim=0)
    y_test = torch.cat(labels, dim=0)

    dataset = TensorDataset(x_test, y_test)


    train_ratio = 0.8
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    y_test_geo = torch.cat(y_tensors, dim=0)


    ########################################################

    net = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    net.to(device)
    net.eval()
    net = ClipWrap(net, args.data_path, device=device)

    # train_loader, val_loader = get_data(args.train_data_dir, args.val_data_dir, args.batch_size)
    attacker = FG_UAP(args.xi, args.p, net, logger, args.target, args.target_param)
    uap, fr = attacker.attack(train_loader, val_loader, args.max_epoch, args.lr, args.top_k, args.val_freq)
    logger.info('Best FR: {:.2f}. Total time: {:.2f}\n'.format(fr, time.time()-start_time))
    uap = uap.data.cpu()
    torch.save(uap, os.path.join(log_dir, args.model_name + '_{:.2f}.pth'.format(fr)))

    # check robust accuracy and other statistics
    model = GeoCLIP()
    model.to(device)
    model.eval()

    all_clean_dist = []
    all_adv_dist = []
    n_examples = len(x_test)
    bs = args.batch_size
    adv_complete = x_test + uap
    adv_complete = torch.clamp(adv_complete, 0, 1)

    n_batches = int(np.ceil(n_examples / bs))
    for batch_idx in range(n_batches):
        torch.cuda.empty_cache()

        start_idx = batch_idx * bs
        end_idx = min((batch_idx + 1) * bs, n_examples)

        x_curr_clean = x_test[start_idx:end_idx].clone().detach().to(device)
        x_curr_adv = adv_complete[start_idx:end_idx].clone().detach().to(device)
        y_curr = y_test_geo[start_idx:end_idx].clone().detach().to(device)
        
        diff_imgs = (x_curr_clean - x_curr_adv).abs().view(x_curr_clean.shape[0], -1).max(dim=1)[0]
        threshold = 1e-6  # images with maximum difference below this are considered unaltered
        successful_mask = diff_imgs > threshold

        if successful_mask.sum() == 0:
            continue

        x_curr_clean = x_curr_clean[successful_mask]
        x_curr_adv = x_curr_adv[successful_mask]
        y_curr = y_curr[successful_mask]
        
        output_clean, _ = model.predict_from_tensor(x_curr_clean)
        output_adv, _ = model.predict_from_tensor(x_curr_adv)

        clean_dist = haversine_distance(output_clean, y_curr, True)
        adv_dist = haversine_distance(output_adv, y_curr, True)
        
        all_clean_dist.append(clean_dist.cpu())
        all_adv_dist.append(adv_dist.cpu())


    thresholds = [STREET_R, CITY_R, REGION_R, COUNTRY_R, CONTINENT_R]
    labels = {
        STREET_R: "STREET_R (1 km)",
        CITY_R: "CITY_R (25 km)",
        REGION_R: "REGION_R (200 km)",
        COUNTRY_R: "COUNTRY_R (750 km)",
        CONTINENT_R: "CONTINENT_R (2500 km)"
    }

    distances_clean = torch.cat(all_clean_dist)
    distances_adv = torch.cat(all_adv_dist)
    logger.info("GeoCLIP attack stats")
    logger.info("--------------------")
    targeted_srt = f"targeted location: {args.target_class}"
    untargeted_srt = f"true location of the examples"
    logger.info(f"The following logs are relative to the location of the {targeted_srt if args.targeted else untargeted_srt}")
    for T in thresholds:
        percent_T_clean = (distances_clean <= T).float().mean().item() * 100.0
        logger.info(f"Percentage of clean predictions within {labels[T]}: {percent_T_clean:.2f}%")
    for T in thresholds:
        percent_T_adv = (distances_adv <= T).float().mean().item() * 100.0
        logger.info(f"Percentage of adv predictions within {labels[T]}: {percent_T_adv:.2f}%")
    

    if args.targeted:
        improvement = distances_clean - distances_adv
    else:
        improvement = distances_adv - distances_clean

    pos_mask = improvement > 0
    percent_improved = pos_mask.float().mean().item() * 100.0
    if pos_mask.sum() > 0:
        avg_improvement = improvement[pos_mask].mean().item()
        median_improvement = improvement[pos_mask].median().item()
    else:
        avg_improvement = 0.0
        median_improvement = 0.0
    logger.info(f"Percentage of positivly changed examples distance: {percent_improved:.2f}%")
    logger.info(f"Average change in distance (km) from positivly predicted examples: {avg_improvement:.2f}")
    logger.info(f"Median change in distance (km) from positivly predicted examples: {median_improvement:.2f}")

    
    logger.info(f"Average change in distance (km) in overall predicted examples: {improvement.mean().item():.2f}")
    logger.info(f"Median change in distance (km) in overall predicted examples: {improvement.median().item():.2f}")


if __name__ == '__main__':
    main()