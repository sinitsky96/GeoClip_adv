import os
import argparse
import torch
import numpy as np


from geoclip.model.GeoCLIP import GeoCLIP
from data.Im2GPS3k.download import load_im2gps_data, CLIP_load_data_tensor
from data.MP_16.download import load_mp16_data, get_transforms as get_mp16_transforms

from sparse_rs.attack_sparse_rs import AttackGeoCLIP, ClipWrap
from sparse_rs.util import haversine_distance, CONTINENT_R, STREET_R, CITY_R, REGION_R, COUNTRY_R 
from transformers import CLIPProcessor, CLIPModel

from datetime import datetime

from utils import SingleChannelModel

from torchvision import transforms



def random_target_classes(y_pred, n_classes):
    y = torch.zeros_like(y_pred)
    for counter in range(y_pred.shape[0]):
        l = list(range(n_classes))
        l.remove(y_pred[counter])
        t = torch.randint(0, len(l), size=[1])
        y[counter] = l[t] + 0

    return y.long()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--norm', type=str, default='L0') # type of the attack : 'L0', 'patches', 'frames', 'patches_universal', 'frames_universal'
    
    # pixel attack k = number of pixels, 
    # feature space attack k = number of features, 
    # image-specific frames of width (patch) k, k = frame width
    parser.add_argument('--k', default=150., type=float)

    parser.add_argument('--n_restarts', type=int, default=1) # Number of random restarts
    parser.add_argument('--loss', type=str, default='margin') # loss function for the attack, options: 'margin', 'ce'
    parser.add_argument('--n_ex', type=int, default=1000) # dataset size
    parser.add_argument('--bs', type=int, default=128) # batch size
    parser.add_argument('--n_queries', type=int, default=1000) # max number of queries, how many times we can prob the model.
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--constant_schedule', action='store_true') 
    parser.add_argument('--use_feature_space', action='store_true') # whether we attack the feature space or the picture
    
    # Sparse-RS parameter
    parser.add_argument('--alpha_init', type=float, default=.3)
    parser.add_argument('--resample_period_univ', type=int)
    parser.add_argument('--loc_update_period', type=int)

    
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--target_class', type=eval)

    parser.add_argument('--model', default='geoclip', type=str)
    parser.add_argument('--dataset', type=str, default='Im2GPS3k', choices=['Im2GPS', 'Im2GPS3k', 'YFCC26k', 'MP_16'])
    parser.add_argument('--save_dir', type=str, default='./results/new')
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--max_images', type=int, default=1000, help='Maximum number of images to download for MP-16 dataset')


    parser.add_argument('--device', type=str, default='cuda')

    
    args = parser.parse_args()
    
    args.eps = args.k + 0
    # args.bs = args.n_ex + 0
    args.p_init = args.alpha_init + 0.
    args.resample_loc = args.resample_period_univ
    args.update_loc_period = args.loc_update_period
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device( args.device if torch.cuda.is_available() else "cpu" )
    print("getting data")
    if args.dataset == 'Im2GPS3k':
        if args.model == 'clip':
            x_test, y_test, y_test_geo = CLIP_load_data_tensor(args.data_path)
        else: # geo clip
            x_test, y_test = load_im2gps_data(args.data_path)
        n_examples = y_test.shape[0]
        args.n_ex = n_examples
        print(f"ytest shape: P{y_test.shape}")
        print(f"y_test: {y_test}")

    elif args.dataset == 'YFCC26k':
        pass

    elif args.dataset == 'MP_16':
        # Get transforms for preprocessing MP-16 images
        transform = get_mp16_transforms(apply_transforms=True)
        
        # Load data with a limit on the number of images to download
        print(f"Loading MP-16 dataset with max_images={args.max_images}")
        x_list, y_list = load_mp16_data(args.data_path, max_images=args.max_images, transform=transform)
        
        # Convert image list to tensor
        x_tensors = []
        for img in x_list:
            if not isinstance(img, torch.Tensor):
                img = transform(img)
            x_tensors.append(img)
        
        # Stack all images into a single tensor
        x_test = torch.stack(x_tensors)
        
        # Convert coordinates list to tensor 
        y_coords = []
        for coords in y_list:
            y_coords.append(torch.tensor(coords, dtype=torch.float32))
        
        y_test = torch.stack(y_coords)
        
        n_examples = len(x_test)
        args.n_ex = min(args.n_ex, n_examples)
        print("x_test shape: {}, y_test shape: {}".format(x_test.shape, y_test.shape))
        


    if args.model.lower() == "geoclip":
        model = GeoCLIP()
        model.to(device)
        model.eval()
    elif args.model.lower() == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model.to(device)
        model.eval()
        model = ClipWrap(model, args.data_path, device= device)

    
        
    # run Sparse-RS attacks
    logsdir = '{}/logs_{}_{}'.format(args.save_dir, "sparse_rs", args.norm)
    savedir = '{}/{}_{}'.format(args.save_dir, "sparse_rs", args.norm)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(logsdir):
        os.makedirs(logsdir)

    data_loader = None # The code uses this in universial attacks, we dont need this.

    if args.use_feature_space:
        # reshape images to single color channel to perturb them individually
        assert args.norm == 'L0'
        bs, c, h, w = x_test.shape
        x_test = x_test.view(bs, 1, h, w * c)
        model = SingleChannelModel(model)
        str_space = 'feature space'
    else:
        str_space = 'pixel space'
    
    param_run = '{}_{}_{}_1_{}_nqueries_{:.0f}_pinit_{:.2f}_loss_{}_eps_{:.0f}_targeted_{}_targetclass_{}_seed_{:.0f}'.format(
        "sparse_rs", args.norm, args.model, args.n_ex, args.n_queries, args.p_init,
        args.loss, args.eps, args.targeted, args.target_class, args.seed)
    if args.constant_schedule:
        param_run += '_constantpinit'
    if args.use_feature_space:
        param_run += '_featurespace'

    # set target classes
    if args.targeted and 'universal' in args.norm:
        if args.target_class is None:
            y_test = torch.ones_like(y_test) * torch.randint(1000, size=[1]).to(y_test.device)
        else:
            target_tensor = torch.tensor(args.target_class, dtype=torch.float32)
            y_test = torch.ones_like(y_test) * args.target_class
        print('target labels', y_test)
    
    elif args.targeted:
        if args.target_class is None:
            raise ValueError(f'Expected a --target_class tuple argumanet. for example: --target_class "(37.090924,25.370521)"')
        target_tensor = torch.tensor(args.target_class, dtype=torch.float32)
        y_test = target_tensor.repeat(y_test.shape[0],1)
        print('target labels', y_test)
    

    if args.model.lower() == "geoclip":
        from sparse_rs.attack_sparse_rs import AttackGeoCLIP
        adversary = AttackGeoCLIP(model, norm=args.norm, eps=int(args.eps), verbose=True, n_queries=args.n_queries,
            p_init=args.p_init, log_path='{}/log_run_{}_{}.txt'.format(logsdir, str(datetime.now())[:-7], param_run),
            loss=args.loss, targeted=args.targeted, seed=args.seed, constant_schedule=args.constant_schedule,
            data_loader=data_loader, resample_loc=args.resample_loc,device=device, geoclip_attack=True, y_test=y_test)
    elif args.model.lower() == "clip":
        from sparse_rs.attack_sparse_rs import AttackCLIP
        adversary = AttackCLIP(model, data_path=args.data_path, norm=args.norm, eps=int(args.eps), verbose=True, n_queries=args.n_queries,
            p_init=args.p_init, log_path='{}/log_run_{}_{}.txt'.format(logsdir, str(datetime.now())[:-7], param_run),
            loss=args.loss, targeted=args.targeted, seed=args.seed, constant_schedule=args.constant_schedule,
            data_loader=data_loader, resample_loc=args.resample_loc,device=device)
    else:
        from rs_attacks import RSAttack
        adversary = RSAttack(model, norm=args.norm, eps=int(args.eps), verbose=True, n_queries=args.n_queries,
            p_init=args.p_init, log_path='{}/log_run_{}_{}.txt'.format(logsdir, str(datetime.now())[:-7], param_run),
            loss=args.loss, targeted=args.targeted, seed=args.seed, constant_schedule=args.constant_schedule,
            data_loader=data_loader, resample_loc=args.resample_loc, device=device)
        
    
    

    bs = args.bs
    n_batches = int(np.ceil(n_examples / bs))
    adv_complete = x_test.clone()
    qr_complete = torch.zeros([x_test.shape[0]]).cpu()
    pred = torch.zeros([0]).float().cpu()
    print("starting clean classification")
    distances_clean = []

    with torch.no_grad():
        # find points originally correctly classified
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, n_examples)

            
            x_curr = x_test[start_idx:end_idx].clone().detach().to(device)
            y_curr = y_test[start_idx:end_idx].clone().detach().to(device)
            
            
            if args.model.lower() != "geoclip":
                output = model(x_curr)
                if not args.targeted:
                    pred = torch.cat((pred, (output.max(1)[1] == y_curr).float().cpu()), dim=0)
                else:
                    pred = torch.cat((pred, (output.max(1)[1] != y_curr).float().cpu()), dim=0)
            else:
                output, _ = model.predict_from_tensor(x_curr)
                distances = haversine_distance(output, y_curr)
                distances_clean.append(distances)

                if not args.targeted: # no need to attack guesses that are already very bad
                    pred = torch.cat((pred, (distances <= CONTINENT_R).float().cpu()), dim=0)
                else:
                    pred = torch.cat((pred, (distances > STREET_R).float().cpu()), dim=0)

            del x_curr
            del y_curr
            del output
            torch.cuda.empty_cache()
        
        adversary.logger.log('clean accuracy {:.2%}'.format(pred.mean()))

        n_batches = int(np.ceil(pred.sum() / bs))

        if args.model.lower() == "geoclip":
            ind_to_fool = (pred == 1).nonzero(as_tuple=True)[0]
        else:
            ind_to_fool = (pred == 1).nonzero().squeeze()
        
        # run the attack
        pred_adv = pred.clone()
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, n_examples)
            # print(f"y_test{y_test}")
            x_curr = x_test[ind_to_fool[start_idx:end_idx]].clone().detach().to(device)
            y_curr = y_test[ind_to_fool[start_idx:end_idx]].clone().detach().to(device)
            # print(f"y_curr{y_curr}")
            qr_curr, adv = adversary.perturb(x_curr, y_curr)
            adv = adv.to(device)

            if args.model.lower() != "geoclip":
                output = model(adv)
                if not args.targeted:
                    acc_curr = (output.max(1)[1] == y_curr).float().cpu()
                else:
                    acc_curr = (output.max(1)[1] != y_curr).float().cpu()
            else:
                output, _ = model.predict_from_tensor(adv)
                dists = haversine_distance(output, y_curr)
                if not args.targeted:
                    acc_curr = (dists <= CONTINENT_R).float().cpu()
                else:
                    acc_curr = (dists > STREET_R).float().cpu()

            
            pred_adv[ind_to_fool[start_idx:end_idx]] = acc_curr.clone() # mask of successful predictions under adv (robust)
            adv_complete[ind_to_fool[start_idx:end_idx]] = adv.cpu().clone() # adv examples
            qr_complete[ind_to_fool[start_idx:end_idx]] = qr_curr.cpu().clone() # amount of queries
            
            print('batch {}/{} - {:.0f} of {} successfully perturbed'.format(
                batch_idx + 1, n_batches, x_curr.shape[0] - acc_curr.sum(), x_curr.shape[0]))
            
            del x_curr
            del y_curr
            del adv
            del qr_curr
            del acc_curr
            torch.cuda.empty_cache()
            
        adversary.logger.log('robust accuracy {:.2%}'.format(pred_adv.float().mean()))
        
        # check robust accuracy and other statistics
        acc = 0.
        n_batches = int(np.ceil(n_examples / bs))
        distances_adv = []

        for batch_idx in range(n_batches):
            torch.cuda.empty_cache()

            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, n_examples)

            x_curr = adv_complete[start_idx:end_idx].clone().detach().to(device)
            y_curr = y_test[start_idx:end_idx].clone().detach().to(device)
            
            if args.model.lower() != "geoclip":
                output = model(x_curr)
                if not args.targeted:
                    acc += (output.max(1)[1] == y_curr).float().sum().item()
                else:
                    acc += (output.max(1)[1] != y_curr).float().sum().item()
            else:
                output, _ = model.predict_from_tensor(x_curr)
                dists = haversine_distance(output, y_curr)
                distances_adv.append(dists)
                if not args.targeted:
                    acc += (dists <= CONTINENT_R).float().sum().item()
                else:
                    acc += (dists > STREET_R).float().sum().item()

        thresholds = [STREET_R, CITY_R, REGION_R, COUNTRY_R, CONTINENT_R]
        labels = {
            STREET_R: "STREET_R (1 km)",
            CITY_R: "CITY_R (25 km)",
            REGION_R: "REGION_R (200 km)",
            COUNTRY_R: "COUNTRY_R (750 km)",
            CONTINENT_R: "CONTINENT_R (2500 km)"
        }

        if args.model.lower() != "geoclip":
            adversary.logger.log('robust accuracy {:.2%}'.format(acc / args.n_ex))
        else:
            distances_clean = torch.cat(distances_clean)
            distances_adv = torch.cat(distances_adv)
            adversary.logger.log("GeoCLIP attack stats")
            adversary.logger.log("--------------------")
            targeted_srt = f"targeted location: {args.target_class}"
            untargeted_srt = f"true location of the examples"
            adversary.logger.log(f"The following logs are relative to the location of the {targeted_srt if args.targeted else untargeted_srt}")
            
            if args.targeted:
                improvement = distances_clean - distances_adv
            else:
                improvement = distances_adv - distances_clean
            neg_mask = improvement <= 0
            pos_mask = improvement > 0
            distances_adv[neg_mask] = distances_clean[neg_mask] # filter bad adversarial attacks

            for T in thresholds:
                percent_T_clean = (distances_clean <= T).float().mean().item() * 100.0
                percent_T_adv = (distances_adv <= T).float().mean().item() * 100.0
                adversary.logger.log(f"Percentage of clean predictions within {labels[T]}: {percent_T_clean:.2f}%")
                adversary.logger.log(f"Percentage of adv predictions within {labels[T]}: {percent_T_adv:.2f}%")

            percent_improved = pos_mask.float().mean().item() * 100.0
            if pos_mask.sum() > 0:
                avg_improvement = improvement[pos_mask].mean().item()
                median_improvement = improvement[pos_mask].median().item()
            else:
                avg_improvement = 0.0
                median_improvement = 0.0
            adversary.logger.log(f"Percentage of examples with positivly changed distance: {percent_improved:.2f}%")
            adversary.logger.log(f"Average change in predicted distance (km): {avg_improvement:.2f}")
            adversary.logger.log(f"Median change in predicted distance (km): {median_improvement:.2f}")
        
        res = (adv_complete - x_test != 0.).max(dim=1)[0].sum(dim=(1, 2))
        adversary.logger.log('max L0 perturbation ({}) {:.0f} - nan in img {} - max img {:.5f} - min img {:.5f}'.format(
            str_space, res.max(), (adv_complete != adv_complete).sum(), adv_complete.max(), adv_complete.min()))
            
        ind_corrcl = pred == 1.
        ind_succ = (pred_adv == 0.) * (pred == 1.)
        
        str_stats = 'success rate={:.0f}/{:.0f} ({:.2%}) \n'.format(
            pred.sum() - pred_adv.sum(), pred.sum(), (pred.sum() - pred_adv.sum()).float() / pred.sum()) +\
            '[successful points] avg # queries {:.1f} - med # queries {:.1f}\n'.format(
            qr_complete[ind_succ].float().mean(), torch.median(qr_complete[ind_succ].float()))
        qr_complete[~ind_succ] = args.n_queries + 0
        str_stats += '[correctly classified points] avg # queries {:.1f} - med # queries {:.1f}\n'.format(
            qr_complete[ind_corrcl].float().mean(), torch.median(qr_complete[ind_corrcl].float()))
        adversary.logger.log(str_stats)

        if args.model.lower() == 'clip': # run adv photos on geoclip
            # check robust accuracy and other statistics
            model = GeoCLIP()
            model.to(device)
            model.eval()

            all_clean_dist = []
            all_adv_dist = []

            trans = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
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

                clean_dist = haversine_distance(output_clean, y_curr)
                adv_dist = haversine_distance(output_adv, y_curr)
                
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
            adversary.logger.log("GeoCLIP attack stats")
            adversary.logger.log("--------------------")
            targeted_srt = f"targeted location: {args.target_class}"
            untargeted_srt = f"true location of the examples"
            adversary.logger.log(f"The following logs are relative to the location of the {targeted_srt if args.targeted else untargeted_srt}")
            for T in thresholds:
                percent_T_clean = (distances_clean <= T).float().mean().item() * 100.0
                adversary.logger.log(f"Percentage of clean predictions within {labels[T]}: {percent_T_clean:.2f}%")
            for T in thresholds:
                percent_T_adv = (distances_adv <= T).float().mean().item() * 100.0
                adversary.logger.log(f"Percentage of adv predictions within {labels[T]}: {percent_T_adv:.2f}%")
            

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
            adversary.logger.log(f"Percentage of positivly changed examples distance: {percent_improved:.2f}%")
            adversary.logger.log(f"Average change in distance (km) from positivly predicted examples: {avg_improvement:.2f}")
            adversary.logger.log(f"Median change in distance (km) from positivly predicted examples: {median_improvement:.2f}")

            
            adversary.logger.log(f"Average change in distance (km) in overall predicted examples: {improvement.mean().item():.2f}")
            adversary.logger.log(f"Median change in distance (km) in overall predicted examples: {improvement.median().item():.2f}")

            
        

        # save results depending on the threat model
        if args.norm in ['L0', 'patches', 'frames']:
            if args.use_feature_space:
                # reshape perturbed images to original rgb format
                bs, _, h, w = adv_complete.shape
                adv_complete = adv_complete.view(bs, 3, h, w // 3)
            torch.save({'adv': adv_complete, 'qr': qr_complete},
                '{}/{}.pth'.format(savedir, param_run))
                
        elif args.norm in ['patches_universal']:
            # extract and save patch
            ind = (res > 0).nonzero().squeeze()[0]
            ind_patch = (((adv_complete[ind] - x_test[ind]).abs() > 0).max(0)[0] > 0).nonzero().squeeze()
            t = [ind_patch[:, 0].min().item(), ind_patch[:, 0].max().item(), ind_patch[:, 1].min().item(), ind_patch[:, 1].max().item()]
            loc = torch.tensor([t[0], t[2]])
            s = t[1] - t[0] + 1
            patch = adv_complete[ind, :, loc[0]:loc[0] + s, loc[1]:loc[1] + s].unsqueeze(0)
            
            torch.save({'adv': adv_complete, 'patch': patch},
                '{}/{}.pth'.format(savedir, param_run))

        elif args.norm in ['frames_universal']:
            # extract and save frame and indeces of the perturbed pixels
            # to easily apply the frame to new images
            ind_img = (res > 0).nonzero().squeeze()[0]
            mask = torch.zeros(x_test.shape[-2:])
            s = int(args.eps)
            mask[:s] = 1.
            mask[-s:] = 1.
            mask[:, :s] = 1.
            mask[:, -s:] = 1.
            ind = (mask == 1.).nonzero().squeeze()
            frame = adv_complete[ind_img, :, ind[:, 0], ind[:, 1]]
            
            torch.save({'adv': adv_complete, 'frame': frame, 'ind': ind},
                '{}/{}.pth'.format(savedir, param_run))
    
