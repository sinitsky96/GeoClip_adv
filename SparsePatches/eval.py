import os
import argparse
import torch
import numpy as np
from datetime import datetime
from torchvision import transforms
from tqdm import tqdm

from geoclip.model.GeoCLIP import GeoCLIP
from data.Im2GPS.download import load_im2gps_data, get_transforms as get_im2gps_transforms
from data.MP_16.download import load_mp16_data, get_transforms as get_mp16_transforms
from transformers import CLIPModel

from SparsePatches.attack_sparse_patches import AttackGeoCLIP_SparsePatches, AttackCLIP_SparsePatches, AttackGeoCLIP_SparsePatches_Kernel
from sparse_rs.util import haversine_distance, CONTINENT_R, STREET_R


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--attack_type', type=str, default='sparse', choices=['sparse', 'kernel'], 
                      help='Type of attack to use: "sparse" for regular sparse patches, "kernel" for kernel-based attack')
    parser.add_argument('--norm', type=str, default='L0')  # type of the attack: 'L0', 'patches'
    
    # Number of pixels to perturb
    parser.add_argument('--sparsity', default=224, type=int, help='Number of pixels to perturb (optimal for 224x224 images)')
    
    # Kernel parameters (used only with kernel attack)  
    parser.add_argument('--kernel_size', default=4, type=int, help='Size of kernel for kernel attack (used only with --attack_type kernel)')
    parser.add_argument('--kernel_sparsity', default=8, type=int, help='Sparsity within each kernel (used only with --attack_type kernel)')

    parser.add_argument('--n_restarts', type=int, default=10)  # Number of random restarts
    parser.add_argument('--loss', type=str, default='margin')  # loss function for the attack, options: 'margin', 'ce'
    parser.add_argument('--n_ex', type=int, default=20)  # dataset size
    parser.add_argument('--bs', type=int, default=32)  # batch size
    parser.add_argument('--n_iter', type=int, default=20)  # number of iterations
    parser.add_argument('--seed', type=int, default=42)
    
    # PGDTrim parameters
    parser.add_argument('--eps_l_inf', type=float, default=0.05)  # L_inf constraint
    
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--target_class', type=eval)

    parser.add_argument('--model', default='geoclip', type=str)
    parser.add_argument('--dataset', type=str, default='Im2GPS3k', choices=['Im2GPS', 'Im2GPS3k', 'YFCC26k', 'MP_16'])
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--max_images', type=int, default=1000, help='Maximum number of images to download for MP-16 dataset')

    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU: {}".format(torch.cuda.get_device_name(device)))
    else:
        print("Using CPU")

    cpu_device = torch.device("cpu")

    if args.dataset == 'Im2GPS':
        # Get transforms for preprocessing images
        transform = get_im2gps_transforms(apply_transforms=True)
        
        # Load data
        x_list, y_list = load_im2gps_data(args.data_path, transform=transform)
        
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

    elif args.dataset == 'YFCC26k':
        # Add YFCC26k data loading if needed
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

    # Load the model
    if args.model.lower() == "geoclip":
        model = GeoCLIP()
        model.to(device)
        model.eval()
    elif args.model.lower() == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model.to(device)
        model.eval()
    
    # Create log directories
    logsdir = '{}/logs_{}_{}'.format(args.save_dir, "sparse_patches", args.norm)
    savedir = '{}/{}_{}'.format(args.save_dir, "sparse_patches", args.norm)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(logsdir):
        os.makedirs(logsdir)
    
    # Set loss type
    if args.targeted:
        args.loss = 'ce'
    
    # Create parameter string for logging
    param_run = '{}_{}_{}_1_{}_iter_{}_eps_l_inf_{:.2f}_loss_{}_sparsity_{:.0f}_targeted_{}_targetclass_{}_seed_{:.0f}'.format(
        "sparse_patches", args.norm, args.model, args.n_ex, args.n_iter, args.eps_l_inf,
        args.loss, args.sparsity, args.targeted, args.target_class, args.seed)
    
    # Initialize the attack
    if args.attack_type == 'sparse':
        if args.model.lower() == "geoclip":
            adversary = AttackGeoCLIP_SparsePatches(
                model=model,
                norm=args.norm,
                sparsity=args.sparsity,
                eps_l_inf=args.eps_l_inf,
                n_iter=args.n_iter,
                n_restarts=args.n_restarts,
                targeted=args.targeted,
                loss=args.loss,
                device=device,
                verbose=True,
                seed=args.seed,
                log_path='{}/log_run_{}_{}.txt'.format(logsdir, str(datetime.now())[:-7], param_run)
            )
        elif args.model.lower() == "clip":
            adversary = AttackCLIP_SparsePatches(
                model=model,
                data_path=args.data_path,
                norm=args.norm,
                sparsity=args.sparsity,
                eps_l_inf=args.eps_l_inf,
                n_iter=args.n_iter,
                n_restarts=args.n_restarts,
                targeted=args.targeted,
                loss=args.loss,
                device=device,
                verbose=True,
                seed=args.seed,
                log_path='{}/log_run_{}_{}.txt'.format(logsdir, str(datetime.now())[:-7], param_run)
            )
            model = adversary.get_logits  # Use the wrapped model
    elif args.attack_type == 'kernel':
        adversary = AttackGeoCLIP_SparsePatches_Kernel(
            model=model,
            norm=args.norm,
            sparsity=args.sparsity,
            kernel_size=args.kernel_size,
            kernel_sparsity=args.kernel_sparsity,
            eps_l_inf=args.eps_l_inf,
            n_iter=args.n_iter,
            n_restarts=args.n_restarts,
            targeted=args.targeted,
            loss=args.loss,
            device=device,
            verbose=True,
            seed=args.seed,
            log_path='{}/log_run_{}_{}.txt'.format(logsdir, str(datetime.now())[:-7], param_run)
        )
    
    # Set target classes for targeted attacks
    if args.targeted:
        if args.target_class is None:
            raise ValueError('Expected a --target_class tuple argument. For example: --target_class "(37.090924,25.370521)"')
        target_tensor = torch.tensor(args.target_class, dtype=torch.float32)
        y_test = target_tensor.repeat(y_test.shape[0], 1)
        print('Target location:', args.target_class)
    
    # Configure batch processing
    bs = args.bs
    n_batches = int(np.ceil(n_examples / bs))
    adv_complete = x_test.clone()
    pred = torch.zeros([0]).float().cpu()
    
    print("Starting clean classification")
    
    with torch.no_grad():
        # Find points originally correctly classified
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, n_examples)

            x_curr = x_test[start_idx:end_idx].clone().detach().to(device)
            y_curr = y_test[start_idx:end_idx].clone().detach().to(device)

            if args.model.lower() == "geoclip":
                output, _ = model.predict_from_tensor(x_curr)
            else:  # CLIP
                output = model(x_curr)

            # Keep output on the same device as y_curr for consistent device handling
            output = output.to(device=device)
            
            if args.model.lower() == "geoclip":
                if not args.targeted:
                    pred = torch.cat((pred, (haversine_distance(output, y_curr) <= CONTINENT_R).float().to(cpu_device)), dim=0)
                else:
                    pred = torch.cat((pred, (haversine_distance(output, y_curr) > STREET_R).float().to(cpu_device)), dim=0)
            else:
                if not args.targeted:
                    pred = torch.cat((pred, (output.max(1)[1] == y_curr).float().to(cpu_device)), dim=0)
                else:
                    pred = torch.cat((pred, (output.max(1)[1] != y_curr).float().to(cpu_device)), dim=0)

            del x_curr
            del y_curr
            del output
            torch.cuda.empty_cache()
        
        print('Clean accuracy: {:.2%}'.format(pred.mean()))
        print("Finished clean classification")
        
        # Process only correctly classified examples
        n_batches = int(np.ceil(pred.sum() / bs))
        
        if args.model.lower() == "geoclip":
            ind_to_fool = (pred == 1).nonzero(as_tuple=True)[0]
        else:
            ind_to_fool = (pred == 1).nonzero().squeeze()
        
        # Run the attack
        pred_adv = pred.clone()
        
        print(f"\nRunning attack on {len(ind_to_fool)} correctly classified examples")
        print(f"Attack type: {args.attack_type}, Norm: {args.norm}, Sparsity: {args.sparsity}")
        if args.attack_type == 'kernel':
            print(f"Kernel size: {args.kernel_size}, Kernel sparsity: {args.kernel_sparsity}")
        print(f"Number of batches: {n_batches}, Batch size: {bs}")
        
        progress_bar = tqdm(range(n_batches), desc="Attack progress", leave=True)
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, len(ind_to_fool))

            x_curr = x_test[ind_to_fool[start_idx:end_idx]].clone().detach().to(device)
            y_curr = y_test[ind_to_fool[start_idx:end_idx]].clone().detach().to(device)

            # Run the attack to get adversarial examples
            adv = adversary.perturb(x_curr, y_curr)
            
            adv_complete[ind_to_fool[start_idx:end_idx]] = adv.detach().to(cpu_device)
            
            if args.model.lower() == "geoclip":
                output, _ = model.predict_from_tensor(adv)
                
                if not args.targeted:
                    pred_adv[ind_to_fool[start_idx:end_idx]] = (haversine_distance(output, y_curr) <= CONTINENT_R).float().to(cpu_device)
                else:
                    pred_adv[ind_to_fool[start_idx:end_idx]] = (haversine_distance(output, y_curr) > STREET_R).float().to(cpu_device)
            else:
                with torch.no_grad():
                    output = model(adv)
                
                if not args.targeted:
                    pred_adv[ind_to_fool[start_idx:end_idx]] = (output.max(1)[1] == y_curr).float().to(cpu_device)
                else:
                    pred_adv[ind_to_fool[start_idx:end_idx]] = (output.max(1)[1] != y_curr).float().to(cpu_device)
            
            # Update progress bar with current success rate
            current_success = 1.0 - pred_adv[ind_to_fool[:end_idx]].mean().item()
            progress_bar.set_postfix({"Success Rate": f"{current_success:.2%}"})
            progress_bar.update(1)
            
            del adv
            del x_curr
            del y_curr
            torch.cuda.empty_cache()
        
        progress_bar.close()
        
        robust_accuracy = pred_adv.mean()
        attack_success_rate = 1.0 - robust_accuracy
        print(f'Attack Success Rate: {attack_success_rate:.2%}')
        print('Robust accuracy: {:.2%}'.format(robust_accuracy))
        
        # Save results
        torch.save(adv_complete, '{}/adv_complete_{}.pt'.format(savedir, param_run))
        
        print('Done!') 