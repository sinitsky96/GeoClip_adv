import os
import argparse
import torch
import numpy as np
from datetime import datetime
from torchvision import transforms
from tqdm import tqdm

# Model imports
from geoclip.model.GeoCLIP import GeoCLIP
from transformers import CLIPModel

# Data imports
from data.Im2GPS3k.download import load_im2gps_data, get_transforms as get_im2gps_transforms
from data.MP_16.download import load_mp16_data, get_transforms as get_mp16_transforms
from data.mixed_dataset.download import get_mixed_dataloader, get_transforms as get_mixed_transforms

# Attack imports
from SparsePatches.attack_sparse_patches import AttackGeoCLIP_SparsePatches, AttackCLIP_SparsePatches, AttackGeoCLIP_SparsePatches_Kernel, AttackCLIP_SparsePatches_Kernel
try:
    from sparse_rs.util import haversine_distance, CONTINENT_R, STREET_R
except ImportError:
    # Define fallback implementations
    def haversine_distance(point1, point2):
        """Fallback haversine distance implementation"""
        device = point1.device
        # Convert degrees to radians
        lat1, lon1 = point1[..., 0] * np.pi / 180, point1[..., 1] * np.pi / 180
        lat2, lon2 = point2[..., 0] * np.pi / 180, point2[..., 1] * np.pi / 180
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        # Radius of earth in kilometers = 6371
        distance = 6371 * c
        return distance
    
    # Default thresholds
    CONTINENT_R = 2500.0  # 2500 km
    STREET_R = 1.0        # 1 km


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
    parser.add_argument('--loss', type=str, default='margin', choices=['margin', 'ce'])  # loss function for the attack, options: 'margin', 'ce'
    parser.add_argument('--n_ex', type=int, default=20)  # dataset size
    parser.add_argument('--bs', type=int, default=32)  # batch size
    parser.add_argument('--n_iter', type=int, default=20)  # number of iterations
    parser.add_argument('--seed', type=int, default=42)
    
    # PGDTrim parameters
    parser.add_argument('--eps_l_inf', type=float, default=0.05)  # L_inf constraint
    
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--target_class', type=eval)

    parser.add_argument('--model', default='geoclip', type=str)
    parser.add_argument('--dataset', type=str, default='Im2GPS3k', choices=['Im2GPS', 'Im2GPS3k', 'YFCC26k', 'MP_16', 'mixed'])
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--max_images', type=int, default=1000, help='Maximum number of images to download for MP-16 dataset')
    parser.add_argument('--samples_per_dataset', type=int, default=75, help='Number of samples to take from each dataset for mixed dataset')

    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Add debugging information
    print("\nCUDA Debug Info:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device requested: {args.device}")

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

    elif args.dataset == 'mixed':
        # Get transforms for preprocessing images
        transform = get_mixed_transforms()
        
        # Load data using the mixed dataset dataloader
        print(f"Loading mixed dataset with {args.samples_per_dataset} samples per dataset")
        dataloader = get_mixed_dataloader(
            args.data_path,
            batch_size=args.bs,
            samples_per_dataset=args.samples_per_dataset,
            transform=transform,
            clip_varient=True if args.model.lower() == 'clip' else False
        )
        
        # Convert dataloader to tensors
        if args.model.lower() == 'clip':
            # Special handling for CLIP model
            x_tensors = []
            y_tensors = []
            labels = []
            for x, y, label in dataloader:
                x_tensors.append(x)
                y_tensors.append(y)
                labels.append(label)
            
            x_test = torch.cat(x_tensors, dim=0)
            y_test = torch.cat(labels, dim=0)
            y_test_geo = torch.cat(y_tensors, dim=0)  # Keep GPS coords for cross-model evaluation
        else:
            # Regular GeoCLIP handling
            x_tensors = []
            y_tensors = []
            for x, y in dataloader:
                x_tensors.append(x)
                y_tensors.append(y)
            
            x_test = torch.cat(x_tensors, dim=0)
            y_test = torch.cat(y_tensors, dim=0)
        
        n_examples = len(x_test)
        args.n_ex = min(args.n_ex, n_examples)
        print("x_test shape: {}, y_test shape: {}".format(x_test.shape, y_test.shape))

    # Load the model
    if args.model.lower() == "geoclip":
        model = GeoCLIP()
        model.to(device)
        model.eval()
    elif args.model.lower() == "clip":
        # Load CLIP model
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model.to(device)
        model.eval()
        # Note: We don't wrap model in ClipWrap here - the AttackCLIP_SparsePatches class will handle it

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
    
    # Add kernel size to param_run if using kernel attack
    if args.attack_type == 'kernel':
        param_run = '{}_kernel_size_{}'.format(param_run, args.kernel_size)
    
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
            # Replace model with the adversary's callable interface
            model = adversary  # Use the adversary directly, which is callable via __call__
    elif args.attack_type == 'kernel':
        if args.model.lower() == "geoclip":
            adversary = AttackGeoCLIP_SparsePatches_Kernel(
                model=model,
                norm=args.norm,
                sparsity=args.sparsity,
                kernel_size=args.kernel_size,
                kernel_sparsity=args.kernel_size*args.kernel_size,
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
            adversary = AttackCLIP_SparsePatches_Kernel(
                model=model,
                data_path=args.data_path,
                norm=args.norm,
                sparsity=args.sparsity,
                kernel_size=args.kernel_size,
                kernel_sparsity=args.kernel_size*args.kernel_size,
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
            # Replace model with the adversary's callable interface
            model = adversary  # Use the adversary directly, which is callable via __call__
    
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
        
        # Check if there are any correctly classified examples
        if pred.sum() == 0:
            print("\nNo correctly classified examples to attack. Skipping attack phase.")
            # Save empty results
            torch.save(adv_complete, '{}/adv_complete_{}.pt'.format(savedir, param_run))
            print('\nDone!')
        else:
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
                print(f"Kernel size: {args.kernel_size}, Kernel sparsity: {args.kernel_size*args.kernel_size}")
            print(f"Number of batches: {n_batches}, Batch size: {bs}")
            
            # Store all distances for final statistics
            clean_distances_all = []  # For all examples
            clean_distances_attacked = []  # Only for attacked examples
            adv_distances = []
            
            # First collect clean prediction distances for all examples
            with torch.no_grad():
                for batch_idx in range(int(np.ceil(n_examples / bs))):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, n_examples)

                    x_curr = x_test[start_idx:end_idx].clone().detach().to(device)
                    y_curr = y_test[start_idx:end_idx].clone().detach().to(device)

                    if args.model.lower() == "geoclip":
                        output, _ = model.predict_from_tensor(x_curr)
                        distances = haversine_distance(output, y_curr)
                        clean_distances_all.append(distances.cpu())

            # Get clean distances for attacked examples only
            with torch.no_grad():
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, len(ind_to_fool))

                    x_curr = x_test[ind_to_fool[start_idx:end_idx]].clone().detach().to(device)
                    y_curr = y_test[ind_to_fool[start_idx:end_idx]].clone().detach().to(device)

                    if args.model.lower() == "geoclip":
                        output, _ = model.predict_from_tensor(x_curr)
                        distances = haversine_distance(output, y_curr)
                        clean_distances_attacked.append(distances.cpu())

            # Run attack and collect adversarial distances
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
                    distances = haversine_distance(output, y_curr)
                    adv_distances.append(distances.cpu())
                    
                    if not args.targeted:
                        batch_success = (distances > CONTINENT_R)
                        pred_adv[ind_to_fool[start_idx:end_idx]] = (~batch_success).float().to(cpu_device)
                    else:
                        batch_success = (distances <= STREET_R)
                        pred_adv[ind_to_fool[start_idx:end_idx]] = (~batch_success).float().to(cpu_device)
                else:  # CLIP model
                    output = model(adv)
                    if not args.targeted:
                        batch_success = (output.max(1)[1] != y_curr)
                        pred_adv[ind_to_fool[start_idx:end_idx]] = (~batch_success).float().to(cpu_device)
                    else:
                        batch_success = (output.max(1)[1] == y_curr)
                        pred_adv[ind_to_fool[start_idx:end_idx]] = (~batch_success).float().to(cpu_device)
                
                # Update progress bar
                current_success = 1.0 - pred_adv[ind_to_fool[:end_idx]].mean().item()
                progress_bar.set_postfix({"Success Rate": f"{current_success:.2%}"})
                progress_bar.update(1)
                
                del adv
                del x_curr
                del y_curr
                torch.cuda.empty_cache()
            
            progress_bar.close()
            
            # Combine all distances
            if clean_distances_all:  # Check if list is not empty
                clean_distances_all = torch.cat(clean_distances_all)
            else:
                clean_distances_all = torch.tensor([])
            
            if clean_distances_attacked:  # Check if list is not empty
                clean_distances_attacked = torch.cat(clean_distances_attacked)
            else:
                clean_distances_attacked = torch.tensor([])
            
            if adv_distances:  # Check if list is not empty
                adv_distances = torch.cat(adv_distances)
            else:
                adv_distances = torch.tensor([])
            
            # Calculate statistics for GeoCLIP
            if args.model.lower() == "geoclip" and len(clean_distances_all) > 0:
                STREET_R = 1.0  # 1 km
                CITY_R = 25.0   # 25 km
                REGION_R = 200.0  # 200 km
                COUNTRY_R = 750.0  # 750 km
                CONTINENT_R = 2500.0  # 2500 km
                
                # Function to calculate percentage within radius
                def calc_percentage(distances, radius):
                    return (distances <= radius).float().mean().item() * 100

                # Print clean prediction statistics (for all examples)
                print("\nClean Prediction Statistics (All Examples):")
                print(f"Percentage of clean predictions within STREET_R (1 km): {calc_percentage(clean_distances_all, STREET_R):.2f}%")
                print(f"Percentage of clean predictions within CITY_R (25 km): {calc_percentage(clean_distances_all, CITY_R):.2f}%")
                print(f"Percentage of clean predictions within REGION_R (200 km): {calc_percentage(clean_distances_all, REGION_R):.2f}%")
                print(f"Percentage of clean predictions within COUNTRY_R (750 km): {calc_percentage(clean_distances_all, COUNTRY_R):.2f}%")
                print(f"Percentage of clean predictions within CONTINENT_R (2500 km): {calc_percentage(clean_distances_all, CONTINENT_R):.2f}%")

                # Print adversarial prediction statistics (only for attacked examples)
                if len(adv_distances) > 0:
                    print("\nAdversarial Prediction Statistics (Attacked Examples):")
                    print(f"Percentage of adv predictions within STREET_R (1 km): {calc_percentage(adv_distances, STREET_R):.2f}%")
                    print(f"Percentage of adv predictions within CITY_R (25 km): {calc_percentage(adv_distances, CITY_R):.2f}%")
                    print(f"Percentage of adv predictions within REGION_R (200 km): {calc_percentage(adv_distances, REGION_R):.2f}%")
                    print(f"Percentage of adv predictions within COUNTRY_R (750 km): {calc_percentage(adv_distances, COUNTRY_R):.2f}%")
                    print(f"Percentage of adv predictions within CONTINENT_R (2500 km): {calc_percentage(adv_distances, CONTINENT_R):.2f}%")

                    # Calculate percentage of positively changed examples (comparing only attacked examples)
                    improved_mask = adv_distances < clean_distances_attacked
                    percent_improved = improved_mask.float().mean().item() * 100
                    print(f"\nPercentage of positively changed examples distance: {percent_improved:.2f}%")
            
            # Save results
            torch.save(adv_complete, '{}/adv_complete_{}.pt'.format(savedir, param_run))
            
            print('\nDone!')

    # Add cross-model evaluation after the main attack
    if args.model.lower() == "clip" and pred.sum() > 0:  # Only run if there were correctly classified examples
        # After running the attack, evaluate adversarial examples on GeoCLIP model
        print("\nRunning cross-model evaluation (CLIP adversarial examples on GeoCLIP)...")
        
        # Load GeoCLIP model for evaluation
        geoclip_model = GeoCLIP()
        geoclip_model.to(device)
        geoclip_model.eval()
        
        all_clean_dist = []
        all_adv_dist = []
        
        # Add normalization if needed for GeoCLIP (adjust if required)
        trans = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        with torch.no_grad():
            for batch_idx in range(int(np.ceil(n_examples / bs))):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, n_examples)
                
                x_curr_clean = x_test[start_idx:end_idx].clone().detach().to(device)
                x_curr_adv = adv_complete[start_idx:end_idx].clone().detach().to(device)
                y_curr = y_test_geo[start_idx:end_idx].clone().detach().to(device)
                
                # Filter images that were actually modified
                diff_imgs = (x_curr_clean - x_curr_adv).abs().view(x_curr_clean.shape[0], -1).max(dim=1)[0]
                threshold = 1e-6  # images with maximum difference below this are considered unaltered
                successful_mask = diff_imgs > threshold
                
                if successful_mask.sum() == 0:
                    continue
                
                x_curr_clean = x_curr_clean[successful_mask]
                x_curr_adv = x_curr_adv[successful_mask]
                y_curr = y_curr[successful_mask]
                
                # Get GeoCLIP predictions
                output_clean, _ = geoclip_model.predict_from_tensor(x_curr_clean)
                output_adv, _ = geoclip_model.predict_from_tensor(x_curr_adv)
                
                # Calculate distances
                clean_dist = haversine_distance(output_clean, y_curr)
                adv_dist = haversine_distance(output_adv, y_curr)
                
                all_clean_dist.append(clean_dist.cpu())
                all_adv_dist.append(adv_dist.cpu())
        
        if all_clean_dist and all_adv_dist:  # Only if we have data to analyze
            # Combine all distances
            clean_distances = torch.cat(all_clean_dist)
            adv_distances = torch.cat(all_adv_dist)
            
            # Define thresholds
            thresholds = [STREET_R, CITY_R, REGION_R, COUNTRY_R, CONTINENT_R]
            labels = {
                STREET_R: "STREET_R (1 km)",
                CITY_R: "CITY_R (25 km)",
                REGION_R: "REGION_R (200 km)",
                COUNTRY_R: "COUNTRY_R (750 km)",
                CONTINENT_R: "CONTINENT_R (2500 km)"
            }
            
            # Print cross-model evaluation statistics
            print("\nGeoCLIP Evaluation Statistics:")
            print("------------------------------")
            
            for T in thresholds:
                percent_T_clean = (clean_distances <= T).float().mean().item() * 100.0
                print(f"Percentage of clean predictions within {labels[T]}: {percent_T_clean:.2f}%")
            
            for T in thresholds:
                percent_T_adv = (adv_distances <= T).float().mean().item() * 100.0
                print(f"Percentage of adv predictions within {labels[T]}: {percent_T_adv:.2f}%")
            
            if args.targeted:
                improvement = clean_distances - adv_distances
            else:
                improvement = adv_distances - clean_distances
            
            pos_mask = improvement > 0
            percent_improved = pos_mask.float().mean().item() * 100.0
            if pos_mask.sum() > 0:
                avg_improvement = improvement[pos_mask].mean().item()
                median_improvement = improvement[pos_mask].median().item()
            else:
                avg_improvement = 0.0
                median_improvement = 0.0
                
            print(f"Percentage of positively changed examples distance: {percent_improved:.2f}%")
            print(f"Average change in distance (km) from positively predicted examples: {avg_improvement:.2f}")
            print(f"Median change in distance (km) from positively predicted examples: {median_improvement:.2f}")
            print(f"Average change in distance (km) in overall predicted examples: {improvement.mean().item():.2f}")
            print(f"Median change in distance (km) in overall predicted examples: {improvement.median().item():.2f}")
    elif args.model.lower() == "clip":
        print("\nSkipping cross-model evaluation as there were no correctly classified examples.") 