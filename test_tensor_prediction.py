import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.abspath('.'))

# Import GeoCLIP model
from geoclip.model.GeoCLIP import GeoCLIP

# Function to calculate distance between GPS coordinates (in km)
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def compute_errors(pred_coords, true_coords):
    """
    Compute error metrics between predicted and true coordinates
    
    Args:
        pred_coords: Tensor of predicted coordinates (batch_size, 2)
        true_coords: Tensor of true coordinates (batch_size, 2)
        
    Returns:
        Dictionary of error metrics
    """
    errors = {}
    
    # Convert to numpy for easier calculation
    pred_coords = pred_coords.cpu().numpy()
    true_coords = true_coords.cpu().numpy()
    
    # Calculate distances
    distances = []
    for i in range(len(pred_coords)):
        dist = haversine_distance(
            pred_coords[i][0], pred_coords[i][1],
            true_coords[i][0], true_coords[i][1]
        )
        distances.append(dist)
    
    # Calculate error metrics
    errors['mean_distance'] = np.mean(distances)
    errors['median_distance'] = np.median(distances)
    errors['min_distance'] = np.min(distances)
    errors['max_distance'] = np.max(distances)
    
    # Calculate accuracy at different thresholds
    thresholds = [25, 200, 750, 2500]
    for threshold in thresholds:
        accuracy = np.mean([1 if d <= threshold else 0 for d in distances])
        errors[f'accuracy_{threshold}km'] = accuracy
    
    return errors, distances

def visualize_prediction(image_tensor, true_coords, pred_coords, idx, output_dir=None, is_raw=True):
    """
    Visualize the original image and its prediction on a map
    
    Args:
        image_tensor: Tensor of the image
        true_coords: True coordinates (lat, lon)
        pred_coords: Predicted coordinates (lat, lon)
        idx: Index of the image
        output_dir: Directory to save visualization
        is_raw: Whether the image tensor is raw (not normalized)
    """
    # If normalized, denormalize the image tensor
    if not is_raw:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = image_tensor.cpu() * std + mean
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
    else:
        # For raw tensors, just convert to numpy (already in [0, 1] range)
        img = image_tensor.cpu().permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(12, 6))
    
    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Test Image {idx}")
    plt.axis('off')
    
    # Create a simple map visualization using matplotlib
    plt.subplot(1, 2, 2)
    # World map coordinates
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    
    # Plot equator and prime meridian
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add continent outlines (simplistic)
    plt.grid(alpha=0.3)
    
    # Plot true and predicted locations
    plt.scatter(true_coords[1], true_coords[0], c='green', marker='o', s=100, label='True Location')
    plt.scatter(pred_coords[1], pred_coords[0], c='red', marker='x', s=100, label='Predicted Location')
    
    dist = haversine_distance(true_coords[0], true_coords[1], pred_coords[0], pred_coords[1])
    plt.title(f"Prediction Error: {dist:.1f} km")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"prediction_{idx}.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# This function directly loads the preprocessed tensors from the specified location
def load_tensors_direct(tensor_dir):
    """
    Load preprocessed tensors directly from the specified directory
    
    Args:
        tensor_dir: Directory containing the preprocessed tensors
        
    Returns:
        Tuple of (images_tensor, coords_tensor)
        images_tensor could be a tensor or a dictionary of {idx: tensor}
    """
    images_tensor_path = os.path.join(tensor_dir, 'images_tensor.pt')
    coords_tensor_path = os.path.join(tensor_dir, 'coords_tensor.pt')
    
    if not os.path.exists(images_tensor_path) or not os.path.exists(coords_tensor_path):
        raise FileNotFoundError(f"Tensor files not found in {tensor_dir}")
    
    images_tensor = torch.load(images_tensor_path)
    coords_tensor = torch.load(coords_tensor_path)
    
    return images_tensor, coords_tensor

def main(args):
    # 1. Load GeoCLIP model
    print("Loading GeoCLIP model...")
    model = GeoCLIP(from_pretrained=True)
    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    
    # 2. Load preprocessed tensors directly from the specified path
    print("Loading raw (non-normalized) tensors...")
    tensor_dir = os.path.join(args.data_dir, 'Im2GPS', 'preprocessed')
    
    try:
        images_tensor, coords_tensor = load_tensors_direct(tensor_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure the tensor files exist at: {tensor_dir}")
        return
    
    # Check if images_tensor is a dictionary (raw tensors) or a tensor (normalized)
    is_dict_format = isinstance(images_tensor, dict)
    
    if is_dict_format:
        print(f"Loaded {len(images_tensor)} raw image tensors (dictionary format)")
        # Get the keys for indexing
        tensor_keys = list(images_tensor.keys())
    else:
        print(f"Loaded {len(images_tensor)} image tensors with shape {images_tensor.shape}")
    
    print(f"Loaded {len(coords_tensor)} coordinate pairs with shape {coords_tensor.shape}")
    
    # 4. Make predictions and calculate error metrics
    print(f"Making predictions for {args.num_test} images...")
    total_examples = min(args.num_test, len(coords_tensor))
    
    # Store predictions
    if is_dict_format:
        # For dictionary format, select from the available keys
        test_indices = np.random.choice(tensor_keys, total_examples, replace=False)
    else:
        # For tensor format, select from indices
        test_indices = np.random.choice(len(images_tensor), total_examples, replace=False)
    
    pred_coords_list = []
    
    # Create output directory if visualizing
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Make predictions
    for i, idx in enumerate(tqdm(test_indices)):
        # Get image and ground truth
        if is_dict_format:
            # For dictionary format
            image = images_tensor[idx]
        else:
            # For tensor format
            image = images_tensor[idx]
            
        true_coords = coords_tensor[idx]
        
        # Predict - Note that we're setting apply_transforms=True since we're using raw tensors
        pred_coords, pred_probs = model.predict_from_tensor(image, args.top_k, apply_transforms=True)
        top1_pred = pred_coords[0]
        pred_coords_list.append(top1_pred)
        
        # Visualize if requested
        if args.visualize and i < args.num_visualize:
            visualize_prediction(
                image, 
                true_coords.cpu().numpy(), 
                top1_pred.cpu().numpy(), 
                idx,
                args.output_dir,
                is_raw=True  # Indicate that we're using raw tensors
            )
    
    # Stack predictions for computation
    pred_coords_tensor = torch.stack(pred_coords_list)
    
    # For test true coords, we need to select the right indices from the coords tensor
    test_true_coords = coords_tensor[test_indices]
    
    # Calculate metrics
    errors, distances = compute_errors(pred_coords_tensor, test_true_coords)
    
    # Print results
    print("\nPrediction Results:")
    print(f"Mean distance error: {errors['mean_distance']:.2f} km")
    print(f"Median distance error: {errors['median_distance']:.2f} km")
    print(f"Min distance error: {errors['min_distance']:.2f} km")
    print(f"Max distance error: {errors['max_distance']:.2f} km")
    print("\nAccuracy at different thresholds:")
    print(f"Street level (25km): {errors['accuracy_25km']*100:.2f}%")
    print(f"City level (200km): {errors['accuracy_200km']*100:.2f}%")
    print(f"Region level (750km): {errors['accuracy_750km']*100:.2f}%")
    print(f"Country level (2500km): {errors['accuracy_2500km']*100:.2f}%")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50)
    plt.xlabel('Distance Error (km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(alpha=0.3)
    
    if args.visualize:
        plt.savefig(os.path.join(args.output_dir, "error_distribution.png"))
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GeoCLIP Tensor Prediction with Raw Tensors")
    parser.add_argument("--data_dir", type=str, default="D:/Study Docs/Degree Material/Sem 9 proj/GeoClip_adv/geoclip_adv_attacks/data", 
                        help="Directory containing Im2GPS data")
    parser.add_argument("--output_dir", type=str, default="D:\Study Docs\Degree Material\Sem 9 proj\GeoClip_adv\outputs", 
                        help="Directory to save visualization results")
    parser.add_argument("--num_test", type=int, default=10, 
                        help="Number of images to test")
    parser.add_argument("--max_samples", type=int, default=-1, 
                        help="Maximum number of samples to preprocess (-1 for all)")
    parser.add_argument("--top_k", type=int, default=5, 
                        help="Number of top predictions to return")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize predictions")
    parser.add_argument("--num_visualize", type=int, default=10, 
                        help="Number of predictions to visualize")
    parser.add_argument("--cuda", action="store_true", 
                        help="Use CUDA if available")
    
    args = parser.parse_args()
    main(args)