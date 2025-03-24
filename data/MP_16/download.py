import os
import subprocess
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import urllib.request
import multiprocessing
from tqdm import tqdm
import time
import random

# Define constants
MP16_PLACES_CSV = "mp16_places365.csv"  # Downloaded by download_csv.sh
MP16_URLS_CSV = "mp16_urls.csv"  # Downloaded by download_images.sh
SAMPLED_CSV = "mp16_sampled.csv"  # Created by our sampling function

def get_transforms(apply_transforms=True):
    """
    Get image transforms for GeoCLIP
    This should match the preprocessing used by CLIP
    """
    if not apply_transforms:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
    
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def download_data(data_dir):
    """
    Ensure that the dataset metadata (CSV files) is present.
    Runs the shell scripts to download them if they are missing.
    """
    # Full absolute paths
    mp16_path = os.path.join(os.path.abspath(data_dir), 'MP_16')
    images_path = os.path.join(mp16_path, 'images')
    
    # Check for directories
    os.makedirs(images_path, exist_ok=True)
    
    # Check for CSV files with absolute paths
    places_csv_path = os.path.join(mp16_path, MP16_PLACES_CSV)
    urls_csv_path = os.path.join(mp16_path, MP16_URLS_CSV)
    
    # Download places CSV if needed
    if not os.path.exists(places_csv_path):
        print(f"Places CSV file not found at {places_csv_path}. Downloading...")
        # Use absolute path to script
        script_path = os.path.join(mp16_path, "download_csv.sh")
        if os.path.exists(script_path):
            # Save current directory
            cwd = os.getcwd()
            # Change to MP_16 directory to run script
            os.chdir(mp16_path)
            subprocess.run(["bash", "./download_csv.sh"])
            # Return to original directory
            os.chdir(cwd)
        else:
            print(f"Warning: Download script not found at {script_path}")
    
    # Download URLs CSV if needed
    if not os.path.exists(urls_csv_path):
        print(f"URLs CSV file not found at {urls_csv_path}. Downloading...")
        # Use absolute path to script
        script_path = os.path.join(mp16_path, "download_images.sh")
        if os.path.exists(script_path):
            # Save current directory
            cwd = os.getcwd()
            # Change to MP_16 directory to run script
            os.chdir(mp16_path)
            subprocess.run(["bash", "./download_images.sh"])
            # Return to original directory
            os.chdir(cwd)
        else:
            print(f"Warning: Download script not found at {script_path}")
    
    return mp16_path, images_path, places_csv_path, urls_csv_path

def download_image(args):
    """
    Download a single image from URL
    
    Args:
        args: tuple containing (url, filename, save_dir)
        
    Returns:
        success: bool indicating if download was successful
    """
    url, filename, save_dir = args
    output_path = os.path.join(save_dir, filename)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        return True
    
    try:
        # Add user agent to avoid being blocked
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        req = urllib.request.Request(url, headers=headers)
        
        # Download with timeout
        with urllib.request.urlopen(req, timeout=5) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())
        
        # Small delay to avoid overloading server
        time.sleep(0.1)
        return True
    except Exception as e:
        # Silently fail - some URLs are expected to be unavailable
        return False

def download_images_from_urls(urls_csv_path, images_path, max_images=None, num_workers=8):
    """
    Download images from URLs listed in the CSV file
    
    Args:
        urls_csv_path: Path to CSV with URLs
        images_path: Directory to save images
        max_images: Maximum number of images to download (None for all)
        num_workers: Number of parallel downloads
    
    Returns:
        available_images: List of successfully downloaded image filenames
    """
    # Load URLs CSV
    df = pd.read_csv(urls_csv_path, header=None, names=['IMG_PATH', 'URL'])
    print(f"Found {len(df)} images in URLs file")
    print(f"CSV columns: {df.columns.tolist()}")
    
    # Limit number of downloads if specified
    if max_images is not None:
        df = df.sample(min(max_images, len(df)), random_state=42)
    
    # Prepare download arguments
    download_args = []
    for _, row in df.iterrows():
        # Get image ID from the first column (path)
        image_id = row['IMG_PATH'].replace('/', '_') if '/' in row['IMG_PATH'] else row['IMG_PATH']
        download_args.append((row['URL'], image_id, images_path))
    
    # Download images in parallel
    print(f"Downloading {len(download_args)} images using {num_workers} workers...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(download_image, download_args), total=len(download_args)))
    
    # Count successful downloads
    success_count = sum(results)
    print(f"Successfully downloaded {success_count} out of {len(download_args)} images")
    
    # Return list of available image filenames
    available_images = [args[1] for args, success in zip(download_args, results) if success]
    return available_images

def merge_csv_data(places_csv_path, available_images, output_path):
    """
    Create a filtered CSV containing only information for available images
    
    Args:
        places_csv_path: Path to places CSV
        available_images: List of available image filenames
        output_path: Path to save the filtered CSV
    """
    # Load Places365 CSV
    places_df = pd.read_csv(places_csv_path)
    
    # Create set of available images for faster lookup
    available_set = set(available_images)
    
    # Filter to only include available images
    filtered_df = places_df[places_df['IMG_ID'].apply(
        lambda x: x.replace('/', '_') if '/' in x else x).isin(available_set)]
    
    # Update IMG_ID to match the filenames on disk
    filtered_df['IMG_ID'] = filtered_df['IMG_ID'].apply(
        lambda x: x.replace('/', '_') if '/' in x else x)
    
    # Save filtered CSV
    filtered_df.to_csv(output_path, index=False)
    print(f"Created filtered CSV with {len(filtered_df)} entries at {output_path}")
    
    return filtered_df

def sample_kmeans(csv_path, save_path, n_samples=200):
    """
    Sample a subset of the dataset using KMeans clustering on geographic coordinates.
    This helps to get a diverse set of locations while keeping the dataset size manageable.
    """
    if os.path.exists(save_path):
        print(f"Sampled dataset already exists at {save_path}")
        return pd.read_csv(save_path)
        
    print(f"Sampling {n_samples} diverse images from {csv_path} using KMeans clustering...")
    
    df = pd.read_csv(csv_path)
    
    # Use geographic coordinates and other features if available
    features = ['LAT', 'LON']
    if 'Prob_indoor' in df.columns:
        features.extend(['Prob_indoor', 'Prob_natural', 'Prob_urban'])
    
    X = df[features].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    k = min(n_samples, len(df))  # Can't have more clusters than data points
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    
    df['cluster_label'] = kmeans.labels_
    
    # Select the closest point to each centroid
    sampled_rows = []
    for cluster_label in range(k):
        cluster_indices = df.index[df['cluster_label'] == cluster_label]
        if len(cluster_indices) == 0:
            continue
        
        cluster_points_scaled = X_scaled[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_label]
        distances = np.linalg.norm(cluster_points_scaled - centroid, axis=1)
        closest_index = cluster_indices[distances.argmin()]
        sampled_rows.append(closest_index)
    
    # Create sampled dataframe
    sampled_df = df.loc[sampled_rows].copy()
    sampled_df.to_csv(save_path, index=False)
    print(f"Sampled dataset saved to {save_path} with {len(sampled_df)} images")
    
    return sampled_df

class MP16Dataset(Dataset):
    """
    Custom dataset for MP-16 that:
      - Reads filenames and their lat/lon from a CSV file.
      - Loads images from the directory.
      - Returns (transformed_image, (lat, lon)).
    """
    def __init__(self, images_dir, csv_path, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform if transform else get_transforms()
        
        self.data = pd.read_csv(csv_path)
        
        # Verify images exist
        self.valid_indices = []
        for idx, row in enumerate(self.data.iterrows()):
            img_path = os.path.join(self.images_dir, row[1]["IMG_ID"])
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
        
        print(f"Found {len(self.valid_indices)} valid images out of {len(self.data)}")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map to valid index
        actual_idx = self.valid_indices[idx]
        row = self.data.iloc[actual_idx]
        
        image_file = row["IMG_ID"]
        lat = row["LAT"]
        lon = row["LON"]
        
        # Full path to the image
        img_path = os.path.join(self.images_dir, image_file)
        
        # Load the image
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Handle multiple frames if needed
            if getattr(image, "n_frames", 1) > 1:
                image.seek(0)
                
            if self.transform:
                image = self.transform(image)
                
            target = torch.tensor([lat, lon], dtype=torch.float)
            
            # Return (image, (lat, lon))
            return image, target
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random dummy sample in case of error
            dummy_img = torch.rand(3, 224, 224)
            dummy_target = torch.tensor([0.0, 0.0], dtype=torch.float)
            return dummy_img, dummy_target
    
def load_mp16_data(data_dir, max_images=1000, transform=None, sample_size=None):
    """
    Load the MP-16 dataset.
    
    Args:
        data_dir: Root directory of the dataset
        max_images: Maximum number of images to download (None for all)
        transform: Optional transforms to apply to images
        sample_size: If provided, samples a smaller dataset using KMeans
        
    Returns:
        x_list: List of images
        y_list: List of coordinates
    """
    # Ensure CSV data is available and get paths
    mp16_path, images_path, places_csv_path, urls_csv_path = download_data(data_dir)
    
    # Download images from URLs (if needed)
    available_images = download_images_from_urls(
        urls_csv_path, images_path, max_images=max_images)
    
    if not available_images:
        raise ValueError("No images could be downloaded. Please check your internet connection.")
    
    # Create filtered CSV with only available images
    filtered_csv_path = os.path.join(mp16_path, "mp16_filtered.csv")
    filtered_df = merge_csv_data(places_csv_path, available_images, filtered_csv_path)
    
    # Sample dataset if requested
    csv_path = filtered_csv_path
    if sample_size and sample_size < len(filtered_df):
        sampled_csv_path = os.path.join(mp16_path, SAMPLED_CSV)
        sample_kmeans(filtered_csv_path, sampled_csv_path, n_samples=sample_size)
        csv_path = sampled_csv_path
    
    # Load data
    transform_fn = transform if transform else get_transforms()
    
    # Prepare data lists
    x_list = []
    y_list = []
    
    # Load each image
    for _, row in tqdm(pd.read_csv(csv_path).iterrows(), desc="Loading images"):
        image_file = row["IMG_ID"]
        lat = row["LAT"]
        lon = row["LON"]
        
        img_path = os.path.join(images_path, image_file)
        
        if not os.path.exists(img_path):
            continue
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Handle multiple frames if needed
            if getattr(image, "n_frames", 1) > 1:
                image.seek(0)
                
            # Apply transform if image loaded successfully
            if transform_fn:
                image = transform_fn(image)
                
            x_list.append(image)
            y_list.append([lat, lon])
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Skip this image
            continue
    
    print(f"Loaded {len(x_list)} images from MP-16 dataset")
    return x_list, y_list

def get_mp16_dataloader(data_dir, batch_size, max_images=1000, 
                      transform=None, shuffle=True, num_workers=2,
                      sample_size=None):
    """
    Get a DataLoader for the MP-16 dataset.
    """
    # Ensure data is downloaded/available
    mp16_path, images_path, places_csv_path, urls_csv_path = download_data(data_dir)
    
    # Download images from URLs (if needed)
    available_images = download_images_from_urls(
        urls_csv_path, images_path, max_images=max_images)
    
    if not available_images:
        raise ValueError("No images could be downloaded. Please check your internet connection.")
    
    # Create filtered CSV with only available images
    filtered_csv_path = os.path.join(mp16_path, "mp16_filtered.csv")
    filtered_df = merge_csv_data(places_csv_path, available_images, filtered_csv_path)
    
    # Sample dataset if requested
    csv_path = filtered_csv_path
    if sample_size and sample_size < len(filtered_df):
        sampled_csv_path = os.path.join(mp16_path, SAMPLED_CSV)
        sample_kmeans(filtered_csv_path, sampled_csv_path, n_samples=sample_size)
        csv_path = sampled_csv_path
    
    # Create dataset
    dataset = MP16Dataset(
        images_dir=images_path,
        csv_path=csv_path,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader 