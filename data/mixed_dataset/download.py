import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Add parent directory to path to import from other datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Im2GPS3k.download import get_transforms as get_im2gps_transforms
from MP_16.download import get_transforms as get_mp16_transforms

SAMPLED_CSV = "mixed_sampled.csv"

def get_transforms():
    """
    Get image transforms for GeoCLIP
    This should match the preprocessing used by CLIP
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def sample_kmeans(csv_path, save_path, n_samples=150):
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

class MixedDataset(Dataset):
    """
    Custom dataset that combines Im2GPS3k and MP_16 datasets:
      - Reads filenames and their lat/lon from a CSV file.
      - Loads images from the respective directories.
      - Returns (transformed_image, (lat, lon)).
    """
    def __init__(self, data_dir, csv_path, clip_varient, transform=None):
        super().__init__()
        self.transform = transform if transform else get_transforms()
        
        # Store paths to both datasets
        self.im2gps_dir = os.path.join(data_dir, 'Im2GPS3k', 'images')
        self.mp16_dir = os.path.join(data_dir, 'MP_16', 'images')
        self.clip_varient = clip_varient
        
        self.data = pd.read_csv(csv_path)
        
        # Verify images exist and can be loaded
        self.valid_indices = []
        valid_im2gps = 0
        valid_mp16 = 0
        
        for idx, row in enumerate(self.data.iterrows()):
            dataset = row[1]["dataset"]
            img_id = row[1]["IMG_ID"]
            
            # Handle different path formats
            if dataset == "mp16":
                # MP-16 images might have paths with slashes, convert to underscores
                img_id = img_id.replace('/', '_')
            
            img_path = os.path.join(
                self.im2gps_dir if dataset == "im2gps" else self.mp16_dir,
                img_id
            )
            
            try:
                if os.path.exists(img_path):
                    # Try to load the image to verify it's valid
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        self.valid_indices.append(idx)
                        if dataset == "im2gps":
                            valid_im2gps += 1
                        else:
                            valid_mp16 += 1
            except Exception as e:
                print(f"Error verifying image {img_path}: {e}")
                continue
        
        if len(self.valid_indices) == 0:
            raise RuntimeError("No valid images found in the dataset. Please check the image paths and files.")
            
        print(f"Found {len(self.valid_indices)} valid images out of {len(self.data)}:")
        print(f"  - Im2GPS3k: {valid_im2gps} images")
        print(f"  - MP-16: {valid_mp16} images")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map to valid index
        actual_idx = self.valid_indices[idx]
        row = self.data.iloc[actual_idx]
        
        dataset = row["dataset"]
        img_id = row["IMG_ID"]
        
        # Handle different path formats
        if dataset == "mp16":
            img_id = img_id.replace('/', '_')
            
        # Full path to the image
        img_path = os.path.join(
            self.im2gps_dir if dataset == "im2gps" else self.mp16_dir,
            img_id
        )
        
        # Load and process the image
        image = Image.open(img_path).convert("RGB")
        if getattr(image, "n_frames", 1) > 1:
            image.seek(0)
            
        if self.transform:
            image = self.transform(image)
            
        # Prepare targets
        lat = row["LAT"]
        lon = row["LON"]
        target = torch.tensor([lat, lon], dtype=torch.float)
        
        if self.clip_varient:
            label = torch.tensor(row["S365_Label"])
            return image, target, label
            
        return image, target

def create_mixed_dataset(data_dir, samples_per_dataset=75):
    """
    Create a mixed dataset by combining samples from both Im2GPS3k and MP_16.
    
    Args:
        data_dir: Root directory containing both datasets
        samples_per_dataset: Number of samples to take from each dataset
        
    Returns:
        DataFrame containing the mixed dataset
    """
    # Import data from both datasets
    from Im2GPS3k.download import download_data as download_im2gps
    from MP_16.download import (
        download_data as download_mp16,
        download_images_from_urls,
        download_image,
        sample_kmeans as sample_mp16
    )
    
    # Download/prepare both datasets
    im2gps_path, im2gps_images_path, im2gps_csv_path, _ = download_im2gps(data_dir)
    mp16_path, mp16_images_path, mp16_csv_path, mp16_urls_csv = download_mp16(data_dir)
    
    # # Download MP-16 images if needed
    # print("Checking and downloading MP-16 images...")
    # available_mp16_images = download_images_from_urls(mp16_urls_csv, mp16_images_path)
    # if not available_mp16_images:
    #     raise RuntimeError("Failed to download any MP-16 images. Please check your internet connection.")
    output_path = os.path.join(data_dir, 'mixed_dataset', SAMPLED_CSV)
    if os.path.exists(output_path):
        return pd.read_csv(output_path)
    # # Create sampled datasets if they don't exist
    mp16_sampled_path = os.path.join(mp16_path, "mp16_sampled.csv")
    # if not os.path.exists(mp16_sampled_path):
    #     print("Sampling MP-16 dataset...")
    #     sample_mp16(mp16_csv_path, mp16_sampled_path)
    
    # Load data from both datasets
    im2gps_df = pd.read_csv(os.path.join(im2gps_path, "sampled_data.csv"))
    mp16_df = pd.read_csv(mp16_sampled_path)  # Now using the correct path that we ensured exists
    
    valid_im2gps_rows = []
    for _, row in im2gps_df.iterrows():
        img_id = row["IMG_ID"]
        
        img_path = os.path.join(im2gps_images_path, img_id)
        if os.path.exists(img_path):
            valid_im2gps_rows.append(row)
    
    valid_im2gps_df = pd.DataFrame(valid_im2gps_rows)
    print(f"Found {len(valid_im2gps_df)} valid Im2GPS images on disk.")
    
    # Take the first samples_per_dataset (or fewer if not enough)
    valid_im2gps_df = valid_im2gps_df.head(samples_per_dataset)
    print(f"Taking {len(valid_im2gps_df)} from Im2GPS (wanted {samples_per_dataset}).")
    
    valid_mp16_rows = []
    for _, row in mp16_df.iterrows():
        img_id = row["IMG_ID"]
        
        img_path = os.path.join(mp16_images_path, img_id)
        if os.path.exists(img_path):
            valid_mp16_rows.append(row)
    
    valid_mp16_df = pd.DataFrame(valid_mp16_rows)
    print(f"Found {len(valid_mp16_df)} valid MP-16 images on disk.")
    
    # Take the first samples_per_dataset (or fewer if not enough)
    valid_mp16_df = valid_mp16_df.head(samples_per_dataset)
    print(f"Taking {len(valid_mp16_df)} from MP-16 (wanted {samples_per_dataset}).")
    

    # # Take first N samples from each dataset
    # im2gps_df = im2gps_df.head(samples_per_dataset)
    # mp16_df = mp16_df.head(samples_per_dataset)
    
    # Add dataset identifier
    valid_im2gps_df['dataset'] = 'im2gps'
    valid_mp16_df['dataset'] = 'mp16'
    
    # Combine datasets
    mixed_df = pd.concat([valid_im2gps_df, valid_mp16_df], ignore_index=True)    
    # Save combined dataset
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mixed_df.to_csv(output_path, index=False)
    
    return mixed_df

def get_mixed_dataloader(data_dir, batch_size, clip_varient=False, samples_per_dataset=75,
                        transform=None, shuffle=True, num_workers=2):
    """
    Get a DataLoader for the mixed dataset.
    """
    # Create mixed dataset if it doesn't exist
    csv_path = os.path.join(data_dir, 'mixed_dataset', SAMPLED_CSV)
    if not os.path.exists(csv_path):
        create_mixed_dataset(data_dir, samples_per_dataset)
    
    # Create dataset
    dataset = MixedDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        transform=transform,
        clip_varient=clip_varient
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader