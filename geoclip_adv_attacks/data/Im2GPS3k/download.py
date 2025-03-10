import os
import subprocess
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

def download_data(args):
    """
    Ensure that the dataset (images + CSV files) is present.
    Run the shell scripts to download them if they are missing.
    """
    images_path = os.path.join(args.data_dir, 'Im2GPS3k', 'images')
    csv_path_1 = os.path.join(args.data_dir, 'Im2GPS3k', 'im2gps3k_places365.csv')
    csv_path_2 = os.path.join(args.data_dir, 'Im2GPS3k', 'im2gps_places365.csv')
    download_images_path = os.path.join(args.data_dir, 'Im2GPS3k', 'download_images.sh')
    download_csv_path = os.path.join(args.data_dir, 'Im2GPS3k', 'download_csv.sh')

    # 1. Download images if they don't exist
    if not os.path.exists(images_path):
        print("Images directory not found. Downloading dataset...")
        subprocess.run(["bash", download_images_path])

    # 2. Download CSVs if they don't exist
    if not os.path.exists(csv_path_1) or not os.path.exists(csv_path_2):
        print("CSV file(s) not found. Downloading CSVs...")
        subprocess.run(["bash", download_csv_path])


class Im2GPSDataset(Dataset):
    """
    Custom dataset that:
      - Reads filenames and their lat/lon from a CSV file.
      - Loads images from the directory.
      - Returns (transformed_image, (lat, lon)).
    """
    def __init__(self, images_dir, csv_path, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform
        
        self.data = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_file = row["IMG_ID"]  
        
        lat = row["LAT"]  
        lon = row["LON"]  
        
        # Full path to the image
        img_path = os.path.join(self.images_dir, image_file)
        
        # Load the image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Return (image, (lat, lon)) 
        return image, (lat, lon)


def get_im2gps_dataloader(args, 
                          transform=None, 
                          shuffle=True, 
                          num_workers=4,
                          csv_version='im2gps3k_places365.csv'):
    """
    Main function to:
      - Ensure data is downloaded (images + CSV).
      - Construct the Dataset.
      - Return a DataLoader.
    
    Args:
      args: Object holding necessary paths in `args.data_dir`.
      transform: Any torchvision transforms to be applied to images.
      batch_size: Batch size for the DataLoader.
      shuffle: Whether to shuffle the data.
      num_workers: Number of worker processes for loading.
      csv_version: Which CSV file to read from. You can choose
                   'im2gps3k_places365.csv' or 'im2gps_places365.csv'.
    """
    # 1. Make sure the data is available
    download_data(args)
    batch_size = args.batch_size
    
    # 2. Paths
    images_path = os.path.join(args.data_dir, 'Im2GPS3k', 'images')
    csv_path = os.path.join(args.data_dir, 'Im2GPS3k', csv_version)
    
    # 3. Create the dataset
    dataset = Im2GPSDataset(
        images_dir=images_path,
        csv_path=csv_path,
        transform=transform
    )
    
    # 4. Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


def load_im2gps_data(args, transform=None, csv_version='im2gps3k_places365.csv'):
    """
    Make sure data is available, then load images and their lat/lon from CSV into memory.
    
    Returns:
      X: list of image tensors (or PIL images if you prefer not to convert yet).
      y: list of (lat, lon) tuples.
    """
    # 1. Ensure the data is downloaded
    download_data(args)
    
    # 2. Set paths
    images_dir = os.path.join(args.data_dir, 'Im2GPS3k', 'images')
    csv_path = os.path.join(args.data_dir, 'Im2GPS3k', csv_version)
    
    # 3. Read the CSV
    df = pd.read_csv(csv_path)

    # 4. Prepare output containers
    X = []
    y = []
    
    # 5. Loop through each row, load images, store lat/lon
    for idx, row in df.iterrows():
        image_file = row["IMG_ID"]  
        lat = row["LAT"]  
        lon = row["LON"]          
        
        img_path = os.path.join(images_dir, image_file)
        
        # Load the image
        image = Image.open(img_path).convert("RGB")
        
        # If a transform is provided (e.g., resize, normalization), apply it
        if transform:
            image = transform(image)
        
        # Append to our lists
        X.append(image)
        y.append((lat, lon))

    return X, y