import os
import subprocess
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



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

def download_data(data_dir):
    """
    Ensure that the dataset (images + CSV files) is present.
    Run the shell scripts to download them if they are missing.
    """
    Im2GPS3k_path = os.path.join(data_dir, 'Im2GPS3k')
    images_path = os.path.join(Im2GPS3k_path, 'images')
    csv_path_1 = os.path.join(Im2GPS3k_path, 'im2gps3k_places365.csv')
    csv_path_2 = os.path.join(Im2GPS3k_path, 'im2gps_places365.csv')
    download_images_path = os.path.join(Im2GPS3k_path, 'download_images.sh')
    download_csv_path = os.path.join(Im2GPS3k_path, 'download_csv.sh')

    # # 1. Download images if they don't exist
    # if not os.path.exists(images_path):
    #     print("Images directory not found. Downloading dataset...")
    #     subprocess.run(["bash", download_images_path, Im2GPS3k_path])

    # 2. Download CSVs if they don't exist
    # if not os.path.exists(csv_path_1) or not os.path.exists(csv_path_2):
    if not os.path.exists(csv_path_1):
        print("CSV file(s) not found. Downloading CSVs...")
        subprocess.run(["bash", download_csv_path, Im2GPS3k_path])


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

        target = torch.tensor([lat, lon], dtype=torch.float)
        
        # Return (image, (lat, lon)) 
        return image, target


def get_im2gps_dataloader(
        data_dir,
        batch_size,
        # transform=None, 
        shuffle=True, 
        num_workers=2,
        csv_version='im2gps3k_places365.csv'):
    """
    Main function to:
      - Ensure data is downloaded (images + CSV).
      - Construct the Dataset.
      - Return a DataLoader.
    
    Args:
    data_dir: Directory where the data is stored.
    batch_size: Batch size for the DataLoader.
    transform: Any torchvision transforms to be applied to images.
    shuffle: Whether to shuffle the data.
    num_workers: Number of worker processes for loading.
    csv_version: Which CSV file to read from. You can choose
                'im2gps3k_places365.csv' or 'im2gps_places365.csv'.
    """
    # 1. Make sure the data is available
    download_data(data_dir)
    
    # 2. Paths
    images_path = os.path.join(data_dir, 'Im2GPS3k', 'images')
    csv_path = os.path.join(data_dir, 'Im2GPS3k', csv_version)
    
    # 3. Create the dataset
    dataset = Im2GPSDataset(
        images_dir=images_path,
        csv_path=csv_path,
        transform=get_transforms()
    )
    
    # 4. Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


def load_im2gps_data(data_dir,
                    #  transform=None,
                     csv_version='im2gps3k_places365.csv'):
    """
    Make sure data is available, then load images and their lat/lon from CSV into memory.
    
    Returns:
      X: list of image tensors (or PIL images if you prefer not to convert yet).
      y: list of (lat, lon) tuples.
    """
    # 1. Ensure the data is downloaded
    download_data(data_dir)
    
    # 2. Set paths
    images_dir = os.path.join(data_dir, 'Im2GPS3k', 'images')
    csv_path = os.path.join(data_dir, 'Im2GPS3k', csv_version)
    
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
        image = get_transforms()(image)

        # image = transforms.PILToTensor(image)
        
        # If a transform is provided (e.g., resize, normalization), apply it
        # if transform:
        #     image = transform(image)
        
        # Append to our lists
        X.append(image)
        y.append((lat, lon))

    return X, y