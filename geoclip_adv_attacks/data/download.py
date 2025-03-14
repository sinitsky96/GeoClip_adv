import os
import subprocess
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



def get_transforms(apply_transforms=True):
    """
    Get image transforms for GeoCLIP
    This should match the preprocessing used by CLIP
    
    Args:
        apply_transforms (bool): If True, apply full transformations including normalization.
                               If False, only convert to tensor without any other changes.
    """
    if apply_transforms:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    else:
        # Just convert to tensor without any other transformations
        return transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
            ])

def download_data(data_dir):
    """
    Ensure that the dataset (images + CSV files) is present.
    Run the shell scripts to download them if they are missing.
    """
    images_path = os.path.join(data_dir, 'Im2GPS', 'images')
    csv_path_1 = os.path.join(data_dir, 'Im2GPS', 'im2gps3k_places365.csv')
    csv_path_2 = os.path.join(data_dir, 'Im2GPS', 'im2gps_places365.csv')
    download_images_path = os.path.join(data_dir, 'Im2GPS', 'download_images.sh')
    download_csv_path = os.path.join(data_dir, 'Im2GPS', 'download_csv.sh')

    # 1. Check if images directory exists
    if not os.path.exists(images_path):
        print("Images directory not found. Downloading dataset...")
        subprocess.run(["bash", download_images_path])
    else:
        # Verify that the images directory actually contains images
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print("Images directory exists but contains no image files. Downloading dataset...")
            subprocess.run(["bash", download_images_path])
        else:
            print(f"Using existing images found at {images_path}")

    # 2. Download CSVs if they don't exist
    if not os.path.exists(csv_path_1) or not os.path.exists(csv_path_2):
        print("CSV file(s) not found. Downloading CSVs...")
        subprocess.run(["bash", download_csv_path])
    else:
        print(f"Using existing CSV files found in {os.path.dirname(csv_path_1)}")


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


def get_im2gps_dataloader(
        data_dir,
        batch_size,
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
    images_path = os.path.join(data_dir, 'Im2GPS', 'images')
    csv_path = os.path.join(data_dir, 'Im2GPS', csv_version)
    
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


def load_im2gps_data(data_dir,
                     transform=None,
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
    images_dir = os.path.join(data_dir, 'Im2GPS', 'images')
    csv_path = os.path.join(data_dir, 'Im2GPS', csv_version)
    
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


def preprocess_and_save_for_pgdtrim(
        data_dir,
        output_dir=None,
        csv_version='im2gps_places365.csv',
        max_samples=None,
        apply_transforms=True):
    """
    Preprocess the Im2GPS3k dataset and save it in a format suitable for PGDTrim attacks.
    
    This function:
    1. Downloads the data if not already present
    2. Applies transformations required by GeoCLIP (if apply_transforms=True)
       If apply_transforms=False, only converts to tensor without changing dimensions
    3. Saves the preprocessed images and coordinates as tensors
    
    Args:
        data_dir: Directory where the data is stored
        output_dir: Directory to save the preprocessed data (default: data_dir/preprocessed)
        csv_version: Which CSV file to use
        max_samples: Maximum number of samples to process (None=all)
        apply_transforms: Whether to apply full transformations including normalization
                        If False, only convert to tensor without resizing
    
    Returns:
        Tuple of (images_tensor_path, coords_tensor_path)
    """
    # Ensure data is downloaded
    download_data(data_dir)
    
    # Set up paths
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'Im2GPS', f'preprocessed')
    
    os.makedirs(output_dir, exist_ok=True)
    
    images_dir = os.path.join(data_dir, 'Im2GPS', 'images')
    csv_path = os.path.join(data_dir, 'Im2GPS', csv_version)
    
    # Output paths for tensors
    images_tensor_path = os.path.join(output_dir, 'images_tensor.pt')
    coords_tensor_path = os.path.join(output_dir, 'coords_tensor.pt')
    
    # Check if preprocessed files already exist
    if os.path.exists(images_tensor_path) and os.path.exists(coords_tensor_path):
        print(f"Preprocessed tensors already exist at {output_dir}")
        return images_tensor_path, coords_tensor_path
    
    # Get transform
    transform = get_transforms(apply_transforms)
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Limit number of samples if specified
    if max_samples is not None:
        df = df.iloc[:max_samples]
    
    # Initialize tensors to hold all data
    n_samples = len(df)
    
    if apply_transforms:
        # For transformed images, we know the exact size
        all_images = torch.zeros((n_samples, 3, 224, 224), dtype=torch.float32)
    else:
        # For raw images, we'll collect them in a list then process
        all_images = []
        
    all_coords = torch.zeros((n_samples, 2), dtype=torch.float32)
    
    print(f"Processing {n_samples} images{' with transformations' if apply_transforms else ' without transformations'}...")
    
    # Process each image
    for idx, row in tqdm(df.iterrows(), total=n_samples):
        image_file = row["IMG_ID"]
        lat = row["LAT"]
        lon = row["LON"]
        
        img_path = os.path.join(images_dir, image_file)
        
        # Load and transform the image
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image)
            
            # Store in the tensors
            if apply_transforms:
                all_images[idx] = image_tensor
            else:
                all_images.append(image_tensor)
                
            all_coords[idx, 0] = lat
            all_coords[idx, 1] = lon
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            # Fill with zeros in case of error
            continue
      

    
    # Convert the list to a dictionary if not applying transforms
    if not apply_transforms:
        # We can't stack variable-sized tensors, so save as a list
        print(f"Saving raw tensors of various sizes")
        all_images_dict = {idx: img for idx, img in enumerate(all_images)}
        # Save as a dictionary with indices as keys
        torch.save(all_images_dict, images_tensor_path)
    else:
        # Save as a stacked tensor for fixed-size images
        torch.save(all_images, images_tensor_path)
        
    # Save coordinates
    torch.save(all_coords, coords_tensor_path)
    
    print(f"Preprocessed tensors saved to {output_dir}")
    if apply_transforms:
        print(f"Images tensor shape: {all_images.shape}")
    else:
        print(f"Saved {len(all_images)} raw image tensors of varying dimensions")
    print(f"Coordinates tensor shape: {all_coords.shape}")
    print(f"Transformations applied: {apply_transforms}")
    
    return images_tensor_path, coords_tensor_path


def load_preprocessed_data(data_dir=None, preprocessed_dir=None, apply_transforms=True):
    """
    Load preprocessed Im2GPS3k data for PGDTrim attacks.
    
    Args:
        data_dir: Base data directory (ignored if preprocessed_dir is provided)
        preprocessed_dir: Directory containing preprocessed tensors
        apply_transforms: Whether the data had transformations applied
                        (True = normalized data, False = raw tensor data)
        
    Returns:
        Tuple of (images_tensor, coords_tensor)
        If apply_transforms=False, images_tensor will be a dictionary of {idx: tensor}
    """
    if preprocessed_dir is None:
        if data_dir is None:
            raise ValueError("Either data_dir or preprocessed_dir must be provided")
        
        preprocessed_dir = os.path.join(data_dir, 'Im2GPS', f'preprocessed')
    
    images_tensor_path = os.path.join(preprocessed_dir, 'images_tensor.pt')
    coords_tensor_path = os.path.join(preprocessed_dir, 'coords_tensor.pt')
    
    if not os.path.exists(images_tensor_path) or not os.path.exists(coords_tensor_path):
        raise FileNotFoundError(f"Preprocessed tensors not found in {preprocessed_dir}. Run preprocess_and_save_for_pgdtrim first.")
    
    images_tensor = torch.load(images_tensor_path)
    coords_tensor = torch.load(coords_tensor_path)
    
    # For raw tensors (not transformed), we saved as a dictionary
    if not apply_transforms and isinstance(images_tensor, dict):
        print(f"Loaded {len(images_tensor)} raw image tensors of varying dimensions")
    else:
        print(f"Loaded tensor of shape {images_tensor.shape}")
    
    return images_tensor, coords_tensor


if __name__ == "__main__":
    # Example usage
    data_dir = r"D:\Study Docs\Degree Material\Sem 9 proj\GeoClip_adv\geoclip_adv_attacks\data" # Images are already present at D:\Study Docs\Degree Material\Sem 9 proj\GeoClip_adv\geoclip_adv_attacks\data\Im2GPS\images
    
    # Process without transformations (raw)
    print("Processing without normalization...")
    preprocess_and_save_for_pgdtrim(data_dir, apply_transforms=False)
    
    # Uncomment to process with full transformations
    # print("Processing with normalization...")
    # preprocess_and_save_for_pgdtrim(data_dir, max_samples=100, apply_transforms=True)
    
