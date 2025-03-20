import os
import subprocess
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SAMPLED_CSV = "sampled_data.csv"

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
    sample_path = os.path.join(Im2GPS3k_path, SAMPLED_CSV)
    csv_path_2 = os.path.join(Im2GPS3k_path, 'im2gps_places365.csv')
    download_images_path = os.path.join(Im2GPS3k_path, 'download_images.sh')
    download_csv_path = os.path.join(Im2GPS3k_path, 'download_csv.sh')

    # if not os.path.exists(images_path):
    #     print("Images directory not found. Downloading dataset...")
    #     subprocess.run(["bash", download_images_path, Im2GPS3k_path])

    # if not os.path.exists(csv_path_1) or not os.path.exists(csv_path_2):
    if not os.path.exists(csv_path_1):
        print("CSV file(s) not found. Downloading CSVs...")
        subprocess.run(["bash", download_csv_path, Im2GPS3k_path])

    if not os.path.exists(sample_path):
        sample_kmeans(csv_path_1, sample_path)

def sample_kmeans(csv_path, save_path):
    df = pd.read_csv(csv_path)

    features = ['LAT', 'LON', 'Prob_indoor', 'Prob_natural', 'Prob_urban']
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 200
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    df['cluster_label'] = kmeans.labels_

    sampled_rows = []
    for cluster_label in range(k):
        cluster_indices = df.index[df['cluster_label'] == cluster_label]
        cluster_points_scaled = X_scaled[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_label]
        distances = np.linalg.norm(cluster_points_scaled - centroid, axis=1)
        closest_index = cluster_indices[distances.argmin()]
        sampled_rows.append(closest_index)

    sampled_df = df.loc[sampled_rows].copy()
    sampled_df.to_csv(save_path, index=False)

def load_places365_categories(txt_file_path):
    """
    Reads a file such as categories_365.txt and returns a list, idx = class id, val = str class

    Example line format:
    /a/artists_loft 22
    /a/assembly_line 23
    ...
    """
    mapping = [0] * 365
    with open(txt_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cat_str, label_str = line.rsplit(' ', 1)
            label_id = int(label_str)

            cat_str = cat_str.strip('/')
            _, cat_str = cat_str.rsplit('/', 1)
            
            mapping[label_id] = cat_str
    return mapping


class Im2GPSDataset(Dataset):
    """
    Custom dataset that:
      - Reads filenames and their lat/lon from a CSV file.
      - Loads images from the directory.
      - Returns (transformed_image, (lat, lon)).
    """
    def __init__(self, images_dir, data_path, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform
        
        self.data = pd.read_csv(os.path.join(data_path, SAMPLED_CSV))
        
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

        try: # if an image has multiple frames
            if getattr(image, "n_frames", 1) > 1:
                image.seek(0)
        except Exception:
            pass

        if self.transform:
            image = self.transform(image)

        target = torch.tensor([lat, lon], dtype=torch.float)
        
        # Return (image, (lat, lon)) 
        return image, target
    

class Places365Dataset(Dataset):
    """
    A simple dataset to load (image, label) pairs from your CSV.

    csv_path: path to the CSV file
    images_dir: directory containing the actual image files
    """
    def __init__(self, csv_path, images_dir):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_filename = row["IMG_ID"]
        label_365 = row["S365_Label"]

        # Load the image from disk
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        # Return the PIL image and the integer label
        return image, torch.tensor(label_365, dtype=torch.long)


def get_im2gps_dataloader(
        data_dir,
        batch_size,
        # transform=None, 
        shuffle=True, 
        num_workers=2,
        csv_version=SAMPLED_CSV):

    download_data(data_dir)
    
    images_path = os.path.join(data_dir, 'Im2GPS3k', 'images')
    csv_path = os.path.join(data_dir, 'Im2GPS3k', csv_version)
    
    dataset = Im2GPSDataset(
        images_dir=images_path,
        csv_path=csv_path,
        transform=get_transforms()
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


def load_im2gps_data(data_dir,
                    #  transform=None,
                     csv_version=SAMPLED_CSV):

    download_data(data_dir)
    
    images_dir = os.path.join(data_dir, 'Im2GPS3k', 'images')
    csv_path = os.path.join(data_dir, 'Im2GPS3k', csv_version)
    
    df = pd.read_csv(csv_path)

    X = []
    y = []
    
    for idx, row in df.iterrows():
        image_file = row["IMG_ID"]  
        lat = row["LAT"]  
        lon = row["LON"]          
        
        img_path = os.path.join(images_dir, image_file)
        
        # Load the image
        image = Image.open(img_path).convert("RGB")
        image = get_transforms()(image)
        
        # Append to our lists
        X.append(image)
        y.append((lat, lon))
    
    X = torch.stack(X, dim=0)  # Shape: [N, C, H, W]
    y = torch.tensor(y, dtype=torch.float)  # Shape: [N, 2]

    return X, y


def CLIP_load_data_tensor(data_dir,
                          csv_version=SAMPLED_CSV):
    """
    Reads the CSV file, loads images and labels into memory.
    
    Args:
        csv_path   (str): path to your CSV file
        images_dir (str): directory containing the image files
    
    Returns:
        images (list of PIL Images)
        labels (torch.LongTensor)  -- the S365_Label column
    """


    images_dir = os.path.join(data_dir, 'Im2GPS3k', 'images')
    csv_path = os.path.join(data_dir, 'Im2GPS3k', csv_version)

    download_data(data_dir)

    df = pd.read_csv(csv_path)

    images = []
    labels = []

    for _, row in df.iterrows():
        img_filename = row["IMG_ID"]
        class_id = row["S365_Label"]
        
        img_path = os.path.join(images_dir, img_filename)
        img = Image.open(img_path).convert("RGB")
        image = get_transforms()(img)

        images.append(image)
        labels.append(class_id)

    images = torch.stack(images, dim=0)
    labels_tensor = torch.tensor(labels, dtype=str)

    return images, labels_tensor