"""
This module is responsible for downloading a subset of the MP16 datasets,
which include image URLs and associated metadata. The script downloads CSV files 
containing metadata, merges them, and downloads the corresponding images while 
resizing them to an appropriate format. Finally, it saves the metadata of successfully 
downloaded images.

The main functionalities include:
    - Downloading files and images from given URLs.
    - Processing CSV metadata for image URLs and additional place data.
    - Merging CSV data based on image IDs.
    - Downloading images, resizing them to 224x224 pixels, and saving them locally.
    - Saving metadata for successfully downloaded images in a CSV file.
"""

import os
import requests
import pandas as pd
import random
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import time

# URLs for MP16 dataset metadata
MP16_URLS_CSV = "https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_urls.csv"
MP16_PLACES_CSV = "https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_places365.csv"

# Local paths for storing downloaded data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "download")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

# Create directories if they don't exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def download_file(url, save_path):
    """
    Download a file from a URL and save it to the specified path.

    Parameters:
        url (str): The URL to fetch the file from.
        save_path (str): The path on the local system to save the downloaded file.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_image(url, save_path):
    """
    Download an image from a URL, convert it to RGB, resize it to 224x224 pixels,
    and save it to the specified path.

    Parameters:
        url (str): The URL of the image to download.
        save_path (str): The path on the local system to save the image.

    Returns:
        bool: True if the image download and processing were successful, False otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Convert to RGB to ensure compatibility
        img = img.resize((224, 224))  # Resize to match GeoCLIP input size
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return False

def download_subset_csv(url, save_path, num_rows=10):
    """
    Download a subset (first `num_rows` rows) of a CSV file from a URL and save it locally.
    The function handles CSV files with or without headers based on the URL.

    Parameters:
        url (str): The URL of the CSV file.
        save_path (str): The local path where the subset CSV should be saved.
        num_rows (int, optional): The number of rows to download from the CSV. Defaults to 10.

    Returns:
        bool: True if the subset download was successful, False otherwise.
    """
    try:
        # Read only the first `num_rows` rows
        if 'places365' in url.lower():
            # For places CSV which has headers
            df = pd.read_csv(url, nrows=num_rows)
            df.to_csv(save_path, index=False)
        else:
            # For URLs CSV which doesn't have headers
            df = pd.read_csv(url, nrows=num_rows, header=None)
            df.to_csv(save_path, index=False, header=False)
        print(f"Downloaded {num_rows} rows from {url} to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    """
    Main function that coordinates the download and processing of the MP16 dataset.
    
    Steps performed:
        - Check and download CSV subsets for image URLs and places metadata.
        - Load the CSV files into pandas DataFrames.
        - Merge the two DataFrames on the image ID.
        - Download and process each image (resize and convert to RGB).
        - Save the metadata of successfully downloaded images to a CSV file.
    """
    # Paths for the subset CSV files
    mp16_urls_path = os.path.join(DOWNLOAD_DIR, "mp16_urls_subset.csv")
    mp16_places_path = os.path.join(DOWNLOAD_DIR, "mp16_places365_subset.csv")
    
    # Check if the CSV files already exist and download if they do not
    if not os.path.exists(mp16_urls_path):
        # Download a subset of MP16 URLs CSV
        download_subset_csv(MP16_URLS_CSV, mp16_urls_path)
    else:
        print(f"{mp16_urls_path} already exists. Skipping download.")
    
    if not os.path.exists(mp16_places_path):
        # Download a subset of MP16 Places CSV
        download_subset_csv(MP16_PLACES_CSV, mp16_places_path)
    else:
        print(f"{mp16_places_path} already exists. Skipping download.")
    
    # Load the URLs and metadata
    print("Loading MP16 URLs and metadata...")
    urls_df = pd.read_csv(mp16_urls_path, header=None, names=['img_id', 'url'])
    places_df = pd.read_csv(mp16_places_path)
    
    # Rename column to ensure consistent naming for merging
    places_df = places_df.rename(columns={'IMG_ID': 'img_id'})
    merged_df = pd.merge(urls_df, places_df, on='img_id', how='inner')
    
    # Process each image
    num_images = len(merged_df)
    print(f"Processing {num_images} images...")
    
    successful_downloads = []
    
    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        image_url = row['url']
        image_id = row['img_id']
        lat = row['LAT']
        lon = row['LON']
        
        # Create a filename for the image
        image_filename = f"img_{idx+1:04d}.jpg"
        image_path = os.path.join(IMAGES_DIR, image_filename)
        
        # Skip if image already exists
        if os.path.exists(image_path):
            print(f"Image {image_filename} already exists. Skipping download.")
            successful_downloads.append({
                'IMG_FILE': image_filename,
                'LAT': lat,
                'LON': lon
            })
            continue
        
        # Download the image and add to successful_downloads if successful
        if download_image(image_url, image_path):
            successful_downloads.append({
                'IMG_FILE': image_filename,
                'LAT': lat,
                'LON': lon
            })
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
    
    # Save metadata for successfully downloaded images if any exist
    if successful_downloads:
        print(f"Successfully processed {len(successful_downloads)} images.")
        metadata_df = pd.DataFrame(successful_downloads)
        metadata_path = os.path.join(METADATA_DIR, "mp16_subset.csv")
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")
    else:
        print("No images were successfully downloaded.")

if __name__ == "__main__":
    main()