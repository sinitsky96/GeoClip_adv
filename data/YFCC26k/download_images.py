import os
import csv
import time
import requests
from pathlib import Path
from requests.exceptions import RequestException

# Path to your CSV file
CSV_FILE = 'yfcc25600_urls.csv'
# Base directory to save images
OUTPUT_DIR = Path('./images')

# Request settings
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/122.0.0.0 Safari/537.36'
}
MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds, will double each time
REQUEST_DELAY = 1  # seconds between requests to reduce rate-limiting

def download_image(filename, url):
    if not url:
        print(f"Skipping {filename}: no URL provided.")
        return

    output_path = OUTPUT_DIR / filename
    if output_path.exists():
        print(f"Skipping {filename}: already exists.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, stream=True, timeout=15, headers=HEADERS)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Downloaded {filename}")
                time.sleep(REQUEST_DELAY)  # Delay between successful downloads
                return
            elif response.status_code == 429:
                wait_time = RETRY_BACKOFF * (2 ** retries)
                print(f"Rate limited on {filename}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"Failed to download {filename}: HTTP {response.status_code}")
                return
        except RequestException as e:
            wait_time = RETRY_BACKOFF * (2 ** retries)
            print(f"Error downloading {filename}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            retries += 1

    print(f"Giving up on {filename} after {MAX_RETRIES} retries.")

def main():
    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 2:
                print(f"Skipping invalid row: {row}")
                continue

            filename, url = row[0].strip(), row[1].strip()
            if not filename or not url:
                print(f"Skipping row with missing data: {row}")
                continue

            download_image(filename, url)

if __name__ == '__main__':
    main()
