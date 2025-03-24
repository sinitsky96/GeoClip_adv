#!/bin/bash

# Set to download only 1000 rows
export MAX_ROWS=1000

# Run the download scripts
echo "Downloading sample of MP-16 dataset (1000 rows)"
echo "==============================================="

# Download CSV files first
./download_csv.sh
./download_images.sh

echo "==============================================="
echo "Sample CSV files downloaded. To use a different sample size:"
echo "  export MAX_ROWS=5000  # Or any other number"
echo "  ./download_csv.sh"
echo "  ./download_images.sh" 