#!/bin/bash

# Download data if not already present (assumes the download scripts are in the download/ folder)
echo "Downloading images and CSV files if necessary..."
bash ./download/download_images.sh
bash ./download/download_csv.sh

# Run the attack on GeoCLIP using our RSAttackGeoCLIP implementation
echo "Running GeoCLIP attack with Sparse-RS..."
python eval.py --model geoclip --norm L0 --n_ex 100 --n_queries 1000 --k 150 --alpha_init 0.3 --device cuda
