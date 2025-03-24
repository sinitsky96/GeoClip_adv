#!/bin/bash

# Use Python from conda base environment
export PYTHONPATH=$PYTHONPATH:/home/sinitsky96/project/GeoClip_adv
PYTHON=/home/sinitsky96/miniforge3/bin/python

echo "hello from $($PYTHON --version) in $PYTHON"

echo "Running GeoCLIP attack with SparsePatches PGDTrim on a single sample..."

# Configure CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# First, set up a small sample of the MP-16 dataset
echo "Setting up MP-16 dataset sample..."
# Store the original directory
ORIGINAL_DIR=$(pwd)
cd /home/sinitsky96/project/GeoClip_adv/data/MP_16
export MAX_ROWS=100  # Use just 100 rows for testing
./download_csv.sh
./download_images.sh
# Return to the GeoClip_adv directory
cd /home/sinitsky96/project/GeoClip_adv

# Untargeted L0 attack with sparse patches on a single sample
$PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 4 --model geoclip --norm L0 --bs 1 --n_ex 1  --eps_l_inf 0.1 --n_restarts 2 --n_iter 50 --device cuda --dataset mixed --max_images 10

# The above runs with:
# - Only 1 example (--n_ex 1)
# - Batch size of 1 (--bs 1)
# - 3 restarts for quick testing
# - 50 iterations per restart
# - L0 sparsity of 500 pixels
# - L-infinity constraint of 0.1
# - Using MP-16 dataset with max 20 images

# Return to the original directory
cd $ORIGINAL_DIR 