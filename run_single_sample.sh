#!/bin/bash

# Use Python from conda base environment
export PYTHONPATH=$PYTHONPATH:/home/sinitsky96/project/GeoClip_adv
PYTHON=/home/sinitsky96/miniforge3/bin/python

echo "hello from $($PYTHON --version) in $PYTHON"

echo "Running GeoCLIP attack with SparsePatches PGDTrim on a single sample..."

# Configure CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Untargeted L0 attack with sparse patches on a single sample
$PYTHON ./SparsePatches/eval.py --attack_type sparse --model geoclip --norm L0 --bs 1 --n_ex 1 --sparsity 500 --eps_l_inf 0.1 --n_restarts 3 --n_iter 50 --device cuda

# The above runs with:
# - Only 1 example (--n_ex 1)
# - Batch size of 1 (--bs 1)
# - 3 restarts for quick testing
# - 50 iterations per restart
# - L0 sparsity of 500 pixels
# - L-infinity constraint of 0.1 