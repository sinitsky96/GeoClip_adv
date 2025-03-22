#!/bin/bash

# Use Python from conda base environment
export PYTHONPATH=$PYTHONPATH:/home/sinitsky96/project/GeoClip_adv
PYTHON=/home/sinitsky96/miniforge3/bin/python

echo "hello from $($PYTHON --version) in $PYTHON"

echo "Running GeoCLIP attack with SparsePatches PGDTrim..."

# Configure CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Untargeted L0 attack with sparse patches (improved parameters)
$PYTHON ./SparsePatches/eval.py --attack_type sparse --model geoclip --norm L0 --bs 32 --sparsity 500 --eps_l_inf 0.1 --n_restarts 15 --n_iter 100 --device cuda

# Targeted L0 attack with sparse patches (improved parameters)
$PYTHON ./SparsePatches/eval.py --attack_type sparse --model geoclip --norm L0 --targeted --target_class "(37.090924,25.370521)" --bs 32 --sparsity 500 --eps_l_inf 0.1 --n_restarts 15 --n_iter 100 --device cuda

# Untargeted L0 attack with kernel patches (improved parameters)
$PYTHON ./SparsePatches/eval.py --attack_type kernel --model geoclip --norm L0 --bs 32 --sparsity 500 --kernel_size 5 --kernel_sparsity 9 --eps_l_inf 0.1 --n_restarts 15 --n_iter 100 --device cuda

# Targeted L0 attack with kernel patches (improved parameters)
$PYTHON ./SparsePatches/eval.py --attack_type kernel --model geoclip --norm L0 --targeted --target_class "(37.090924,25.370521)" --bs 32 --sparsity 500 --kernel_size 5 --kernel_sparsity 9 --eps_l_inf 0.1 --n_restarts 15 --n_iter 100 --device cuda


