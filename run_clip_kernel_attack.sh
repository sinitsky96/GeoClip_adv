#!/bin/bash

# Use Python from conda base environment
export PYTHONPATH=$PYTHONPATH:/home/sinitsky96/project/GeoClip_adv
PYTHON=/home/sinitsky96/miniforge3/bin/python

echo "hello from $($PYTHON --version) in $PYTHON"

# Print CUDA environment information
echo "CUDA Environment:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "Running CLIP attack with SparsePatches PGDTrim on a single sample..."

# Configure CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Make sure CUDA is visible (Slurm might set this, but let's be explicit)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# Store the original directory
ORIGINAL_DIR=$(pwd)

# Run CLIP attack with kernel-based PGDTrim
$PYTHON ./SparsePatches/eval.py \
    --attack_type kernel \
    --model clip \
    --norm L0 \
    --sparsity 224 \
    --kernel_size 4 \
    --kernel_sparsity 8 \
    --bs 32 \
    --n_ex 1 \
    --eps_l_inf 0.1 \
    --n_restarts 3 \
    --n_iter 100 \
    --dataset mixed \
    --device cuda \
    --loss ce \
    --data_path ./data

cd $ORIGINAL_DIR


