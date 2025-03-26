#!/bin/bash

# Use Python from conda base environment
export PYTHONPATH=$PYTHONPATH:/home/sinitsky96/project/GeoClip_adv
PYTHON=/home/sinitsky96/miniforge3/bin/python

echo "hello from $($PYTHON --version) in $PYTHON"

# Print CUDA environment information
echo "CUDA Environment:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "Running GeoCLIP attack with SparsePatches PGDTrim on a single sample..."

# Configure CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Make sure CUDA is visible (Slurm might set this, but let's be explicit)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# Run the attack with improved parameters
$PYTHON ./SparsePatches/eval.py \
    --attack_type kernel \
    --kernel_size 8 \
    --model geoclip \
    --norm L0 \
    --bs 4 \
    --n_ex 1 \
    --eps_l_inf 0.3 \
    --n_restarts 1 \
    --n_iter 5 \
    --dataset mixed \
    --device cuda \
    --sparsity 128 \
    --samples_per_dataset 2 \
    --targeted \
    --target_class "(37.090924, 25.370521)" \
    --loss margin

# Return to original directory
cd $ORIGINAL_DIR 