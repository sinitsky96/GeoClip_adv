#!/bin/bash

# Set up environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Empty CUDA cache before running - using proper quoting
python -c "import torch; torch.cuda.empty_cache()"

# Echo commands
set -x

# Set optimization level for pytorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Reduce memory usage by optimizing GPU memory allocation
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Running CLIP kernel attack"

# Run CLIP attack with reduced batch size and memory usage
python ./SparsePatches/eval.py \
    --attack_type kernel \
    --kernel_size 4 \
    --kernel_sparsity 8 \
    --model clip \
    --norm L0 \
    --bs 1 \
    --n_ex 1 \
    --sparsity 256 \
    --eps_l_inf 0.1 \
    --n_restarts 1 \
    --n_iter 1 \
    --dataset mixed \
    --device cuda

echo "Attack complete" 