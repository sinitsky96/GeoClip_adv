#!/bin/bash

# Use Python from conda base environment
export PYTHONPATH=$PYTHONPATH:/home/sinitsky96/project/GeoClip_adv
PYTHON=/home/sinitsky96/miniforge3/bin/python

echo "hello from $($PYTHON --version) in $PYTHON"

echo "Running GeoCLIP attack with SparsePatches PGDTrim..."

# Configure CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4X4 256 sparsity
$PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 4 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 256 --seed 42

# # 4X4 128 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 4 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 128 --seed 43

# # 4X4 64 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 4 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 64 --seed 44

# # 4X4 16 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 4 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 16 --seed 45

# # 2X2 256 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 2 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 256 --seed 46

# # 2X2 128 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 2 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 128 --seed 47

# # 2X2 64 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 2 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 64 --seed 48

# # 2X2 16 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 2 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 16 --seed 49

# # 2X2 8 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 2 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 8 --seed 50

# # 1X1 256 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 1 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 256 --seed 51

# # 1X1 128 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 1 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 128 --seed 52

# # 1X1 64 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 1 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 64 --seed 53

# # 1X1 16 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 1 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 16 --seed 54

# # 1X1 8 sparsity
# $PYTHON ./SparsePatches/eval.py --attack_type kernel --kernel_size 1 --model geoclip --norm L0 --bs 32 --n_ex 1  --eps_l_inf 0.1 --n_restarts 1 --n_iter 100  --dataset mixed  --device cuda --sparsity 8 --seed 55


