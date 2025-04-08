#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/sinitsky96/project/GeoClip_adv
PYTHON=/home/sinitsky96/miniforge3/bin/python

echo "Running GeoCLIP attack with SparsePatches PGDTrim..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 256 \
    --seed 42 \
    --att_kernel_size 2 \
    --batch_size 2 \
    --n_iter 5 \
    --samples_per_dataset 2 \
    --n_examples 2 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256 \
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results
