#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/sinitsky96/project/GeoClip_adv
PYTHON=/home/sinitsky96/miniforge3/bin/python


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "4X4 kernels"
echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 4X4 kernels with sparsities 256"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 256 \
    --seed 42 \
    --att_kernel_size 2 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256\
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results

echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 4X4 kernels with sparsities 128"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 128 \
    --seed 42 \
    --att_kernel_size 4 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256 128 \
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results

echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 4X4 kernels with sparsities 64"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 64 \
    --seed 42 \
    --att_kernel_size 4 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256 128 64 \
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results


echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 4X4 kernels with sparsities 16"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 16 \
    --seed 42 \
    --att_kernel_size 4 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256 128 64 32 16 \
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results


echo "2X2 kernels"
echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 2X2 kernels with sparsities 256"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 256 \
    --seed 42 \
    --att_kernel_size 2 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256\
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results

echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 2X2 kernels with sparsities 128"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 128 \
    --seed 42 \
    --att_kernel_size 2 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256 128\
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results

echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 2X2 kernels with sparsities 64"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 64 \
    --seed 42 \
    --att_kernel_size 2 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256 128 64  \
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results


echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 2X2 kernels with sparsities 16"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 16 \
    --seed 42 \
    --att_kernel_size 2 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256 128 64 32 16 \
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results






echo "1X1 kernels"
echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 1X1 kernels with sparsities 256"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 256 \
    --seed 42 \
    --att_kernel_size 1 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
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

echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 1X1 kernels with sparsities 128"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 128 \
    --seed 42 \
    --att_kernel_size 1 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256 128 \
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results

echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 1X1 kernels with sparsities 64"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 64 \
    --seed 42 \
    --att_kernel_size 1 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256 128 64 \
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results


echo "Running GeoCLIP attack with SparsePatches PGDTrimkernel for 1X1 kernels with sparsities 16"
$PYTHON ./SparsePatches/run_attack.py \
    --dataset mixed \
    --device cuda \
    --sparsity 16 \
    --seed 42 \
    --att_kernel_size 1 \
    --batch_size 12 \
    --n_iter 20 \
    --samples_per_dataset 20 \
    --n_examples 20 \
    --att_trim_steps 32768 16384 8192 4096 2048 1024 512 256 128 64 32 16 \
    --att_trim_steps_reduce none \
    --att_mask_dist bernoulli \
    --att_trim_best_mask 0 \
    --eps_l_inf_from_255 8 \
    --att_mask_prob_amp_rate 1.0 \
    --att_norm_mask_amp \
    --attack_verbose \
    --report_info \
    --save_results

    




