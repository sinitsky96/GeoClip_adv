#!/bin/bash

# Setup env
# mamba init
mamba activate cs236207

echo "hello from $(python --version) in $(which python)"

echo "Running GeoCLIP attack with Sparse-RS..."
export PYTHONPATH=$PYTHONPATH:/home/daniellebed/project/GeoClip_adv



### geoclip

# untargted geoclip, loss margin:

for k in 1 2 4 8 16 32 64 128 256
do
    echo "Running with k=$k"
    python ./sparse_rs/eval.py \
        --loss margin \
        --model geoclip \
        --norm L0 \
        --bs 150 \
        --n_queries 1000 \
        --k "$k" \
        --device cuda

done

for k in 4 16
do
  eps=$k
  while [ $eps -le 256 ]
  do
    echo "Running with k=$k, eps=$eps"
    python ./sparse_rs/eval.py \
      --loss margin \
      --model geoclip \
      --norm patches \
      --bs 150 \
      --n_queries 1000 \
      --eps "$eps" \
      --k "$k" \
      --device cuda

    # double eps
    eps=$(( eps * 2 ))
  done
done

# ####

# # untargted clip, loss margin:
for k in 1 2 4 8 16 32 64 128 256
do
    echo "Running with k=$k"
    python ./sparse_rs/eval.py \
        --loss margin \
        --model clip \
        --norm L0 \
        --bs 150 \
        --n_queries 1000 \
        --k "$k" \
        --device cuda

done

for k in 4 16
do
  eps=$k
  while [ $eps -le 256 ]
  do
    echo "Running with k=$k, eps=$eps"
    python ./sparse_rs/eval.py \
      --loss margin \
      --model clip \
      --norm patches \
      --bs 150 \
      --n_queries 1000 \
      --eps "$eps" \
      --k "$k" \
      --device cuda

    # double eps
    eps=$(( eps * 2 ))
  done
done


# targted geoclip, loss ce, target eiffel tower:

for k in 1 2 4 8 16 32 64 128 256
do
    echo "Running with k=$k"
    python ./sparse_rs/eval.py \
        --loss margin \
        --model geoclip \
        --norm L0 \
        --bs 150 \
        --n_queries 1000 \
        --k "$k" \
        --device cuda \
        --targeted \
        --target_class "(48.858093, 2.294694)"

done

for k in 4 16
do
  eps=$k
  while [ $eps -le 256 ]
  do
    echo "Running with k=$k, eps=$eps"
    python ./sparse_rs/eval.py \
      --loss margin \
      --model geoclip \
      --norm patches \
      --bs 150 \
      --n_queries 1000 \
      --eps "$eps" \
      --k "$k" \
      --device cuda \
      --targeted \
      --target_class "(48.858093, 2.294694)"

    # double eps
    eps=$(( eps * 2 ))
  done
done




#   echo "Running sparse with k=128"
#   python ./sparse_rs/eval.py \
#     --loss margin \
#     --model geoclip \
#     --norm L0 \
#     --bs 150 \
#     --n_queries 1 \
#     --k 128 \
#     --device cuda \
#     --targeted \
#     --target_class "(48.858093, 2.294694)"
#    echo "Running sparse with k=128"
#   python ./sparse_rs/eval.py \
#     --loss margin \
#     --model geoclip \
#     --norm patches \
#     --bs 150 \
#     --n_queries 1 \
#     --eps 128 \
#     --k 16 \
#     --device cuda \
#     --targeted \
#     --target_class "(48.858093, 2.294694)"
#   echo "Running CLIP with k=16, eps=128"
#   python ./sparse_rs/eval.py \
#       --loss margin \
#       --model clip \
#       --norm patches \
#       --bs 150 \
#       --n_queries 1 \
#       --eps 128 \
#       --k 16 \
#       --device cuda
# # single targeted run:
#   echo "Running CLIP with eps=128"
#   python ./sparse_rs/eval.py \
#         --loss margin \
#         --model clip \
#         --norm L0 \
#         --bs 150 \
#         --n_queries 1 \
#         --k 128 \
#         --device cuda