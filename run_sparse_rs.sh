#!/bin/bash

# Setup env
# mamba init
mamba activate cs236207

echo "hello from $(python --version) in $(which python)"

echo "Running GeoCLIP attack with Sparse-RS..."
export PYTHONPATH=$PYTHONPATH:/home/daniellebed/project/GeoClip_adv

#### geoclip

# untargted geoclip, loss margin - ran:
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 1 --device cuda
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 2 --device cuda
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 4 --device cuda
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 8 --device cuda
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 16 --device cuda
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 32 --device cuda
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 64 --device cuda
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 128 --device cuda
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 224 --device cuda

# untargted geoclip, loss ce:
python ./sparse_rs/eval.py --loss ce --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 1 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --norm patches --bs 32 --n_queries 1000 --k 2 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --norm patches --bs 32 --n_queries 1000 --k 4 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --norm patches --bs 32 --n_queries 1000 --k 8 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --norm patches --bs 32 --n_queries 1000 --k 16 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --norm patches --bs 32 --n_queries 1000 --k 32 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --norm patches --bs 32 --n_queries 1000 --k 64 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --norm patches --bs 32 --n_queries 1000 --k 128 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --norm patches --bs 32 --n_queries 1000 --k 224 --device cuda

# targted geoclip, loss ce, target eiffel tower:
python ./sparse_rs/eval.py --loss ce --model geoclip --targeted --target_class "(48.858093, 2.294694)" --norm L0 --bs 32 --n_queries 1000 --k 1 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --targeted --target_class "(48.858093, 2.294694)" --norm patches --bs 32 --n_queries 1000 --k 2 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --targeted --target_class "(48.858093, 2.294694)" --norm patches --bs 32 --n_queries 1000 --k 4 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --targeted --target_class "(48.858093, 2.294694)" --norm patches --bs 32 --n_queries 1000 --k 8 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --targeted --target_class "(48.858093, 2.294694)" --norm patches --bs 32 --n_queries 1000 --k 16 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --targeted --target_class "(48.858093, 2.294694)" --norm patches --bs 32 --n_queries 1000 --k 32 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --targeted --target_class "(48.858093, 2.294694)" --norm patches --bs 32 --n_queries 1000 --k 64 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --targeted --target_class "(48.858093, 2.294694)" --norm patches --bs 32 --n_queries 1000 --k 128 --device cuda
python ./sparse_rs/eval.py --loss ce --model geoclip --targeted --target_class "(48.858093, 2.294694)" --norm patches --bs 32 --n_queries 1000 --k 224 --device cuda

####

# untargted clip, loss margin:
python ./sparse_rs/eval.py --loss margin --model clip --norm L0 --bs 32 --n_queries 1000 --k 1 --device cuda
python ./sparse_rs/eval.py --loss margin --model clip --norm patches --bs 32 --n_queries 1000 --k 2 --device cuda
python ./sparse_rs/eval.py --loss margin --model clip --norm patches --bs 32 --n_queries 1000 --k 4 --device cuda
python ./sparse_rs/eval.py --loss margin --model clip --norm patches --bs 32 --n_queries 1000 --k 8 --device cuda
python ./sparse_rs/eval.py --loss margin --model clip --norm patches --bs 32 --n_queries 1000 --k 16 --device cuda
python ./sparse_rs/eval.py --loss margin --model clip --norm patches --bs 32 --n_queries 1000 --k 32 --device cuda
python ./sparse_rs/eval.py --loss margin --model clip --norm patches --bs 32 --n_queries 1000 --k 64 --device cuda
python ./sparse_rs/eval.py --loss margin --model clip --norm patches --bs 32 --n_queries 1000 --k 128 --device cuda
python ./sparse_rs/eval.py --loss margin --model clip --norm patches --bs 32 --n_queries 1000 --k 224 --device cuda


# untargted clip, loss ce:
python ./sparse_rs/eval.py --loss ce --model clip --norm L0 --bs 32 --n_queries 1000 --k 1 --device cuda
python ./sparse_rs/eval.py --loss ce --model clip --norm patches --bs 32 --n_queries 1000 --k 2 --device cuda
python ./sparse_rs/eval.py --loss ce --model clip --norm patches --bs 32 --n_queries 1000 --k 4 --device cuda
python ./sparse_rs/eval.py --loss ce --model clip --norm patches --bs 32 --n_queries 1000 --k 8 --device cuda
python ./sparse_rs/eval.py --loss ce --model clip --norm patches --bs 32 --n_queries 1000 --k 16 --device cuda
python ./sparse_rs/eval.py --loss ce --model clip --norm patches --bs 32 --n_queries 1000 --k 32 --device cuda
python ./sparse_rs/eval.py --loss ce --model clip --norm patches --bs 32 --n_queries 1000 --k 64 --device cuda
python ./sparse_rs/eval.py --loss ce --model clip --norm patches --bs 32 --n_queries 1000 --k 128 --device cuda
python ./sparse_rs/eval.py --loss ce --model clip --norm patches --bs 32 --n_queries 1000 --k 224 --device cuda


#####

# tests

# python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 11 --k 1 --device cuda
# python ./sparse_rs/eval.py --loss ce --model geoclip --norm L0 --bs 32 --n_queries 11 --k 1 --device cuda
# python ./sparse_rs/eval.py --loss ce --model geoclip --targeted --target_class "(48.858093, 2.294694)" --norm L0 --bs 32 --n_queries 11 --k 1 --device cuda
# python ./sparse_rs/eval.py --loss margin --model clip --norm L0 --bs 32 --n_queries 11 --k 1 --device cuda
# python ./sparse_rs/eval.py --loss ce --model clip --norm L0 --bs 32 --n_queries 11 --k 1 --device cuda
