#!/bin/bash

# Setup env
# mamba init
mamba activate cs236207

echo "hello from $(python --version) in $(which python)"

echo "Running GeoCLIP attack with Sparse-RS..."
export PYTHONPATH=$PYTHONPATH:/home/daniellebed/project/GeoClip_adv
# targted runs:
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --targeted --target_class "(37.090924,25.370521)" --bs 32 --n_queries 100 --k 25 --device cuda
# python ./sparse_rs/eval.py --loss ce --model geoclip --norm L0 --targeted --target_class "(37.090924,25.370521)" --bs 32 --n_queries 100 --k 25 --device cuda
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --targeted --target_class "(37.090924,25.370521)" --bs 32 --n_queries 100 --k 5 --device cuda
# python ./sparse_rs/eval.py --loss ce --model geoclip --norm patches --targeted --target_class "(37.090924,25.370521)" --bs 32 --n_queries 100 --k 5 --device cuda

# untargted runs:
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 100 --k 25 --device cuda
# python ./sparse_rs/eval.py --loss ce --model geoclip --norm L0 --bs 32 --n_queries 100 --k 25 --device cuda
# python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 100 --k 5 --device cuda
# python ./sparse_rs/eval.py --loss ce --model geoclip --norm patches --bs 32 --n_queries 100 --k 5 --device cuda


python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 1 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 2 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 4 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 8 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 16 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 32 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 64 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 128 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 256 --device cuda

python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 1 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 2 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 4 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 8 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 16 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 32 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 64 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 128 --device cuda
python ./sparse_rs/eval.py --loss margin --model geoclip --norm patches --bs 32 --n_queries 1000 --k 224 --device cuda