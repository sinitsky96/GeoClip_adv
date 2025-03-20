#!/bin/bash

# Setup env
# mamba init
mamba activate cs236207

echo "hello from $(python --version) in $(which python)"

echo "Running GeoCLIP attack with Sparse-RS..."
export PYTHONPATH=$PYTHONPATH:/home/daniellebed/project/GeoClip_adv
python ./sparse_rs/eval.py --model geoclip --norm L0 --bs 32 --n_queries 1000 --k 400 --device cuda
python ./sparse_rs/eval.py --model geoclip --norm L0 --targeted --target_class "(37.090924,25.370521)" --bs 32 --n_queries 1000 --k 400 --device cuda
python ./sparse_rs/eval.py --model geoclip --norm patches --bs 32 --n_queries 1000 --k 20 --device cuda
python ./sparse_rs/eval.py --model geoclip --norm patches --targeted --target_class "(37.090924,25.370521)" --bs 32 --n_queries 1000 --k 20 --device cuda
