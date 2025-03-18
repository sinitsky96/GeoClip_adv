#!/bin/bash

# Setup env
# mamba init
mamba activate cs236207

echo "hello from $(python --version) in $(which python)"

echo "Running GeoCLIP attack with Sparse-RS..."
export PYTHONPATH=$PYTHONPATH:/home/daniellebed/project/GeoClip_adv
python ./sparse_rs/eval.py --model geoclip --norm L0 --bs 128 --n_queries 1000 --k 150 --alpha_init 0.3 --device cuda
