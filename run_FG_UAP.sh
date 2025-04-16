#!/bin/bash

# Setup env
# mamba init
mamba activate cs236207

echo "hello from $(python --version) in $(which python)"

echo "Running GeoCLIP attack with FG_UAP..."
export PYTHONPATH=$PYTHONPATH:/home/daniellebed/project/GeoClip_adv

python ./FG_UAP/find_uap.py