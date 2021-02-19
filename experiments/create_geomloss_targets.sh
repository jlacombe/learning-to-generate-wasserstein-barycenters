#!/bin/bash
# usage: ./create_geomloss_targets.sh [results_XPs_path]
echo "==============================="
echo "== GEOMLOSS TARGETS CREATION =="
echo "==============================="

results_path="results"

# 512x512
inputs_folder="input_imgs"
targets_folder="input_targets"
n=512
m=512

# 28x28
# inputs_folder="input_imgs_28x28"
# targets_folder="input_targets_28x28"
# n=512
# m=28

echo "Generating targets from $inputs_folder to $targets_folder ..."
python create_geomloss_targets.py -r $results_path -f $inputs_folder -t $targets_folder -n $n -m $m
