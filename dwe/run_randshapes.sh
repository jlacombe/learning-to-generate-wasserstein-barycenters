#!/bin/bash
# usage: ./run_randshapes.sh
dataset_id="randshapes"
shapes_fpath="../datasets/input_contours.h5"
wdists_fpath="../datasets/wdists_contours.h5"
nd=100000
bs=8
epochs=31

python build_model.py --dataset_name $dataset_id --shapes_fpath $shapes_fpath --wdists_fpath $wdists_fpath --n_data $nd --batch_size $bs --epochs $epochs
