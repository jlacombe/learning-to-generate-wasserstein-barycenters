#!/bin/bash
# usage: ./run_randshapes.sh
shapes_fpath="../datasets/input_contours.h5"
wdists_fpath="../datasets/wdists_contours.h5"
python build_model.py --dataset_name randshapes --shapes_fpath $shapes_fpath --wdists_fpath $wdists_fpath --n_data 100000 --batch_size 8 --epochs 31
python test_model.py --dataset_name "randshapes" --method_name MSE
python test_model.py --dataset_name "randshapes" --method_name INTERPOLATION
