#!/bin/bash
# usage: ./run_generate_wdists.sh
batch_size=1000
chunk_size=1
input_shapes_path="../datasets/input_contours.h5"
barys_path="../datasets/barycenters_contours.h5"
wdists_path="../datasets/wdists_contours.h5"

python generate_wdists.py -b $batch_size -c $chunk_size -i $input_shapes_path -r $barys_path -w $wdists_path
