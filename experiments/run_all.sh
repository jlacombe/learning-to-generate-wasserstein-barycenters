#!/bin/bash
# usage: ./run_all.sh

# XPs parameters
model_id="2020_10_5_3_20_19"
model_class="BarycentricNet"
results_XPs_path="./results/$model_id"
input_imgs_path="input_imgs" # contains .png files
targets_path="input_targets"
contours_input_path="../datasets/input_contours.h5"
contours_barys_path="../datasets/barycenters_contours.h5"

# model_id="randshapes_"
# model_class="DWE"
# results_XPs_path="./results/DWE_$model_id"
# input_imgs_path="input_imgs"
# targets_path="input_targets"

mkdir $results_XPs_path
n_barys_runtimes=1000

./run_runtimes_XP.sh $model_id $results_XPs_path $n_barys_runtimes -i $contours_input_path -r $contours_barys_path
./run_interpolate_XP.sh $model_id $results_XPs_path $model_class $input_imgs_path $targets_path