#!/bin/bash
# usage: ./run_all.sh

# XPs parameters
model_id="bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_dsv2"
model_class="BarycentricUNet"
results_XPs_path="../training_model/results/$model_id"
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

# uncomment next line if you want to reproduce runtimes experiment
#./run_runtimes_XP.sh $model_id $results_XPs_path $n_barys_runtimes -i $contours_input_path -r $contours_barys_path
./run_interpolate_XP.sh $model_id $results_XPs_path $model_class $input_imgs_path $targets_path