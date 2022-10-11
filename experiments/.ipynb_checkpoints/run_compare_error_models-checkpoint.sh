#!/bin/bash
# usage: ./run_compare_error_models.sh 

echo "=========================="
echo "== COMPARE ERROR MODELS =="
echo "=========================="
model1_id="bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_dsv2"
model1_class="BarycentricUNet"
model2_id="randshapes_"
model2_class="DWE"
results_XPs_path="./results"
inputs_path="../datasets/input_contours.h5"
barys_path="../datasets/barycenters_contours.h5"
n_barys=1000

loss="kldiv_loss"
echo "KL DIV LOSS"
python compute_model_errors.py -m $model1_id -c $model1_class -r $results_XPs_path -i $inputs_path -b $barys_path -n $n_barys -l $loss
python compute_model_errors.py -m $model2_id -c $model2_class -r $results_XPs_path -i $inputs_path -b $barys_path -n $n_barys -l $loss
python compare_model_errors.py -m "$model1_id,$model2_id" -r $results_XPs_path -t "Ours,DWE" -n $n_barys -l $loss

loss="l1_loss"
echo "L1 LOSS"
python compute_model_errors.py -m $model1_id -c $model1_class -r $results_XPs_path -i $inputs_path -b $barys_path -n $n_barys -l $loss
python compute_model_errors.py -m $model2_id -c $model2_class -r $results_XPs_path -i $inputs_path -b $barys_path -n $n_barys -l $loss
python compare_model_errors.py -m "$model1_id,$model2_id" -r $results_XPs_path -t "Ours,DWE" -n $n_barys -l $loss

loss="mmd_loss"
echo "MMD LOSS"
python compute_model_errors.py -m $model1_id -c $model1_class -r $results_XPs_path -i $inputs_path -b $barys_path -n $n_barys -l $loss
python compute_model_errors.py -m $model2_id -c $model2_class -r $results_XPs_path -i $inputs_path -b $barys_path -n $n_barys -l $loss
python compare_model_errors.py -m "$model1_id,$model2_id" -r $results_XPs_path -t "Ours,DWE" -n $n_barys -l $loss
