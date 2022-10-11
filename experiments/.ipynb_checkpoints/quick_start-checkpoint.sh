#!/bin/bash
# usage: ./quick_start.sh [IMAGES_IDS_LIST] [WEIGHTS_LIST]
# example: ./quick_start.sh "1,2,3" "0.3,0.3,0.4"
echo "=================================="
echo "== CREATING GEOMLOSS TARGETS... =="
echo "=================================="

# GeomLoss parameters
results_path="results"
inputs_folder="input_imgs"
targets_folder="input_targets"
n=512
m=512

if [[ ! -d "$results_path" ]]
then
    mkdir $results_path
fi

if [[ ! -d "$targets_folder" ]]
then
    mkdir $targets_folder
fi

echo "Generating targets from $inputs_folder to $targets_folder ..."
python create_geomloss_targets.py -r $results_path -f $inputs_folder -t $targets_folder -n $n -m $m -l $1
echo 'Done.'

echo "======================"
echo "== INTERPOLATING... =="
echo "======================"

# XPs parameters
model_id="bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_dsv2"
model_class="BarycentricUNet"
results_XPs_path="./results"
input_imgs_path="input_imgs" # contains .png files
targets_path="input_targets"
xp_name="quickstart"

if [[ ! -d "$results_XPs_path" ]]
then
    mkdir $results_XPs_path
fi

python interpolate_XP.py -m $model_id -r $results_XPs_path -k "line" -f $input_imgs_path -l $1 -t $targets_path -i $xp_name -w $2 -c $model_class
echo 'Done.'
