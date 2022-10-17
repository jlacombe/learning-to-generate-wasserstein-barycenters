#!/bin/bash
# usage: ./run_randshapes_28x28.sh
dataset_id="randshapes_28x28"
train_nd=9000
train_nfiles=10
eval_nd=10000
eval_nfiles=1
emb_size=50
bs=32
epochs=100

python run_emd.py --dataset_name $dataset_id --n_pairwise  $train_nd --train True  --n_iter=$train_nfiles
python run_emd.py --dataset_name $dataset_id --n_pairwise $eval_nd --train False --n_iter=$eval_nfiles
python build_model.py --dataset_name $dataset_id --embedding_size $emb_size --batch_size $bs --epochs $epochs
python test_model.py --dataset_name $dataset_id --method_name MSE
python test_model.py --dataset_name $dataset_id --method_name BARYCENTER
python test_model.py --dataset_name $dataset_id --method_name INTERPOLATION
