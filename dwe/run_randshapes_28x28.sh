#!/bin/bash
# usage: ./run_randshapes_28x28.sh
python run_emd.py --dataset_name "input_contours_28x28" --n_pairwise  9000 --train True  --n_iter=10
python run_emd.py --dataset_name "input_contours_28x28" --n_pairwise 10000 --train False --n_iter=1
python build_model.py --dataset_name "input_contours_28x28" --embedding_size 50 --batch_size 32 --epochs 100
python test_model.py --dataset_name "input_contours_28x28" --method_name MSE
python test_model.py --dataset_name "input_contours_28x28" --method_name BARYCENTER
python test_model.py --dataset_name "input_contours_28x28" --method_name INTERPOLATION
