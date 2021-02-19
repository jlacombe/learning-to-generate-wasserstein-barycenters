#!/bin/bash
python run_emd.py --dataset_name "cat" --n_pairwise  900 --train True  --n_iter=10
python run_emd.py --dataset_name "cat" --n_pairwise 1000 --train False --n_iter=1
python build_model.py --dataset_name "cat" --embedding_size 50 --batch_size 32 --epochs 100
python test_model.py --dataset_name "cat" --method_name MSE
python test_model.py --dataset_name "cat" --method_name BARYCENTER
python test_model.py --dataset_name "cat" --method_name INTERPOLATION

