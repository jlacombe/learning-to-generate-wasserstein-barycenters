#!/bin/bash
# usage: ./run_runtimes_XP.sh [model_id] [results_XPs_path] [n_barys]
echo "=========================="
echo "== RUNTIMES EXPERIMENTS =="
echo "=========================="
python runtime_XP_NN.py -m $1 -r $2 -n $3
python runtime_XP_geomloss.py -r $2 -n $3
