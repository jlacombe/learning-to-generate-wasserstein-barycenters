#!/bin/bash
# usage: ./run_generate_wdists.sh
python generate_wdists.py -b 1000 -c 1 -i "../datasets/input_contours.h5" -r "../datasets/barycenters_contours.h5" -w "../datasets/wdists_contours.h5"
