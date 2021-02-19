#!/bin/bash
img_size=512
n_inputs=2
n_barys=100000
shapes_ids_to_display="0,1,2,3,4"

echo "We call \"target point\" a precomputed optimal transport map between the uniform distribution and an input measure. "
echo "Generating target points ..."
python generate_targets.py -n $img_size -i "../datasets/input_contours.h5" -t "../datasets/targets_contours.h5"
echo "Done. Combining target points into barycenters..."
python generate_barycenters.py -n $img_size -s $n_inputs -i "../datasets/input_contours.h5" -t "../datasets/targets_contours.h5" -r "../datasets/barycenters_contours.h5" -m $n_barys
echo "Done."
echo "Saving some samples into png files..."
python plot_barycenters.py -s $shapes_ids_to_display -i "../datasets/input_contours.h5" -b "../datasets/barycenters_contours.h5" -m "samples"

