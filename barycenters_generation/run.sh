#!/bin/bash
img_size=512
n_inputs=2
n_barys=100000
shapes_ids_to_display="0,1,2,3,4"

# for synthetic data
inputs_path="../datasets/input_contours.h5"
targets_path="../datasets/targets_contours.h5"
barys_path="../datasets/barycenters_contours.h5"

# for Flickr
# inputs_path="../datasets/flickr/input_chrom_histos_10000.h5"
# targets_path="../datasets/flickr/targets_chrom_histos_10000.h5"
# barys_path="../datasets/flickr/barycenters_n=100000_s=2.h5"

echo "We call \"target point\" a precomputed optimal transport map between the uniform distribution and an input measure. "
echo "Generating target points ..."
python generate_targets.py -n $img_size -i $inputs_path -t $targets_path
echo "Done. Combining target points into barycenters..."
python generate_barycenters.py -n $img_size -s $n_inputs -i $inputs_path -t $targets_path -r $barys_path -m $n_barys
echo "Done."
echo "Saving some samples into png files..."
python plot_barycenters.py -s $shapes_ids_to_display -i $inputs_path -b $barys_path -m "samples"

