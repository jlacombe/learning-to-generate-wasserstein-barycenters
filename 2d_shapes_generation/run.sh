#!/bin/bash
# usage: ./run.sh
n_shapes=10000 # number of shapes to generate
img_size=512 # grid size on which the shapes are displayed
nprocs=10 # number of processors used to accelerate computations

# controls the grid of samples we display
n_cols_disp=10
n_rows_disp=10

datasets_dir="../datasets"

if [[ ! -d "$datasets_dir" ]]
then
    mkdir $datasets_dir
fi

# contours Version
echo "Generating input_contours dataset..."
python generate_2d_shapes.py -i $img_size -n $n_shapes -p $nprocs -d "$datasets_dir/input_contours.h5" -w "True"
python plot_2d_shapes.py -d "$datasets_dir/input_contours.h5" -r $n_rows_disp -c $n_cols_disp
echo "Done. Some samples have been stored in a png file for display purposes. "
echo "A histogram of the depth distribution has also been saved. "
echo "Generating the corresponding downsampled version of the dataset..."
python downsample_dataset.py -d "$datasets_dir/input_contours.h5" -n "$datasets_dir/input_contours_28x28.npy" -s "28,28"
python plot_2d_shapes.py -d "$datasets_dir/input_contours_28x28.npy" -r $n_rows_disp -c $n_cols_disp
echo "Done. Some samples have been stored in a png file for display purposes. "

# full shapes Version
echo "Generating input_full_shapes dataset..."
python generate_2d_shapes.py -i $img_size -n $n_shapes -p $nprocs -d "$datasets_dir/input_full_shapes.h5" -w "False"
python plot_2d_shapes.py -d "$datasets_dir/input_full_shapes.h5" -r $n_rows_disp -c $n_cols_disp
echo "Done. Some samples have been stored in a png file for display purposes. "
echo "A histogram of the depth distribution has also been saved. "
echo "Generating the corresponding downsampled version of the dataset..."
python downsample_dataset.py -d "$datasets_dir/input_full_shapes.h5" -n "$datasets_dir/input_full_shapes_28x28.npy" -s "28,28"
python plot_2d_shapes.py -d "$datasets_dir/input_full_shapes_28x28.npy" -r $n_rows_disp -c $n_cols_disp
echo "Done. Some samples have been stored in a png file for display purposes. "
