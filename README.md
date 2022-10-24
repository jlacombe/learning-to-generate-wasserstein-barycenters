# Learning to Generate Wasserstein Barycenters
Implementation of the "Learning to Generate Wasserstein Barycenters" paper (https://arxiv.org/abs/2102.12178).

## Requirements
Our implementation was developed using Python 3.8 under Ubuntu 20.04. CUDA needs to be installed and libcudnn is a plus (you can get libcudnn here: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/). Note that the GeomLoss and the pykeops libraries are not available under Windows, thus scripts using them can't be executed under this OS. 

All the Python dependencies can be installed using the following command (we recommend using a fresh new Python virtual environment):
```
pip install numpy==1.23.3 matplotlib==3.6.0 scipy==1.9.1 Pillow==9.2.0 scikit-image==0.19.3 imageio==2.22.0 h5py==3.7.0 opencv-contrib-python==4.6.0.66 progress==1.6 requests==2.28.1 flickrapi==2.4.0 torch==1.9.1 torchvision==0.10.1 tensorflow==2.7.0 tensorflow-addons==0.17.1 POT==0.8.2 pykeops==1.5 geomloss==0.2.4
```

We also provide a requirements.txt. Please note that it is important to use certain versions for certain libraries. 

## Quick Start
In this section we provide an example of using our model for its primary function: interpolating between multiple - black and white - images using different barycentric weights. To do so:
1. go into the folder `experiments`;
2. execute `./quick_start.sh [IDS_LIST] [WEIGHTS_LIST]` where:
    - `[IDS_LIST]` corresponds to the list of the names of the images you want to interpolate. We assume images use the .png format and have been placed in the `input_imgs` folder. You can use the images we already provide in it or use your own 512x512 images. Note that our implementation considers 1 (white color) as signaling the presence of mass while 0 (black color) corresponds to its absence;
    - `[WEIGHTS_LIST]` corresponds to the list of the barycentric weights associated to the previous images;
    - example: `./quick_start.sh "car1,owl1" "0.5,0.5"` will use our model to produce the approximation of the Wasserstein barycenter of the images `car1.png` and `owl1.png` with the barycentric weights 0.5 associated to car1.png and 0.5 for owl1.png. You can use as many images as you want. 
3. results are stored in `experiments/results`. By default, we provide a comparison of the prediction of our model VS the barycenter computed using GeomLoss. 

Note that the model used to produce this interpolation corresponds to the one shown in our paper, trained on our synthetic dataset of 2D shapes. Its weights are stored in `training_model/results/bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_dsv2`, with some visualisations. 

In the following sections, we explain how to train our model from scratch and how to reproduce the experiments of our paper using the code of this github. 

## Synthetic training data
### Downloading the datasets
We provide the datasets we used in our paper on Zenodo. You can download them (5Gb) [on Zenodo](https://zenodo.org/record/7185972#.Y1ZVI0pBxH5) and extract the archive directly at the root of the project (ie the datasets should be placed in a folder `learning-to-generate-wasserstein-barycenters/datasets`). 

If you wish to generate your own synthetic data, follow the 2 next sections; else you can skip them and directly go to [this part](#Training-a-model). 

### 2D random shapes generation
To generate your own synthetic dataset of 2D random shapes similar to the one used in our paper:
1. go into `2d_shapes_generation`;
2. open run.sh and modify the arguments to your convenience, in particular:
    - the number of shapes `n_shapes`;
    - the image size `img_size`;
    - the multiprocessing argument `nprocs`.
3. execute `run.sh`;
4. resulting datasets are stored in `../datasets`:
    - the main dataset `input_contours.h5`;
    - a downsampled version (28x28) `input_contours_28x28.npy` of the previous dataset
   
   A set of visualizations is also generated in `2d_shapes_generation`: grid of samples, histogram of the depths of the CSG trees corresponding to each shape. 

### Barycenters generation
Once 2D shapes have been generated, they can be used to produce barycenters. To do so:
1. go into `barycenter_generation`;
2. open `run.sh` and modify the arguments to your convenience, in particular:
    - the image size `img_size` which should be the same as previously;
    - the number of 2D random shapes used to generate 1 barycenter `n_inputs`;
    - the number of desired barycenters `n_barys`.
3. execute `run.sh`:
    - during the execution a warning about `torch.meshgrid` will be displayed: do not consider it. 
4. two .h5 files will be generated in `../datasets`:
    - `targets_contours.h5` containing the set of "target points", ie approximated optimal transport maps between the uniform distribution and each input shape;
    - `barycenters_contours.h5` containing the barycenters which are random combinations of these target points.
    
    In `barycenters_generation`, this also produces a histogram showing the distribution of the barycentric weights used to compute barycenters and a figure showing some samples from the dataset of barycenters. 

In `generate_multi_iters_bary.py`, we also provide an example of generation of barycenters which uses a greater number of descent steps (which is only 1 for the barycenters produced previously). 

## Training a model
In order to be able to reproduce the experiments of the paper, we provide a set of pretrained models in `training_model/results`:
* `bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_dsv2` corresponds to the model trained on synthetic contour shapes;
* `bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_flickr` corresponds to the model trained on chrominance histograms from Flickr images. 

If you wish to use one of them, directly go to the [experiments part](#Experiments). Else to train a model from scratch: 
1. go into `training_model`; 
2. open `train_bary_model.py` and adjust the parameters, in particular:
    - `n_data` the total number of pairs (inputs, barycenter) to use for the training, validation and testing steps; 
    - `batch_size` the batch size; 
    - `get_flags` the function returning a dictionary containing all the parameters related to the model such as the number of epochs, the training schedule to use, etc. 
3. execute `python train_bary_model.py`
4. once the training is done, the model is stored with a set of visualizations (in particular the evolution of different metrics and the comparison between real and predicted barycenters) in `training_model/results` in its own sub-folder. 

## Experiments
### Interpolations and runtimes experiments
Once our model has been trained, sketch interpolation and runtimes experiments conducted in our paper can be reproduced as follows:
1. go into `experiments`;
2. open `create_geomloss_targets.sh`:
    - this script will create the target points associated to all the .png files stored in `experiments/input_imgs`, which can be then used to compute GeomLoss barycenters. Some black and white images are already provided in this folder but you can add your own images;
    - also note that in the images we use, the mass is represented by 1 (white) while its absence is represented by 0 (black);
    - if you don't use 512x512 images, you have to change the value of the parameter `m`. 
4. execute `create_geomloss_targets.sh`;
5. the generated target points are stored in the folder `input_targets`;
6. open `run_interpolate_XP.sh`
    - this script contains the set of commands letting you reproduce the interpolations experiments of the paper and provide some additional ones, with different display modes (line, triangle, pentagon, animation...);
    - you can modify the parameters of these experiments or add new ones. However, note that the main parameters (for instance the model to use) are stored in `run_all.sh`.
7. open `run_runtimes_XP.sh`
    - this script executes the runtimes experiments, on GeomLoss and on the model specified in `run_all.sh`. There is nothing to modify in it.
8. open `run_all.sh` and modify the parameters, in particular:
    - `model_id` corresponds to the name of the sub-folder containing the model, in `training_model/results`; 
    - `model_class` should be set to `BarycentricUNet`, unless you want to use a model different from the one we propose;
    - `results_XPs_path` corresponds to the path to the folder where the results of the experiments are going to be stored. It can be set for instance to `../training_model/results/$model_id/experiments`;
    - `input_imgs_path` corresponds to the folder where black and white images are stored, by default set to `input_imgs`;
    - `targets_path` is the folder where the target points computed during the step 4 have been stored, by default `input_targets`;
    - `contours_input_path` is the full path towards the .h5 file containing the synthetic 2D shapes (see "2D random shapes generation"), and is by default `../datasets/input_contours.h5`;
    - `contours_barys_path` is the full path towards the .h5 file containing the barycenters (see "Barycenters generation"), and is by default  `../datasets/barycenters_contours.h5`;
    - `n_barys_runtimes` is the number of barycenters to use in the runtimes experiments. 
9. execute `run_all.sh`: this will execute both the runtimes and the interpolations experiments;
10. the results of the experiments are stored in the folder specified by the variable `results_XPs_path`, which is by default set to `../training_model/results/$model_id/experiments`.  

### Comparison between DWE and our model

The comparison between our model and the Deep Wasserstein Embedding (DWE) method from https://arxiv.org/pdf/1710.07457.pdf can be done by first training such a model using the different .sh scripts we provide in the dwe folder or by using the pretrained DWE models provided in `dwe/models`:
* each DWE model is represented by 5 sub-folders `[MODEL_ID]_autoencoder`, `[MODEL_ID]_dwe`, `[MODEL_ID]_emd`, `[MODEL_ID]_feat`, `[MODEL_ID]_unfeat` where `[MODEL_ID]` corresponds to the id of the DWE model used;
* `[MODEL_ID]` = `randshapes` corresponds to the pretrained modified DWE model, adapted to handle 512x512 images and trained on our synthetic dataset;
* `[MODEL_ID]` = `randshapes_28x28` corresponds to the pretrained original DWE model, trained on our synthetic dataset downsampled from 512x512 to 28x28.

Please also note that the code inside the dwe folder corresponds to a modified version of https://github.com/mducoffe/Learning-Wasserstein-Embeddings. The original implementation of DWE was in Python 2, we adapted it to Python 3 and performed multiple modifications. 

#### Generation of ground truth Wasserstein distances for medium or high resolution images
If you want to train your own DWE model, you will have to consider the Wasserstein distances between each pair of inputs of your dataset. If you are using the barycenters dataset we provide, the corresponding Wasserstein distances dataset is also available in the `datasets` folder. 

If you wish to compute Wasserstein distances, we propose 2 options using EMD or GeomLoss. If the resolution of your images is small enough (28x28 for instance) we recommend to use EMD; else you will probably have to use GeomLoss to compute these distances (this is what we do for the 512x512 images). If you use GeomLoss, follow the following steps (else go directly to the [next part](#Training-a-DWE-model)):
1. go into `wdists_generation`;
2. open `run_generate_wdists.sh` and adjust the following parameters:
    - `input_shapes_path` the full path towards the .h5 file containing the synthetic 2D shapes (see "2D random shapes generation");
    - `barys_path` the full path towards the .h5 file containing the barycenters (see "Barycenters generation");
    - `wdists_path` the full path towards the .h5 file that will be generated and that will contain the Wasserstein distances;
    - `batch_size` and `chunk_size` can usually keep their default values. 
3. execute `run_generate_wdists.sh`;
4. the wasserstein distances are now stored into the .h5 file specified by the parameter `wdists_path`.

#### Training a DWE model
If you are using one of the provided pretrained DWE models, directly go to the [next part](#Comparison-of-the-approximation-errors-of-our-model-VS-DWE). 

To train your own DWE model, you can modify and run one of the provided .sh scripts in the `dwe` folder. Note that for the ones which do not consider GeomLoss, we use `run_emd.py` at the beginning of the script to generate the Wasserstein distances with EMD. Let's for instance consider that we want to train a DWE model adapted to 512x512 images on our synthethic 512x512 dataset with Wasserstein distances generated with GeomLoss:
1. go into `dwe`;
2. open `run_randshapes.sh` and consider the following parameters:
    - `dataset_id` the id of the dataset used, here "randshapes";
    - `shapes_fpath` the path to the .h5 dataset containing the 2D synthetic shapes;
    - `wdists_fpath` the path to the .h5 dataset containing the Wasserstein distances between pairs of 2D shapes computed with GeomLoss;
    - `nd` the number of barycenters used for the training of the model;
    - `bs` the batch size;
    - `epochs` the number of epochs for the training.
3. note that we already provide the pretrained DWE models for "randshapes" and "randshapes_28x28": if you train a DWE model for one of these datasets, it will **erase** the pretrained models we provide. If you want to keep these pretrained versions, don't forget to make a backup;
4. execute `run_randshapes.sh`:
    - during the execution a wordy warning about sharding for run_randshapes will be displayed: do not consider it. 
5. The trained DWE model will be saved in models under the name `randshapes_...`. 

For a dataset where we choose to generate Wasserstein distances with EMD, the steps will be similar to the previous ones, with some differences. For instance for an original DWE model on our synthetic dataset downsampled from 512x512 to 28x28:
1. go into `dwe`;
2. copy `input_contours_28x28.npy` from `../datasets` to `data` (create the folder if it does not already exist) and rename it `randshapes_28x28.npy`; 
3. open `run_randshapes_28x28.sh` and consider the following parameters:
    - `dataset_id` the id of the dataset used, here "run_randshapes_28x28";
    - `train_nd`: the original implementation of DWE stores the training set into a set of .mat files in the folder `dwe/data`. `train_nd` corresponds to the number of tuples (index shape 1, index shape 2, EMD wasserstein distance) per file;
    - `train_nfiles`: the number of files used to store the full training dataset;
    - `eval_nd`: same as `train_nd` but for the evaluation of the model;
    - `eval_nfiles`: same as `train_nfiles` but for the evaluation of the model;
    - `emb_size` the size of the embedding used in the original DWE model, 50 by default;
    - `bs` the batch size;
    - `epochs` the number of epochs for the training.
4. note that we already provide the pretrained DWE models for "randshapes" and "randshapes_28x28": if you train a DWE model for one of these datasets, it will **erase** the pretrained models we provide. If you want to keep these pretrained versions, don't forget to make a backup;
5. execute `run_randshapes_28x28.sh`:
    - during the execution a wordy warning about sharding for run_randshapes will be displayed: do not consider it. 
6. the trained DWE model will be saved in models under the name `run_randshapes_28x28_...` and by default some visualisations will be created in `dwe/imgs`. 

For `run_googledoodle.sh`, the previous steps apply, at the exception of the step 2 where you will instead have to download the cat.npy dataset at https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap and put cat.npy in the `data` folder.

#### Comparison of the approximation errors of our model VS DWE
Once the dataset has been created and a DWE model has been trained - or with a pretrained DWE model - we can compute a comparison of the approximation errors of our model VS DWE:
1. go into `experiments`;
2. open `run_compare_error_models`, the important arguments here are:
    - `model1_id` the id of a model following the architecture proposed in our paper;
    - `model2_id` the id of a DWE model;
    - `results_XPs_path` the path to the folder where the results are going to be stored, by default "./results";
    - `inputs_path` the full path towards the .h5 file containing the synthetic 2D shapes (see "2D random shapes generation");
    - `barys_path` "../datasets/barycenters_contours.h5" the full path towards the .h5 file containing the barycenters (see "Barycenters generation");
    - `n_barys` the number of barycenters used in the comparison.
3. execute `run_compare_error_models.sh`;
4. the results have been generated in the folder specified by `results_XPs_path`.

#### Interpolations with DWE
The interpolation experiments can also be done with a DWE model. To do so:
1. go into `experiments`;
2. open `run_all.sh` and:
    - comment the parameters associated to the BarycentricUNet model and uncomment the ones associated to the DWE model;
    - set `model_id` to the id of the DWE model you want to use for the experiments;
    - comment the line associated to the runtimes experiment. 
3. execute `run_all.sh`;
4. by default the results are stored in `./results/DWE_$model_id`.

### Color transfer experiments
#### Downloading and preprocessing the Flickr dataset
We already provide a preprocessed Flickr dataset in `datasets/flickr` but if you wish to make your own, follow these steps:
1. go into `flickr_preprocessing`;
2. open `download_flickr_imgs.py` and replace the values of `KEY` and `SECRET` with your own Flickr API keys (see https://www.flickr.com/services/api/misc.api_keys.html to get them);
3. execute `python dowload_flickr_imgs.py`;
4. flickr images of different categories (stored in `categories.txt`) will be downloaded in `datasets/flickr/imgs`;
5. execute `python compute_chroma_bounds.py` to compute and save the chrominance space boundaries into `ab_bounds.npy`;
6. open `imgs_to_chrom_histos.py` and set the parameter `n_histos` (at the beginning of the `main` function) to the number of chrominance histograms you want in your dataset;
7. execute `python imgs_to_chrom_histos.py`;
8. the chrominance histograms are stored into `datasets/flickr/input_chrom_histos_$n_histos.h5`. 

Then to generate barycenters, follow the steps explained in [this part](#Barycenters-generation), using the relevant paths for your datasets (a commented example is provided in `run.sh`). 

#### Training a model on Flickr
You can then train a model of your choice on this dataset as it has been explained in [this part](#Training-a-model), or consider a pretrained model. We provide our model pretrained on the Flickr dataset in `training_model/results/bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_flickr`. 

#### Color Transfer
Finally you can reproduce the color transfer experiments by following the next steps:
1. go into `experiments`;
2. open `chrom_histograms_interpolation.py` and modify the parameters at the beginning of the `main` function, in particular: `input_folder`, `img_ids`, `n_iters`, `base_folder`, `model_ids`, `model_classes`. We refer to the comments written in the code for more details about these variables;
3. execute `python chrom_histograms_interpolation.py`
    - during the execution multiple warnings about the color range will be displayed: do not consider them. 
4. by default, results are stored in the `experiments/chrominance_histograms` folder. 
