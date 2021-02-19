# Learning to Generate Wasserstein Barycenters
Implementation of the "Learning to Generate Wasserstein Barycenters" paper. 

## Requirements
We provide in a requirements.txt the different versions of the Python libraries we used to execute our code. 

Our implementation was developed using Python 3.8.5 with CUDA 11.0 under Ubuntu 20.04. 

## How to use our code
### Synthetic training data - 2D random shapes generation
To generate a synthetic dataset of 2D random shapes similar to the one used in our paper, you can execute the run.sh script in the 2d\_shapes\_generation folder. You can adapt the different arguments appearing at the beginning of this script to your convenience (number of shapes, multiprocessing, image size...). 

2D shapes are stored in .h5 files stored in the datasets folder. By default, we also generate a downsampled version of our dataset (512x512 => 28x28). Some visualizations are also provided: grid of samples, histogram of the depths of the CSG trees corresponding to each shape. 

### Synthetic training data - Barycenters generation
Once 2D shapes have been generated, they can be used to produce barycenters using the run.sh script in the folder barycenter\_generation. This will produce 2 .h5 files, a first containing target points which correspond to approximated optimal transport maps between a uniform distribution and an input shape, and a second which contains barycenters which are random combinations of these target points. This also produces a histogram showing the distribution of the barycentric weights used to compute barycenters and a figure showing some samples from the dataset of barycenters.

In generate\_multi\_iters\_bary.py, we also provide an example of generation of barycenters which uses a greater number of descent steps (which is 1 in the previous case). 

### Training a model
To train our model from scratch, you can use the train\_bary\_model.py file in the training\_model folder. You can adjust the training parameters directly inside this file. All the parameters related to the model are stored in a dictionary called "FLAGS". Once the training is finished, it is stored in a sub-folder of the results folder. A set of figures related to the training and to the evaluation is also stored with the model. 

Pretrained models are available in the folder training_model/results and can be directly used to reproduce the experiments of the article. bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_dsv2 corresponds to the model trained on synthetic contour shapes while bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_flickr corresponds to the model trained on chrominance histograms from Flickr images. 

### Experiments
Once our model has been trained, sketch interpolation experiments conducted in our paper can be reproduced using the following scripts:
* experiments/create_geomloss_targets.sh : lets you create target points associated to the .png files stored in experiments/input_imgs. Some images are already provided in this folder. These are then used to compute GeomLoss barycenters;
* experiments/run_all.sh : run runtimes and interpolations experiments. Interpolations experiments consist of interpolations between 2 or more inputs, with different display modes (line, triangle, pentagon, animation...). You can modify parameters of these experiments inside run_all.sh and run_interpolate_XP.sh. Interpolation results are stored inside the corresponding model folder. 

The comparison between our model and the Deep Wasserstein Embedding (DWE) method from https://arxiv.org/pdf/1710.07457.pdf can be done by first training such a model using the different .sh scripts we provide in the dwe folder, for instance dwe/run_randshapes.sh. The code inside the dwe folder corresponds to a modified version of https://github.com/mducoffe/Learning-Wasserstein-Embeddings. This DWE model also needs pairs of inputs with Wasserstein distance: this dataset can be created using the script wdists_generation/run_generate_wdists.sh
Once the dataset has been created and the DWE has been trained, you can use the experiments/run_compare_error_models.sh to obtain a comparison of the approximation errors of our model VS DWE. The interpolation experiments can also be done with DWE. 

By default, experiments are done on 512x512 images, but you can adapt this resolution. The resolution which was used in DWE was for instance 28x28. 

Finally, color transfer experiments can be reproduced following the next steps:
* first we need to create the chrominance histograms datasets. In flickr_preprocessing, use python download_flickr_imgs.py to download flickr images of different categories. Note that in download_flickr_imgs.py you have to replace the values of KEY and SECRET with your own Flickr API keys. Once all images have been correctly downloaded, use python compute_chroma_bounds.py to compute and save the chrominance space boundaries which we are going to use in the next scripts. Finally, you can use imgs_to_chrom_histos.py to create the chrominance histograms dataset;
* you can then train the model on your dataset or just consider a pretrained model;
* finally chrom_histograms_interpolation.py can be called with your own parameters (see main function). Results are stored in the experiments/chrominance_histograms folder. 


