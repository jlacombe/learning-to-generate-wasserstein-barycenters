import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import h5py
from geomloss import SamplesLoss

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

from io_util import analyse_args
from bary_gen_util import *

def main():    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    N, M = 512, 512

    inputs_ids = [0,1] # the ids of the 2d shapes inputs used
    bary_weights = [.5,.5]
    ngrad_iters = 10

    # RANDOM CONTOURS DATASET
    dataset_base_path = os.path.join('..', 'datasets')
    input_shapes_path = os.path.join(dataset_base_path, 'input_contours.h5')
    
    with h5py.File(input_shapes_path, 'r') as f:
        dset = f['input_shapes']
        input_measures = [as_measure(torch.from_numpy(dset[i]).float().to(device), M, device) 
                          for i in inputs_ids]

    Loss = SamplesLoss("sinkhorn", blur=.01, scaling=.9)
    bary = gen_barycenter(input_measures, bary_weights, Loss, M, N, device, ngrad_iters)

    # to display a barycenter on the unit square, we need to use a binning algorithm
    bin_bary = bin_barycenter(bary, M).cpu().detach()

    # as a pdf
    bin_bary /= bin_bary.sum()
    
    vmin = np.percentile(bin_bary[bin_bary>0.], 5)
    vmax = np.percentile(bin_bary[bin_bary>0.], 95)
    
    plt.figure(figsize=(1,1),dpi=512)
    plt.imshow(bin_bary,vmin=vmin,vmax=vmax,cmap='Greys')
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.savefig('multi_grad_iters_bary_{}_{}_{}_{}_{}.png'.format(*inputs_ids, *bary_weights, ngrad_iters))
    
        
if __name__ == "__main__":
    main()
