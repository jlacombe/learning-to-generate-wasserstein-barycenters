import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.functional import avg_pool2d
from torchvision.transforms import Grayscale, Resize, Compose, ToTensor

def gen_random_distrib(n_vals):
    rnd_distrib = np.exp(np.random.randn(n_vals))
    return rnd_distrib / rnd_distrib.sum()

def gen_random_tuple(minval, maxval, tuple_size):
    vals = []
    for i in range(tuple_size):
        val = random.randint(minval, maxval)
        while (val in vals):
            val = random.randint(minval, maxval)
        vals.append(val)
    return tuple(vals)

def gen_bary_from_targets(targets, weights, img_size):
    # compute the barycenter between targets (= wasserstein barycenter of corresponding input measures)
    targets_bary = targets[0] * weights[0]
    for i in range(1, len(targets)):
        targets_bary += targets[i] * weights[i]

    # binning: from (M*M, 2) to (M, M)
    targets_bary = targets_bary.clamp(0, 1 - .1 / img_size)
    bins = (targets_bary[:,0] * img_size).floor() + img_size * (targets_bary[:,1] * img_size).floor()
    count = bins.int().bincount(weights=None, minlength=img_size*img_size).float()

    # normalization
#         count /= count.sum() # as a probability distribution
    count /= count.max() # with values between 0 and 1

    return count.detach().view(img_size, img_size)#.cpu().numpy()

def load_targets(target_fnames, targets_folder_path):
    targets = [torch.load(os.path.join(targets_folder_path, 
                                       target_fname), map_location=torch.device('cpu'))
               for target_fname in target_fnames]
    return torch.stack(targets)

def load_img(img_path):
    img = Image.open(img_path).convert('L')
    img = ToTensor()(img)
    return img

def grid(size, device):
    '''
    Generates a size*size grid with coordinates regularly spaced over the unit square. 
    The grid is returned under the form of a (size*size)*2 list of (x,y) coordinates. 
    
    :param size: determines the size of the grid (seen as a square). 
    :returns: a (size*size)*2 list of (x,y) coordinates regularly spaced over the unit square. 
    '''
    x_vec = torch.arange(0., size, device=device, dtype=torch.float) / size
    y_vec = torch.arange(0., size, device=device, dtype=torch.float) / size
    y_mat, x_mat = torch.meshgrid([x_vec, y_vec])
    xy_coords = torch.stack((x_mat,y_mat), dim=2).view(-1,2)
    return xy_coords

def as_measure_from_path(img_path, size, device, inv_mass=False, eps=1e-20):
    '''
    Loads and transforms an image into a measure of diracs (weights + positions of the diracs). 
    Note: we expect the grid's size on which we sample to be lower than the image's one. 
    
    :param img: the array containing the image to transform. 
    :param size: the size of the grid used to sample the image as a measure. 
    :param inv_mass: determines if we need to inverse the mass of the image.
    :param eps: value replacing 0 in order to avoid numerical instability
    :returns: the weights and the corresponding positions of the diracs
    '''
    # loads the image as a grayscale resized image
    transform_op = Compose([
        Grayscale(),
        Resize(size),
        ToTensor()
    ])
    img = Image.open(img_path)
    img = transform_op(img)[0].float().to(device)
    if (inv_mass):
        img = 1 - img
    # transforms the image into a probability distribution
    weights = img / img.sum()
    return as_measure(weights, size, device, eps=eps)

def as_sparse_measure(img, device):
    x, y = torch.where(img != 0)
    locations = torch.stack((y,x),axis=1).float() / img.shape[0]
    weights = img[(x,y)]
    weights /= weights.sum()
    return weights.view(-1), locations

def as_measure(weights, size, device, eps=1e-20):
    if (weights.shape[0] != size):
        sampling = weights.shape[0] // size
        weights = avg_pool2d( weights.unsqueeze(0).unsqueeze(0), sampling).squeeze(0).squeeze(0)
    weights = weights / weights.sum()
    weights[weights == 0.] = eps
    samples = grid(size, device) # position of the diracs
    return weights.view(-1), samples
   
def gen_barycenter(input_measures, bweights, Loss, M, N, device, ngrad_iters=1, lr=1., display=False):
    '''
    Computes a Wasserstein barycenters given input measures and their corresponding barycentric
    weights, for a given number of gradient steps. 
    
    :param input_measures: the list of the measures
    '''
    bary_coords = grid(N, device).view(-1,2)
    bary_weights = (torch.ones(N*N) / (N*N)).type_as(bary_coords)
    bary_coords.requires_grad = True

    for i in range(ngrad_iters):
        displacement_vectors = [] # displacement vectors
        for measure_weights, measure_coords in input_measures:
            loss_val = Loss(bary_weights, bary_coords, measure_weights, measure_coords)
            [gradient] = torch.autograd.grad(loss_val, [bary_coords])
            displacement_vector = lr * (-gradient / bary_weights.view(-1,1))
            displacement_vectors.append(displacement_vector)
#         print(loss_val.item())
        bary_coords = bary_coords + displacement_vectors[0] * bweights[0] #+ targets[1] * bweights[1]
        
        for j in range(1,len(bweights)):
            bary_coords += displacement_vectors[j] * bweights[j]
        
        if (display):
            bin_bary = bin_barycenter(bary_coords, M).cpu().float().detach()
            bin_bary /= bin_bary.sum()
            vmin = np.percentile(bin_bary, 1)
            vmax = np.percentile(bin_bary, 99)
            plt.figure();plt.imshow(bin_bary,cmap='Greys',vmin=vmin,vmax=vmax);plt.axis('off');plt.show()
    
    return bary_coords

def bin_barycenter(bary, M): # torch version
    '''
    Given a representation of a barycenter under the form of a list of N coordinates
    in the unit square, converts it to its corresponding MxM matrix representation,
    using a binning algorithm. N may be greater that M*M. 
    
    :param bary: the list of coordinates in the unit square representing the barycenter. 
    :param M: the size of a side of the output matrix representing the barycenter. 
    :returns: the MxM matrix representation corresponding to the barycenter. 
    '''
    bary = bary.clamp(0, 1 - .1 / M)
    # we first pass from the unit square to the (M,M)
    # and we then discretize by adding each xindex 
    # to its corresponding (size of a row) * yindex. 
    xaxis_position = (bary[:,0] * M).floor()
    yaxis_position = M * (bary[:,1] * M).floor()
    bins = xaxis_position + yaxis_position
    # we count the number of times each index appear
    histo = torch.histc(bins, bins=M*M, min=0., max=M*M) # <= non differentiable /!\
    return histo.reshape(M,M)

def bin_barycenter_numpy(bary, M): # numpy version
    '''
    Given a representation of a barycenter under the form of a list of N coordinates
    in the unit square, converts it to its corresponding MxM matrix representation,
    using a binning algorithm. N may be greater that M*M. 
    
    :param bary: the list of coordinates in the unit square representing the barycenter. 
    :param M: the size of a side of the output matrix representing the barycenter. 
    :returns: the MxM matrix representation corresponding to the barycenter. 
    '''
    bary = bary.clip(0, 1 - .1 / M)
    # we first pass from the unit square to the (M,M)
    # and we then discretize by adding each xindex 
    # to its corresponding (size of a row) * yindex. 
    xaxis_position = np.floor(bary[:,0] * M)
    yaxis_position = M * np.floor(bary[:,1] * M)
    bins = (xaxis_position + yaxis_position).astype(np.int)
    # we count the number of times each index appear
    count = np.bincount(bins, minlength=M*M)
    return count.reshape(M,M)
