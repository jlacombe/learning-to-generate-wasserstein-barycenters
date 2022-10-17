import torch
from torch import nn
from torch.nn.functional import kl_div
import numpy as np
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

l1_loss = nn.L1Loss()
bce_loss = nn.BCELoss()

def sparsity(pred, real, eps=1e-10, sparsity_expo=0.5): # 2nd argument is useful due to the way 
                                                        # we process eval metrics
    # sparsity metric used: lpp pseudo-norm with p=1/2
    return torch.mean(torch.sum((pred+eps).pow(sparsity_expo), axis=(1,2,3)))

def bce_with_sparsity(pred, real, sparsity_weight=1e-8):
    return nn.BCELoss()(pred, real) + sparsity_weight * sparsity(pred, real)

def kldiv_loss(pred, real, eps=1e-10):
    pred = torch.clamp(pred, eps, 1.)
    real = torch.clamp(real, eps, 1.)
    return kl_div(pred.log(), real, reduction='batchmean')

def kldiv_loss_with_sparsity(pred, real, kldiv_weight=1., sparsity_weight=1e-3, 
                             sparsity_expo=0.25, eps=1e-10):
    return kldiv_weight * kldiv_loss(pred, real, eps=eps) \
         + sparsity_weight * sparsity(pred, real, sparsity_expo=sparsity_expo)

def grid(size): # torch version
    x,y = torch.meshgrid([torch.arange(0.0,size).type(dtype)/size]*2)
    return torch.stack((x,y),dim=2).view(-1,2)

def img2measure(a,size):
    a = a/255
    a = a.squeeze()
    weights = a.type(dtype)
    weights = weights/weights.sum()
    samples = grid(np.shape(a)[0])
    return weights.view(-1),samples

def mmd_loss(a,b):
    weights_a,samples_a = img2measure(a,512)
    weights_b,samples_b = img2measure(b,512)
    Loss = SamplesLoss('energy', blur=0.01, scaling=0.9)
    return Loss(weights_a, samples_a, weights_b, samples_b)
