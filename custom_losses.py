import torch
from torch import nn
from torch.nn.functional import kl_div

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
