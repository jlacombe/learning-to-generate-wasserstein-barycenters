import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn

from model_util import *

class BarycentricUNet(nn.Module):
    def __init__(self, nf=16, norm_params=None, with_skip_co=True):
        super(BarycentricUNet, self).__init__()
        self.with_skip_co = with_skip_co
        
        # external contractive path
        self.external_encoder = nn.Sequential(                    # =>   1 x 512 x 512 (262 144)
           *bloc_conv_relu(    1,    nf, 3, 1, 1, norm_params=norm_params), # =>  16 x 512 x 512
           *bloc_conv_relu(   nf,    nf, 3, 1, 1, norm_params=norm_params), # =>  16 x 512 x 512
              nn.AvgPool2d(2, 2),                                           # =>  16 x 256 x 256
        
           *bloc_conv_relu(   nf,  nf*2, 3, 1, 1, norm_params=norm_params), # =>  32 x 256 x 256
           *bloc_conv_relu( nf*2,  nf*2, 3, 1, 1, norm_params=norm_params), # =>  32 x 256 x 256
              nn.AvgPool2d(2, 2),                                           # =>  32 x 128 x 128
        
           *bloc_conv_relu( nf*2,  nf*4, 3, 1, 1, norm_params=norm_params), # => 64 x 128 x 128
           *bloc_conv_relu( nf*4,  nf*4, 3, 1, 1, norm_params=norm_params), # => 64 x 128 x 128
              nn.AvgPool2d(    2,     2),                                   # => 64 x  64 x  64
        
           *bloc_conv_relu( nf*4,  nf*8, 3, 1, 1, norm_params=norm_params), # => 128 x  64 x  64
           *bloc_conv_relu( nf*8,  nf*8, 3, 1, 1, norm_params=norm_params), # => 128 x  64 x  64
              nn.AvgPool2d(2, 2),                                           # => 128 x  32 x  32
            
           *bloc_conv_relu( nf*8, nf*16, 3, 1, 1, norm_params=norm_params), # => 256 x  32 x  32
           *bloc_conv_relu(nf*16, nf*16, 3, 1, 1, norm_params=norm_params), # => 256 x  32 x  32
              nn.AvgPool2d(2, 2),                                           # => 256 x  16 x  16
        )
        
        # inner contractive path
        self.inner_encoder = nn.Sequential(
           *bloc_conv_relu(nf*16, nf*16, 3, 1, 1, norm_params=norm_params), # => 512 x   8 x   8 (32 768)
           *bloc_conv_relu(nf*16, nf*16, 3, 1, 1, norm_params=norm_params), # => 512 x   8 x   8 (32 768)
        )
        
        # expansive path
        self.decoder = nn.Sequential(          
           *bloc_conv_relu(nf*16, nf*16, 3, 1, 1, norm_params=norm_params), # => 512 x   8 x   8 (32 768)
           *bloc_conv_relu(nf*16, nf*16, 3, 1, 1, norm_params=norm_params), # => 512 x   8 x   8
                  UpSample(2),                                              # => 512 x  16 x  16
        
           *bloc_conv_relu(nf*16*(2 if self.with_skip_co else 1), 
                           nf*16, 3, 1, 1, norm_params=norm_params), # => 256 x  32 x  32
           *bloc_conv_relu(nf*16,  nf*8, 3, 1, 1, norm_params=norm_params), # => 128 x  32 x  32
                  UpSample(2),                                              # => 128 x  64 x  64
        
           *bloc_conv_relu(nf*8*(2 if self.with_skip_co else 1),  
                           nf*8, 3, 1, 1, norm_params=norm_params), # => 128 x  64 x  64
           *bloc_conv_relu( nf*8,  nf*4, 3, 1, 1, norm_params=norm_params), # =>  64 x  64 x  64
                  UpSample(2),                                              # =>  64 x 128 x 128
        
           *bloc_conv_relu( nf*4*(2 if self.with_skip_co else 1), 
                           nf*4, 3, 1, 1, norm_params=norm_params), # =>  64 x 128 x 128
           *bloc_conv_relu( nf*4,  nf*2, 3, 1, 1, norm_params=norm_params), # =>  32 x 128 x 128
                  UpSample(2),                                              # =>  32 x 256 x 256
        
           *bloc_conv_relu( nf*2*(2 if self.with_skip_co else 1),  
                           nf*2, 3, 1, 1, norm_params=norm_params), # =>  32 x 256 x 256
           *bloc_conv_relu( nf*2,    nf, 3, 1, 1, norm_params=norm_params), # =>  16 x 256 x 256
                  UpSample(2),                                              # =>  16 x 512 x 512
        
           *bloc_conv_relu( nf*(2 if self.with_skip_co else 1),    
                           nf, 3, 1, 1, norm_params=norm_params), # =>  16 x 512 x 512
                 nn.Conv2d(   nf,     1, 3, 1, 1),                # =>   1 x 512 x 512 (262 144)
        )
    
    def compute_external_contractive_path(self, x, bary_weight):
        if (self.with_skip_co):
            skip_connections_empty = (self.skip_connections == [])
            skip_index = 0
        for layer in self.external_encoder:
            if (self.with_skip_co and isinstance(layer, nn.AvgPool2d)):
                if (skip_connections_empty):
                    self.skip_connections.append(x * bary_weight)
                else:
                    self.skip_connections[skip_index] =   self.skip_connections[skip_index] \
                                                        + x * bary_weight
                    # note: we do an euclidean mean using '+' operator but other symmetrical 
                    # operations could be studied
                    skip_index += 1
            x = layer(x)
        return x

    def compute_contractive_path(self, x, bary_weight):
        if (self.training):
            # when we are training the model, we use gradient checkpointing to reduce
            # memory consumption. The bottleneck of this network is its expanding part:
            # for n input shapes, we do n encodings. By checkpointing each encoding, 
            # we ensure we still have enough memory when n grows.
            external_activs = checkpoint(self.compute_external_contractive_path, x, bary_weight)
            inner_activs = checkpoint(self.inner_encoder, bary_weight * external_activs)
        else:
            external_activs = self.compute_external_contractive_path(x, bary_weight)
            inner_activs = self.inner_encoder(bary_weight * external_activs)
        return inner_activs
    
    def compute_all_contractive_paths(self, x, bary_weights):
        inner_activs = self.compute_contractive_path(x[:,0:1], bary_weights[:,0:1]) * bary_weights[:,0:1]
        for i in range(1,x.shape[1]):
            inner_activs = inner_activs + self.compute_contractive_path(x[:,i:i+1], bary_weights[:,i:i+1]) * bary_weights[:,i:i+1]
            # note: we do an euclidean mean using '+' operator but other symmetrical 
            # operations could be studied
        return inner_activs
    
    def compute_external_path(self, x):
        if (self.with_skip_co):
            skip_index = -1
        for layer in self.decoder:
            x = layer(x)
            if (self.with_skip_co and isinstance(layer, UpSample)):
                x = torch.cat([self.skip_connections[skip_index], x], axis=1)
                skip_index -= 1
        return x
        
    def forward(self, x, bary_weights):
        if (self.with_skip_co):
            self.skip_connections = []
        inner_activs  = self.compute_all_contractive_paths(x, bary_weights)
        bary = self.compute_external_path(inner_activs)
        
        bary = F.softmax(
                   bary.reshape(*bary.shape[:-2], -1), dim=2
               ).reshape(bary.shape)
        
        return bary

    def backward(self, grad_out):
        grad_in = super(BarycentricUNet, self).backward(x)
        if (self.with_skip_co):
            del self.skip_connections # to avoid GPU memory leak
        return grad_in


