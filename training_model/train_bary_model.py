import sys
import os
import torch
import matplotlib.pyplot as plt
import time
from torch import nn, optim

# allows imports from parent directory
sys.path.insert(0, os.path.dirname(os.getcwd()))

from models import *
from model_util import *
from data_loading_util import *
from training_util import *
from custom_losses import *

def get_flags(batch_size):
    return {
                  'batch_size': batch_size,
                      'epochs': 31,
                          'nf': 32,
        
                        'loss': kldiv_loss,
        
               'eval_metrics_names': ['KL Divergence', 'Sparsity'],
                'eval_metrics': [kldiv_loss, sparsity],
        
        
#                  'norm_params': [nn.BatchNorm2d, 1e-20, 0.1,     # [norm type ; eps ; momemtum ;
#                                  True, True],                    #  affine, track_running_stats]
                 'norm_params': [nn.InstanceNorm2d, 1e-20, 0.1, # [norm type ; eps ; momemtum ;
                                 True, False],                  #  affine, track_running_stats]
        
            'optimizer_params': [optim.SGD, 5e-4, 0.99, # [optim type ; lr ; momemtum]
                                 0, 0, True],           #  dampening ; weight_decay ; nesterov]
        
            'scheduler_params': [optim.lr_scheduler.CosineAnnealingWarmRestarts, # [scheduler type ;
                                 10, 2, 0, -1],                                  #  T_0 ; T_mult ; eta_min ; last_epoch]
        
          'scheduler_interval': 0.1, # given a scheduler, lr is updated every scheduler_interval % epoch
#                   'stop_iter': int(400000*1.5),
        
               'eval_interval': 0.1, # evaluation every dataset_size * eval_inter iterations
        
                  'model_type': BarycentricNet,
                'with_skip_co': True,
          'model_params_names': ['nf', 'norm_params', 'with_skip_co']
          }

def one_shot_training(FLAGS, dataloaders):
    model = FLAGS['model_type'](*[FLAGS[param_name] for param_name in FLAGS['model_params_names']])
    return train_model(FLAGS, dataloaders, model, save=True, parent_res_fold_path='results', show=False)

def main():
    shapes_fpath = os.path.join('..', 'datasets', 'input_contours.h5')
    barys_fpath  = os.path.join('..', 'datasets', 'barycenters_contours.h5')
    
    train_part, valid_part, test_part = 0.8, 0.1, 0.1
    n_data = 100000
    batch_size = 8
    dataloaders = load_geomloss_barys(shapes_fpath, barys_fpath, train_part, 
                                      valid_part, test_part, n_data, batch_size)
    
    set_random_seed(1, True) # for reproducibility
    FLAGS = get_flags(batch_size)
    trained_model, res_fold_path, mean_test_loss = one_shot_training(FLAGS, dataloaders)
    
if __name__ == "__main__":
    main()
