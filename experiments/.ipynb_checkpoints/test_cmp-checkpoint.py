import os
import sys
import torch
import tensorflow as tf
import numpy as np
from matplotlib.animation import PillowWriter
from skimage.transform import resize

# allows imports from parent directories
sys.path.insert(0, os.path.dirname(os.getcwd()))

from data_loading_util import *
from training_util import *
from custom_losses import *
from models import *
from model_util import *
from barycentric_polygon_comparison import *
from bary_gen_util import *
from anim_util import *
from io_util import analyse_args

model_id = '2022_9_28_15_17_45'
shapes_fpath = os.path.join('..', 'datasets', 'input_contours.h5')
barys_fpath  = os.path.join('..', 'datasets', 'barycenters_contours.h5')

train_part, valid_part, test_part = 0.8, 0.1, 0.1
n_data = 1000
batch_size = 8

dataloaders = load_geomloss_barys(shapes_fpath, barys_fpath, train_part, 
                                  valid_part, test_part, n_data, batch_size)

model_results_path = os.path.join('..', 'training_model', 'results', model_id)
flags_path = os.path.join(model_results_path, 'flags.pkl')
model_path = os.path.join(model_results_path, 'model.pth')
FLAGS = load_flags(flags_path)
model = load_model_on_gpu(model_path, BarycentricUNet, 
                          [FLAGS[param_name] for param_name in 
                                                 FLAGS['model_params_names']]).eval()

compare_real_pred(dataloaders['test_loader'], model, 
                  n_samples=5, parent_folder_path='test_cmp', cmap='binary', 
                  with_inputs=True, show=False)

