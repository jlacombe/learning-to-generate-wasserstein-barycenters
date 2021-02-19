import numpy as np
import random
import datetime
import os
import pickle as pkl
import matplotlib.pyplot as plt
import torch
from torch import nn

class UpSample(nn.Module):
    def __init__(self, scale):
        super(UpSample, self).__init__()
        self.scale = scale
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale)

def bloc_conv_activ(n_in, n_out, kernel_size, stride_size, padding=0, dilation=1,
                    activation=nn.ReLU(), convt=False, norm=True, norm_params=None):
    if norm and (norm_params is None):
        norm_params = [BatchNorm2d, 1e-5, 0.1, True, True] # default Pytorch batch norm params 

    bloc = []
    if convt:
        bloc.append(nn.ConvTranspose2d(n_in, n_out, kernel_size, stride_size, padding, dilation=dilation))
    else:
        bloc.append(nn.Conv2d(n_in, n_out, kernel_size, stride_size, padding, dilation=dilation))
        
    if (norm):
        bloc.append(norm_params[0](n_out, *norm_params[1:]))
        
    bloc.append(activation)
    return bloc

def bloc_conv_relu(n_in, n_out, kernel_size, stride_size, padding=0, 
                   dilation=1, inplace=True, norm=True, norm_params=None):
    return bloc_conv_activ(n_in, n_out, kernel_size, stride_size, padding, dilation=dilation,
                           activation=nn.ReLU(), convt=False, norm=norm, norm_params=norm_params)

def set_random_seed(seed, use_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if use_cuda: 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_and_trace(txt, model_fold_path, fname='trace.txt', mode='a'):
    print(txt)
    with open(os.path.join(model_fold_path, fname), mode) as f:
        f.write(txt + '\n')

def display_loss(epochs, train_loss, eval_loss, save=True, 
                 folder_path=os.path.join('results', 'default')):
    plt.figure()
    plt.plot(epochs, train_loss, 'c--', label='Train Loss')
    plt.plot(epochs, eval_loss, 'c', label='Valid Loss')
    plt.title('Evolution of loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    if (save):
        plt.savefig(os.path.join(folder_path, 'loss.png'))
    plt.show()

def get_nparams_model(model):
    return sum([p.numel() for p in list(model.parameters())])

def init_model_folder(parent_res_fold_name='results', res_fold_name=None):
    now = datetime.datetime.now()
    if res_fold_name is None:
        res_fold_name = '{}_{}_{}_{}_{}_{}'.format(now.year, now.month, now.day,
                                                   now.hour, now.minute, now.second)
    model_fold_path = os.path.join(parent_res_fold_name, res_fold_name)
    
    if not os.path.exists(parent_res_fold_name):
        os.makedirs(parent_res_fold_name)
        
    if not os.path.exists(model_fold_path):
        os.makedirs(model_fold_path)
        
    return model_fold_path

def save_model(model, FLAGS, folder_path=os.path.join('results', 'default')):
    with open(os.path.join(folder_path, 'flags.pkl'), 'wb') as f:
        pkl.dump(FLAGS, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    torch.save(model.state_dict(), os.path.join(folder_path, 'model.pth'))
    
def load_flags(flags_path):
    with open(flags_path, 'rb') as f:
        FLAGS = pkl.load(f)
    return FLAGS

def flags_to_str(FLAGS):
    str_repr = ''
    lay_flags = ['lay_sizes', 'enc_sizes', 'dec_sizes', 'features', 'classifier']
    for k, v in FLAGS.items():
        if (k not in lay_flags):
            str_repr += '{:17}: {}\n'.format(k, v)
    for lay_flag in lay_flags:
        if lay_flag in FLAGS:
            str_repr += '{:17}: [\n'.format(lay_flag)
            for lay_size in FLAGS[lay_flag]:
                str_repr += '    {}\n'.format(lay_size)
            str_repr += ']\n'
    return str_repr

def display_flags(FLAGS):
    print(flags_to_str(FLAGS))

def load_model_on_gpu(model_path, model_class, model_args):
    device = torch.device('cuda:0')
    model = model_class(*model_args)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

def load_model_on_cpu(model_path, model_class, model_args):
    model = model_class(*model_args)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model
