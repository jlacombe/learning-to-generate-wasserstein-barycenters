# pykeops==1.4.0 + geomloss + torch==1.4.0 (nok with torch>=1.5)
import os
import sys
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from geomloss import SamplesLoss

# allows imports from parent directory
sys.path.insert(0, os.path.dirname(os.getcwd()))

from bary_gen_util import *
from io_util import analyse_args

def get_img_from_path(img_path):
    img = Image.open(img_path).convert('L')
    img = ToTensor()(img)[0]
    img = img / img.sum()
    return img

def main():
    args = analyse_args([
        ['r',    'results_path', lambda x: x, 'results'],
        ['f',    'input_folder', lambda x: x, 'input_imgs'], 
        ['l',        'imgs_ids', lambda x: x.split(','), []],
        ['t',  'targets_folder', lambda x: x, 'input_targets'], 
        ['n',               'N', lambda x: int(x), 512], 
        ['m',               'M', lambda x: int(x), 512]
    ])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    eps = 1e-20 # value used to replace 0, in order to avoid numerical instability when computing targets
    
    Loss = SamplesLoss('sinkhorn', blur=.01, scaling=.9, backend='multiscale')

    bary_coords = grid(args['N'], device).view(-1,2)
    bary_weights = (torch.ones(args['N']*args['N']) / (args['N']*args['N'])).type_as(bary_coords)
    bary_coords.requires_grad = True
    
    if args['imgs_ids'] == []:
        args['imgs_ids'] = os.listdir(args['input_folder'])
    else:
        args['imgs_ids'] = [x + '.png' for x in args['imgs_ids']]
    
    for fname in args['imgs_ids']:
        print(fname)
        fpath = os.path.join(args['input_folder'], fname)
        img_weights = get_img_from_path(fpath)
        img_weights = img_weights.float().to(device)
        img_weights[img_weights == 0] = eps
        img_weights = img_weights.view(-1)
        img_coords = grid(args['M'], device)
        
        loss_val = Loss(bary_weights, bary_coords, img_weights, img_coords)
        [gradient] = torch.autograd.grad(loss_val, [bary_coords])
        target = bary_coords - gradient / bary_weights.view(-1,1)
        target = target.detach().cpu()
        
        torch.save(target, os.path.join(args['targets_folder'], fname[:-4] + '.pt'))
    

if __name__ == "__main__":
    main()
