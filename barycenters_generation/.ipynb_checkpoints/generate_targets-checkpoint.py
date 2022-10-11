import numpy as np
import os
import sys
import torch
import time
import h5py
from geomloss import SamplesLoss

sys.path.insert(0, os.path.dirname(os.getcwd()))

from io_util import analyse_args
# from data_loading_util import *
from bary_gen_util import *


def main():
    t1 = time.time()
    args = analyse_args([
        ['n',                 'N', lambda x: int(x), 512],
        ['i', 'input_shapes_path', lambda x: x, os.path.join('..', 'datasets', 'input_contours.h5')],
        ['t',      'targets_path', lambda x: x, os.path.join('..', 'datasets', 'targets_contours.h5')],
        ['b',        'batch_size', lambda x: int(x), 100],
        ['c',        'chunk_size', lambda x: int(x), 1]
    ])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dsname = args['targets_path'].split('/')[-1].split('.')[0]

    Loss = SamplesLoss('sinkhorn', blur=.01, scaling=.9, backend='multiscale')

    bary_coords = grid(args['N'], device).view(-1,2)
    bary_weights = (torch.ones(args['N']*args['N']) / (args['N']*args['N'])).type_as(bary_coords)
    bary_coords.requires_grad = True

    with h5py.File(args['input_shapes_path'], 'r') as input_file, \
         h5py.File(args['targets_path'], 'w') as output_file:
        input_dset = input_file['input_shapes']
        n_targets = input_dset.shape[0]
        output_dset = output_file.create_dataset('targets', (n_targets, *bary_coords.shape), 
                                                 compression='gzip', shuffle=True, dtype=np.float32, 
                                                 chunks=(args['chunk_size'], *bary_coords.shape))

        for i in range(0, n_targets, args['batch_size']):
            print('{}/{}'.format(i, n_targets))
            batch = input_dset[i:i+args['batch_size']]

            targets = []
            for shape_weights in batch: # a batchwise approach allows to do less disk access
                                        # & avoids to store all targets in memory
                shape_weights = torch.from_numpy(shape_weights).float().to(device)
                shape_weights, shape_coords = as_sparse_measure(shape_weights, device)

                loss_val = Loss(bary_weights, bary_coords, shape_weights, shape_coords)
                [gradient] = torch.autograd.grad(loss_val, [bary_coords])
                target = bary_coords - gradient / bary_weights.view(-1,1)
                targets.append(target.detach().cpu())
            targets = torch.stack(targets)
            output_dset[i:i+args['batch_size']] = targets
        print('{}/{}'.format(i+args['batch_size'], n_targets))
        deltat = time.time() - t1
        print('Elapsed Time: {:.2f}s for {} targets'.format(deltat, n_targets))
        with open('{}_generation_time.txt'.format(dsname), 'w') as f:
            f.write('Elapsed Time: {:.2f}s for {} targets'.format(deltat, n_targets))
    
if __name__ == "__main__":
    main()
