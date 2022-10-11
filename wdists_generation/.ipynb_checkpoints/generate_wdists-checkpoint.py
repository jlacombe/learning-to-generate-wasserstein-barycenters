import os
import sys
import numpy as np
import torch
import time
import h5py
from geomloss import SamplesLoss

sys.path.insert(0, os.path.dirname(os.getcwd()))

from io_util import analyse_args
from bary_gen_util import *

def main():
    t1 = time.time()
    args = analyse_args([
        ['b',        'batch_size', lambda x: int(x), 1000],
        ['c',        'chunk_size', lambda x: int(x), 1],
        ['i', 'input_shapes_path', lambda x: x, os.path.join('..', 'datasets', 'input_contours.h5')],
        ['r',        'barys_path', lambda x: x, os.path.join('..', 'datasets', 'barycenters_contours.h5')],
        ['w',       'wdists_path', lambda x: x, os.path.join('..', 'datasets', 'wdists_contours.h5')],
    ])
    dsname = args['wdists_path'].split('/')[-1].split('.')[0]
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sinkhorn_loss = SamplesLoss('sinkhorn', blur=.01, scaling=.9, backend='multiscale')

    with h5py.File(args['wdists_path'],       'w') as output_file, \
         h5py.File(args['barys_path'],        'r') as barys_file, \
         h5py.File(args['input_shapes_path'], 'r') as ins_file:
        
        ids_dset = barys_file['input_ids']
        inputs_dset = ins_file['input_shapes']
        n_wdists, n_inputs = ids_dset.shape
        
        out_input_ids_dset = output_file.create_dataset('input_ids', (n_wdists, n_inputs), 
                                                        compression='gzip', shuffle=True, dtype=np.uint32, 
                                                        chunks=(args['chunk_size'],n_inputs))
        wdists_dset = output_file.create_dataset('wdists', [n_wdists], 
                                                 compression='gzip', shuffle=True, dtype=np.float32)

        for batch_first_idx in range(0, n_wdists, args['batch_size']):
            batch_last_idx = batch_first_idx + args['batch_size']

            all_wdists = []
            for i in range(batch_first_idx, batch_last_idx):
                if ((i+1) % (n_wdists//100) == 0):
                    print('{}/{}'.format(i+1,n_wdists))

                pair = ids_dset[i]

                # load the shapes corresponding to the previous ids
                shapes_weights, shapes_coords = [], []
                for shape_id in pair:
                    shape_weights = inputs_dset[shape_id]
                    shape_weights = torch.from_numpy(shape_weights).float().to(device)
                    shape_weights, shape_coords = as_sparse_measure(shape_weights, device)
                    shapes_weights.append(shape_weights)
                    shapes_coords.append(shape_coords)

                # compute the approximation of the wasserstein distance between the 2 shapes
                wdist = sinkhorn_loss(shapes_weights[0], shapes_coords[0], 
                                      shapes_weights[1], shapes_coords[1]).item()
                all_wdists.append(wdist)

            all_wdists = np.array(all_wdists, dtype=np.float32)
            
            out_input_ids_dset[batch_first_idx:batch_last_idx] = ids_dset[batch_first_idx:batch_last_idx]
            wdists_dset[batch_first_idx:batch_last_idx] = all_wdists[:]
    
    deltat = time.time() - t1
    print('Elapsed Time: {:.2f}s for {} wdists'.format(deltat, n_wdists))
    with open('{}_generation_time.txt'.format(dsname), 'w') as f:
        f.write('Elapsed Time: {:.2f}s for {} wdists'.format(deltat, n_wdists))

if __name__ == "__main__":
    main()
