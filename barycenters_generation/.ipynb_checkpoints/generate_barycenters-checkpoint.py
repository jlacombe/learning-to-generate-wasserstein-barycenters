import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.getcwd()))

from io_util import analyse_args
from bary_gen_util import *


def get_randn_weights(n_inputs, mu=.5, sigma=.32):
    distrib = []
    for j in range(n_inputs):
        y = np.random.normal(loc=mu, scale=sigma)
        while (y < 0. or y > 1.):
            y = np.random.normal(loc=mu,scale=sigma)
        distrib.append(y)
    distrib = np.array(distrib)
    distrib /= distrib.sum()
    return distrib

def gen_barycenters_h5(targets_path, barys_path, n_inputs, buffer_size,
                       n_barys, chunk_size, img_size=512):
    with h5py.File(targets_path, 'r') as ft, \
         h5py.File(barys_path, 'w') as fb:
        targets_dset = ft['targets']
        
        barys_dset = fb.create_dataset('barycenters', (n_barys, img_size, img_size), 
                                       compression='gzip', shuffle=True, dtype=np.uint8, 
                                       chunks=(chunk_size,img_size,img_size))
        input_ids_dset = fb.create_dataset('input_ids', (n_barys, n_inputs), 
                                           compression='gzip', shuffle=True, dtype=np.uint32, 
                                           chunks=(chunk_size,n_inputs))
        bweights_dset = fb.create_dataset('bweights', (n_barys, n_inputs), 
                                          compression='gzip', shuffle=True, dtype=np.float32, 
                                          chunks=(chunk_size,n_inputs))

        for i in range(0, n_barys, buffer_size):
            print('{}-{}/{}'.format(i, i+buffer_size, n_barys))
            all_barys = []
            all_shapes_ids = []
            all_bweights = []
            
            for _ in range(buffer_size):
                target_ids = gen_random_tuple(0, targets_dset.shape[0]-1, n_inputs)
                targets = [targets_dset[target_id]
                           for target_id in target_ids]
                
                bweights = get_randn_weights(n_inputs)
                
                bary = bweights[0] * targets[0]
                for k in range(1, len(targets)):
                    bary += bweights[k] * targets[k]
                bin_bary = bin_barycenter_numpy(bary, img_size).astype(np.uint8)
                
                all_barys.append(bin_bary)
                all_shapes_ids.append(list(target_ids))
                all_bweights.append(bweights)
            
            all_barys = np.stack(all_barys)
            barys_dset[i:i+buffer_size] = all_barys
            
            all_shapes_ids = np.array(all_shapes_ids, dtype=np.uint32)
            input_ids_dset[i:i+buffer_size] = all_shapes_ids
            
            all_bweights = np.array(all_bweights, dtype=np.float32)
            bweights_dset[i:i+buffer_size] = all_bweights

            
def main():
    t1 = time.time()
    args = analyse_args([
        ['n',          'img_size', lambda x: int(x), 512],
        ['s',          'n_inputs', lambda x: int(x), 2],
        ['i', 'input_shapes_path', lambda x: x, os.path.join('..', 'datasets', 'input_contours.h5')],
        ['t',      'targets_path', lambda x: x, os.path.join('..', 'datasets', 'targets_contours.h5')],
        ['r',        'barys_path', lambda x: x, os.path.join('..', 'datasets', 'barycenters_contours.h5')],
        ['m',           'n_barys', lambda x: int(x), 1000],
        ['b',        'batch_size', lambda x: int(x), 100],
        ['c',        'chunk_size', lambda x: int(x), 1]
    ])
    dsname = args['barys_path'].split('/')[-1].split('.')[0]
    
    all_distribs = np.array([])
    for i in range(args['n_barys']):
        distrib = get_randn_weights(args['n_inputs'])
        all_distribs = np.concatenate((all_distribs, distrib))
    all_distribs = np.round(np.array(all_distribs), 3)

    unique, counts = np.unique(all_distribs, return_counts=True)
    plt.figure()
    plt.plot(unique, counts, 'o')
    plt.savefig('{}_bweights_distrib.png'.format(dsname))
    
    gen_barycenters_h5(args['targets_path'], args['barys_path'], args['n_inputs'], 
                       args['batch_size'], args['n_barys'], args['chunk_size'], args['img_size'])
        
    deltat = time.time() - t1
    print('Elapsed Time: {:.2f}s for {} barycenters'.format(deltat, args['n_barys']))
    with open('{}_generation_time.txt'.format(dsname), 'w') as f:
        f.write('Elapsed Time: {:.2f}s for {} barycenters'.format(deltat, args['n_barys']))
        
if __name__ == "__main__":
    main()
