import os
import sys
import matplotlib.pyplot as plt
import h5py
import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.getcwd()))
from io_util import analyse_args


def main():
    args = analyse_args([
        ['d', 'dataset_path', lambda x: str(x), os.path.join('.', 'input_shapes.h5')],
        ['r', 'nrows',        lambda x: int(x), 5],
        ['c', 'ncols',        lambda x: int(x), 5]
    ])
    
    dsname, ext = args['dataset_path'].split('/')[-1].split('.')
    if (ext not in ['h5', 'npy']):
        print('Unhandled datatype {}'.format(ext))
        sys.exit(-1)
    
    if (ext == 'h5'):
        with h5py.File(args['dataset_path'], 'r') as f:
            imgs = f['input_shapes'][:args['nrows']*args['ncols']]
        img_size = imgs.shape[-1]
    else:
        imgs = np.load(args['dataset_path'])[:args['nrows']*args['ncols']]
        img_size = int(math.sqrt(imgs.shape[-1])) # we assume square images
        imgs = imgs.reshape((-1,img_size,img_size))
        
    fg, axes = plt.subplots(nrows=args['nrows'], ncols=args['ncols'],
                            figsize=(args['ncols'],args['nrows']), dpi=img_size)
    for i in range(args['nrows']):
        for j in range(args['ncols']):
            fig = imgs[i*args['ncols']+j]
            axes[i][j].imshow(fig, cmap='Greys')
            axes[i][j].set_yticks([])
            axes[i][j].set_xticks([])
    fg.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    fg.savefig('{}_{}_{}.png'.format(dsname, args['nrows'], args['ncols']))
    
if __name__ == "__main__":
    main()
