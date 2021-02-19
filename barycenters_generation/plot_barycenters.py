import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.getcwd()))

from io_util import analyse_args


def plot_samples(samples_ids, inputs_path, barys_path):
    dsname = barys_path.split('/')[-1].split('.')[0]
    n_samples = len(samples_ids)
    
    with h5py.File(inputs_path, 'r') as fins, \
         h5py.File(barys_path, 'r') as fbarys:
        ins_dset = fins['input_shapes']
        ins_ids_dset = fbarys['input_ids']
        bweights_dset = fbarys['bweights']
        barys_dset = fbarys['barycenters']
        
        n_dpi_per_img = 4
        img_size = 512
        dpi = img_size // n_dpi_per_img
        nrows = 3 * n_dpi_per_img + 2
        ncols = n_samples * n_dpi_per_img + 2

        fig = plt.figure(figsize=(ncols, nrows), dpi=dpi)
        gs = GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        bbox = dict(boxstyle='round', fc='cyan', alpha=0.3)
        ax = fig.add_subplot(gs[0:n_dpi_per_img,0:2]); ax.axis('off')
        ax.text(.5, .5, '  input 1  ', horizontalalignment='center',
                verticalalignment='center', fontsize=22, bbox=bbox)
        ax = fig.add_subplot(gs[n_dpi_per_img+1:n_dpi_per_img*2+1,0:2]); ax.axis('off')
        ax.text(.5, .5, '  input 2  ', horizontalalignment='center',
                verticalalignment='center', fontsize=22, bbox=bbox)
        ax = fig.add_subplot(gs[n_dpi_per_img*2+2:n_dpi_per_img*3+2,0:2]); ax.axis('off')
        ax.text(.5, .5, 'barycenter', horizontalalignment='center',
                verticalalignment='center', fontsize=22, bbox=bbox)
        
        for i, sample_id in zip(range(2, n_samples*n_dpi_per_img+1, n_dpi_per_img), samples_ids):
            in1_id, in2_id = ins_ids_dset[sample_id]
            in1 = ins_dset[in1_id]
            in2 = ins_dset[in2_id]
            bary = barys_dset[sample_id]
            in1 = in1 / in1.sum()
            in2 = in2 / in2.sum()
            bary = bary / bary.sum()
            weights = bweights_dset[sample_id]
            
            col_figs = np.stack([in1, in2, bary])
            vmin = np.percentile(col_figs[col_figs>0.], 5)
            vmax = np.percentile(col_figs[col_figs>0.], 95)
            
            ax = fig.add_subplot(gs[0:n_dpi_per_img,i:i+n_dpi_per_img])
            ax.axis('off')
            ax.imshow(in1, vmin=vmin, vmax=vmax, cmap='Greys')
            ax.set_title('w1={:.4f}'.format(weights[0]), y=-0.15, fontsize=22)
            
            ax = fig.add_subplot(gs[n_dpi_per_img+1:n_dpi_per_img*2+1,i:i+n_dpi_per_img])
            ax.axis('off')
            ax.imshow(in2, vmin=vmin, vmax=vmax, cmap='Greys')
            ax.set_title('w2={:.4f}'.format(weights[1]), y=-0.15, fontsize=22)
            
            ax = fig.add_subplot(gs[n_dpi_per_img*2+2:n_dpi_per_img*3+2,i:i+n_dpi_per_img])
            ax.axis('off')
            ax.imshow(bary, vmin=vmin, vmax=vmax, cmap='Greys')
        
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
        fig.savefig('{}_{}_samples.png'.format(dsname, n_samples))
        

def plot_grid(nrows, ncols, barys_path):
    dsname = barys_path.split('/')[-1].split('.')[0]
    
    with h5py.File(barys_path, 'r') as f:
        dset = f['barycenters']
        print(dset.dtype)
        print(dset.shape)

        fg, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(ncols,nrows), dpi=dset.shape[-1])
        for i in range(nrows):
            for j in range(ncols):
                fig = dset[i*ncols+j]
                vmin = np.percentile(fig[fig>0.], 5)
                vmax = np.percentile(fig[fig>0.], 95)
                axes[i][j].imshow(fig, vmin=vmin, vmax=vmax, cmap='Greys')
                axes[i][j].set_yticks([])
                axes[i][j].set_xticks([])
        fg.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
        fg.savefig('{}_{}_{}_grid.png'.format(dsname, nrows, ncols))

def main():
    args = analyse_args([
        ['r',       'nrows', lambda x: int(x),  10],
        ['c',       'ncols', lambda x: int(x),  10],
        ['s', 'samples_ids', lambda x: [int(id_fig) for id_fig in x.split(',')], [1,2,3]],
        ['i', 'inputs_path', lambda x: x, os.path.join('..', 'datasets', 
                                                       'random_2d_csg_contours_with_barys', 
                                                       'inputs_v2.h5')],
        ['b',  'barys_path', lambda x: x, os.path.join('..', 'datasets', 
                                                       'random_2d_csg_contours_with_barys', 
                                                       'barycenters_v2.h5')],
        ['m',    'plot_mod', lambda x: x, 'grid']
    ])
    
    if (args['plot_mod'] == 'samples'):
        plot_samples(args['samples_ids'], args['inputs_path'], args['barys_path'])
    elif (args['plot_mod'] == 'grid'):
        plot_grid(args['nrows'], args['ncols'], args['barys_path'])
    
if __name__ == "__main__":
    main()
