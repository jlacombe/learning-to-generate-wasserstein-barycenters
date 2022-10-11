import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def interpolate_between_2_shapes(bary_evo, interval, cmap='Greys', vmin=None, vmax=None):
    if vmin is None:
        vmin = np.percentile(bary_evo[bary_evo>0.],  5)
    if vmax is None:
        vmax = np.percentile(bary_evo[bary_evo>0.], 95)
    img_side = bary_evo.shape[-1]
    frames = len(bary_evo)
    
    fig, ax = plt.subplots(figsize=(1,1), dpi=img_side)
    ax.axis('off')
    ax.imshow(bary_evo[0], vmin=vmin, vmax=vmax, cmap=cmap)

    def init():
        return (ax.imshow(bary_evo[0], vmin=vmin, vmax=vmax, cmap=cmap),)

    def animate(j):
        return (ax.imshow(bary_evo[j], vmin=vmin, vmax=vmax, cmap=cmap),)

    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, 
                                   interval=interval, blit=True)
    return anim

def interpolate_3x3_shapes_grid(all_bary_evo, interval):
    img_side = all_bary_evo[0][0].shape[-1]
    frames = len(all_bary_evo[0][0])
    
    fig, ax = plt.subplots(figsize=(3,3), dpi=img_side, nrows=3, ncols=3)

    ims = []
    for i in range(9):
        bary_evo, vmin, vmax = all_bary_evo[i]
        im = ax[i//3][i%3].imshow(bary_evo[0], vmin=vmin, vmax=vmax, cmap='Greys')
        ax[i//3][i%3].axis('off')
        ims.append(im)

    def init():
        for i in range(9):
            bary_evo, vmin, vmax = all_bary_evo[i]
            im = ax[i//3][i%3].imshow(bary_evo[0], vmin=vmin, vmax=vmax, cmap='Greys')
            ims.append(im)
        return ims

    def animate(j):
        print(j)
        for i in range(9):
            bary_evo, vmin, vmax = all_bary_evo[i]
            im = ax[i//3][i%3].imshow(bary_evo[j], vmin=vmin, vmax=vmax, cmap='Greys')
            ims.append(im)
        return ims

    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, 
                                   interval=interval, blit=True)
    return anim


def build_anim(samples, n_frames_per_bary=4): # samples = list of tuples (in1, in2, barys, barys_thresh)
    n_samples = len(samples)
    n_barys = len(samples[0][1]) # we assume that we have the same number of barycenters for each sample
    total_frames_bary = n_frames_per_bary * n_barys
    n_frames = total_frames_bary
    
    cmap = 'Greys'

    txt_props = dict(boxstyle='round', facecolor='aquamarine')

    fig, ax = plt.subplots(nrows=n_samples, ncols=3)
    fig.set_size_inches(14, 14)
    
    all_bary_fig = []
    for i in range(n_samples):
        (in1, in2), evo_bary  = samples[i]
        vmin = 0.
        vmax = max(in1.max(), in2.max())
        ax[i][0].imshow(in1, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i][0].axis('off')
        ax[i][1].imshow(in2, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i][1].axis('off')
        bary_fig = ax[i][2].imshow(in1, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i][2].axis('off')
        all_bary_fig.append(bary_fig)
        
    n_anims = len(all_bary_fig)

    def init():
        for j in range(n_samples):
            bary_fig = all_bary_fig[j]
            bary_fig.set_data(samples[j][1][0])
        return all_bary_fig

    def animate(i):
        for j in range(n_samples):
            bary_fig = all_bary_fig[j]
            (in1, in2), barys = samples[j]
            bary_idx = math.floor(i / n_frames_per_bary)
            bary_fig.set_data(barys[bary_idx])
        txt_str = 'grad_iter={}'.format(bary_idx+1)
        ax[0][2].text(1.05, 1.5, txt_str, transform=ax[0][2].transAxes, 
                          fontsize=14, verticalalignment='top', bbox=txt_props)
        return all_bary_fig

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=20, 
                                   blit=True)
    return anim

