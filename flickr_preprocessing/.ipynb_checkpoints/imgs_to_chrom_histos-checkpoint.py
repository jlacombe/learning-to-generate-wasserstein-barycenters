import os
import sys
import numpy as np
import h5py
from skimage import io, color
from itertools import repeat
from multiprocessing import Pool, cpu_count

# allows imports from parent directories
sys.path.insert(0, os.path.dirname(os.getcwd()))

from cielab_util import get_lab, get_chrominance_histo, get_displayable_histo,\
                        get_and_save_chrom_space, save_colored_chrom_histo, \
                        get_chrominances, get_chrom_histograms

def rgb_to_chrom_histo(img_path, imgs_folders_path, 
                       range_a, range_b, histo_size):
    img_path = os.path.join(imgs_folders_path, img_path)
    # rgb -> lab
    rgb_img = io.imread(img_path, plugin='matplotlib')
    lab = color.rgb2lab(rgb_img)
    a = lab[:,:,1].reshape(-1)
    b = lab[:,:,2].reshape(-1)

    # compute the chrominance 2D histogram
    histo, _, _ = get_chrominance_histo(a, b, range_a, range_b, histo_size)
    
    return histo
    

def main():
    flickr_folder_path = os.path.join('..', 'datasets', 'flickr')
    imgs_folders_path = os.path.join(flickr_folder_path, 'imgs')
    chroma_bounds_fpath = os.path.join(flickr_folder_path, 'chroma_bounds.txt')
    n_histos = 100000
    histo_size = 512
    batch_size = 200
    h5_path = os.path.join(flickr_folder_path, 
                           'input_chrom_histos_{}.h5'.format(n_histos))

    imgs_paths = [
        os.path.join(category, img_fname)
        for category in os.listdir(imgs_folders_path)
        for img_fname in os.listdir(os.path.join(imgs_folders_path, category))
    ]
    
    np.random.shuffle(imgs_paths)
    
    imgs_paths = imgs_paths[:n_histos]
    
    if n_histos > len(imgs_paths):
        print('''WARNING: there are only {} images, not enough to make {}
                 histograms'''.format(len(imgs_paths), n_histos))
        n_histos = len(imgs_paths)
    
    print('max_len={}'.format(max([len(img_path) 
                                   for img_path in imgs_paths])))
    
    # retrieve (a,b) chrominance ranges
    with open(chroma_bounds_fpath, 'r') as f:
        ab_bounds = [float(x) for x in f.readlines()[0].split(',')]
        range_a = ab_bounds[0], ab_bounds[1]
        range_b = ab_bounds[2], ab_bounds[3]
    print('range_a={}'.format(range_a))
    print('range_b={}'.format(range_b))
    
    # compute the chrominance space in which lie the computed lab
    # representations
    chrom_space = get_and_save_chrom_space(range_a, range_b, histo_size,
                                  flickr_folder_path)
    
    print('Creating h5 dataset...')
    with h5py.File(h5_path, 'w') as f:
        histos_dset = f.create_dataset('input_shapes', 
                                       (n_histos,histo_size,histo_size), 
                                       compression='gzip', shuffle=True, 
                                       dtype=np.double, 
                                       chunks=(1,histo_size,histo_size))
        imgs_paths_dset = f.create_dataset('imgs_paths', (n_histos,), 
                                           compression='gzip', shuffle=True, 
                                           dtype='S128')
        
        for i in range(0, n_histos, batch_size):
            print('{}-{}/{}'.format(i,i+batch_size,n_histos))
            
            histos = []
            with Pool(cpu_count()) as p:
                histos = p.starmap(rgb_to_chrom_histo, 
                                   zip(imgs_paths[i:i+batch_size], 
                                       repeat(imgs_folders_path),
                                       repeat(range_a), 
                                       repeat(range_b), 
                                       repeat(histo_size)))
            histos_dset[i:i+batch_size] = np.stack(histos)
            
            ascii_imgs_paths = [img_path.encode('ascii', 'ignore') 
                                for img_path in imgs_paths[i:i+batch_size]]
            imgs_paths_dset[i:i+batch_size] = ascii_imgs_paths

            del histos

if __name__ == "__main__":
    main()
