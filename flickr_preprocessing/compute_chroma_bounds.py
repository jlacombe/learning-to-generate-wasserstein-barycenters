import os
import sys
import numpy as np
from PIL import Image
from skimage import io
from itertools import repeat
from multiprocessing import Pool, cpu_count

# allows imports from parent directories
sys.path.insert(0, os.path.dirname(os.getcwd()))

from cielab_util import get_lab

def filter_and_get_ab_bounds(img_path):
#     rgb_img = io.imread(img_path, plugin='matplotlib')
    
    try:
        rgb_img = np.array(Image.open(img_path))
    
        print('{} -- {}'.format(img_path, rgb_img.shape))
        if rgb_img.shape[-1] == 3: # check image is RGB
            _, _, (min_a, max_a), (min_b, max_b) = get_lab(rgb_img)
            return min_a, max_a, min_b, max_b
        else:
            print('n_channels != 3 => {}'.format(img_path))
            os.remove(img_path)
            return np.inf, -np.inf, np.inf, -np.inf
    except Exception:
        print('Corrupted => {}'.format(img_path))
        os.remove(img_path)
        return np.inf, -np.inf, np.inf, -np.inf

def main():
    flickr_folder_path = os.path.join('..', 'datasets', 'flickr')
    imgs_folders_path = os.path.join(flickr_folder_path, 'imgs')
    ab_bounds = []
    imgs_paths = [
        os.path.join(imgs_folders_path, fname, img_name) 
        for fname in os.listdir(imgs_folders_path)
        for img_name in os.listdir(os.path.join(imgs_folders_path, fname))
    ]
    print(len(imgs_paths))

    with Pool(cpu_count()) as p:
        ab_bounds = p.starmap(filter_and_get_ab_bounds, zip(imgs_paths))
    ab_bounds = np.stack(ab_bounds)
    np.save('ab_bounds.npy', ab_bounds)
    
    print(ab_bounds[:,1])
    min_a = ab_bounds[:,0].min()
    max_a = ab_bounds[:,1].max()
    min_b = ab_bounds[:,2].min()
    max_b = ab_bounds[:,3].max()
    print('a_range=[{},{}]'.format(min_a, max_a))
    print('b_range=[{},{}]'.format(min_b, max_b))
    
    chroma_bounds_fpath = os.path.join(flickr_folder_path, 'chroma_bounds.txt')
    with open(chroma_bounds_fpath, 'w') as f:
        f.write('{},{},{},{}'.format(min_a, max_a, min_b, max_b))

if __name__ == "__main__":
    main()
