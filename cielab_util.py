import os
import numpy as np
from skimage import io, color
from scipy import sparse

def get_lab(rgb_img, with_lum=False):
    # rgb -> lab
    lab = color.rgb2lab(rgb_img)
    a = lab[:,:,1].reshape(-1)
    b = lab[:,:,2].reshape(-1)
    
    # chrominances ranges
    range_a = (a.min(),a.max())
    range_b = (b.min(),b.max())
    
    if (with_lum):
        l = lab[:,:,0].reshape(-1)
        return l, a, b, range_a, range_b
    else:
        return a, b, range_a, range_b

def get_displayable_histo(histo):
    eps = 0.1 / (histo.shape[0]**2)
    histo = np.log(histo+eps)
    histo = (histo-histo.min())/(histo.max()-histo.min())
    return histo

def get_chrom_space(range_a, range_b, histo_size, L=None):
    a = np.linspace(*range_a, histo_size)
    b = np.linspace(*range_b, histo_size)
    B, A = np.meshgrid(b,a)
    if L is None:
        L = np.full((histo_size,histo_size),50)
    lab = np.stack((L,A,B),axis=-1)
    chrom_space = color.lab2rgb(lab)
    return chrom_space

def get_and_save_chrom_space(range_a, range_b, histo_size, res_folder):
    chrom_space_fpath = os.path.join(res_folder, 'chrom_space.png')
    print('Computing {}... '.format(chrom_space_fpath), end='')
    chrom_space = get_chrom_space(range_a, range_b, histo_size)
    io.imsave(chrom_space_fpath, (chrom_space*255.).astype(np.uint8))
    print('Done.')
    return chrom_space

def get_colored_chrom_histo(histo, chrom_space):
    histo = get_displayable_histo(histo)
    
    # filter chrominance space following the histogram and put it on
    # a white background
    dupl_histo = np.stack([histo for _ in range(3)], axis=-1)
    colored_histo = 1. -  (1. - chrom_space) * dupl_histo
    
    colored_histo = (colored_histo * 255.).astype(np.uint8)
    return colored_histo

def save_colored_chrom_histo(histo, chrom_space, histo_fpath):
    colored_histo = get_colored_chrom_histo(histo, chrom_space)
    io.imsave(histo_fpath, colored_histo)

def get_chrominances(img_ids, input_folder):
    chrominances = []
    chrominances_ranges = [] # note: all lab values are not convertible
                             # in rgb, thus we need to compute the 
                             # chrominance range of each image in order 
                             # to have a valid chrominance range to use
    for img_id in img_ids:
        rgb_img_fpath = os.path.join(input_folder, 
                                     '{}.png'.format(img_id))
        rgb_img = io.imread(rgb_img_fpath)

        a, b, local_range_a, local_range_b = get_lab(rgb_img)
        chrominances.append([a,b])
        chrominances_ranges.append([local_range_a, local_range_b])
    chrominances_ranges = np.array(chrominances_ranges)
    range_a = (chrominances_ranges[:,0,:].min(),
               chrominances_ranges[:,0,:].max())
    range_b = (chrominances_ranges[:,1,:].min(),
               chrominances_ranges[:,1,:].max())
    return chrominances, chrominances_ranges, range_a, range_b

def get_images(img_ids, input_folder):
    rgb_imgs = []
    for img_id in img_ids:
        rgb_img_fpath = os.path.join(input_folder, 
                                     '{}.png'.format(img_id))
        rgb_img = io.imread(rgb_img_fpath)
        rgb_imgs.append(rgb_img)
    return rgb_imgs

def get_labs(rgb_imgs):
    luminances = []
    chrominances = []
    chrominances_ranges = [] # note: all lab values are not convertible
                             # in rgb, thus we need to compute the 
                             # chrominance range of each image in order 
                             # to have a valid chrominance range to use
    for rgb_img in rgb_imgs:
        l, a, b, local_range_a, local_range_b = get_lab(rgb_img, with_lum=True)
        luminances.append(l)
        chrominances.append([a,b])
        chrominances_ranges.append([local_range_a, local_range_b])
    chrominances_ranges = np.array(chrominances_ranges)
    range_a = (chrominances_ranges[:,0,:].min(),
               chrominances_ranges[:,0,:].max())
    range_b = (chrominances_ranges[:,1,:].min(),
               chrominances_ranges[:,1,:].max())
    return luminances, chrominances, range_a, range_b

def get_chrominance_histo(a, b, range_a, range_b, histo_size, 
                          dtype=np.float32):
    n_pixels = a.shape[0]
    
    # chrominances ranges (should be in [-128,128]) -> [0,1]
    a = (a-range_a[0])/(range_a[1]-range_a[0])
    b = (b-range_b[0])/(range_b[1]-range_b[0])
    
    # float [0,1] -> integer [0,histo_size-1]
    bins_a = np.floor(a*(histo_size-1)).astype(np.int)
    bins_b = np.floor(b*(histo_size-1)).astype(np.int)
    
    # 2D histogram is found via building a sparse matrix
    histo = sparse.csr_matrix((np.ones((n_pixels),np.int),
                               (bins_a,bins_b)),
                              shape=(histo_size,histo_size))
    # note: bins_a = rows indices ; bins_b = columns indices
    histo = (np.array(histo.todense()).astype(dtype) / n_pixels).astype(dtype)
    
    return histo, bins_a, bins_b

def get_chrom_histograms(chrominances, range_a, range_b, 
                         histo_size, dtype=np.float32):
    histograms = [] # chrominance 2D histograms
    all_bins_a = []
    all_bins_b = []
    
    for a, b in chrominances:
        a = a.astype(dtype)
        b = b.astype(dtype)
        
        # get chrominance histogram
        histo, bins_a, bins_b = get_chrominance_histo(a, b, range_a, 
                                                      range_b, histo_size,
                                                      dtype=dtype)
        histograms.append(histo)
        all_bins_a.append(bins_a)
        all_bins_b.append(bins_b)
    return np.stack(histograms), all_bins_a, all_bins_b

def get_luminance_histo(l, histo_size, dtype=np.float32):
    n_pixels = l.shape[0]
    range_l = [0.,100.]
    
    # luminance ranges (should be in [0,100]) -> [0,1]
    l = (l-range_l[0])/(range_l[1]-range_l[0])
    
    # float [0,1] -> integer [0,histo_size-1]
    bins_l = np.floor(l*(histo_size-1)).astype(np.int)
    
    histo = np.histogram(bins_l,bins=range(histo_size+1))[0]
    
    # note: bins_a = rows indices ; bins_b = columns indices
    histo = (histo.astype(dtype) / n_pixels).astype(dtype)
    
    return histo, bins_l

def get_lum_histograms(luminances, histo_size, dtype=np.float32):
    histograms = []
    all_bins_l = []
    
    for l in luminances:
        l = l.astype(dtype)
        
        # get luminance histogram
        histo, bins_l = get_luminance_histo(l, histo_size, dtype=dtype)
        histograms.append(histo)
        all_bins_l.append(bins_l)
    return np.stack(histograms), all_bins_l
    