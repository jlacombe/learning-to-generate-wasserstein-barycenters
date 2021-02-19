import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from ot.bregman import convolutional_barycenter2d
from geomloss import SamplesLoss
from skimage import io, color
from pykeops.torch import LazyTensor, Vi, Vj
from PIL import Image
from scipy.interpolate import interp1d
from cv2.ximgproc import guidedFilter

# allows imports from parent directories
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from bary_gen_util import grid, as_sparse_measure, bin_barycenter, \
                          gen_bary_from_targets
from models import BarycentricUNet
from model_util import load_flags, load_model_on_gpu
from cielab_util import get_lab, get_chrominance_histo, get_displayable_histo,\
                        get_and_save_chrom_space, save_colored_chrom_histo, \
                        get_images, get_labs, get_chrom_histograms, \
                        get_chrom_space, get_colored_chrom_histo, \
                        get_lum_histograms
from barycentric_polygon_comparison import get_barycentric_triangle, \
                                           get_barycentric_pentagon, \
                                           build_barycentric_polygon_figures

def compute_sink_div_targets(histo_size, device, histograms, blur, scaling):
    n_dims = len(histograms.shape[1:])
    Loss = SamplesLoss('sinkhorn', blur=blur, scaling=scaling, 
                       backend='multiscale')
    print('Computing targets', end='')
    histograms = torch.tensor(histograms).to(device)
    if n_dims == 2:
        bary_coords = grid(histo_size, device, 
                           dtype=histograms.dtype).view(-1,2)
        bary_weights = (torch.ones(histo_size*histo_size) / 
                        (histo_size*histo_size)).type_as(bary_coords)
    elif n_dims == 1:
        bary_coords = torch.linspace(0.,histograms.shape[1], 
                                     histograms.shape[1]).type(histograms.dtype).reshape(-1,1).to(device)
        bary_weights = (torch.ones(histo_size) / histo_size).type_as(bary_coords).to(device)
    bary_coords.requires_grad = True
    
    targets = []
    for histo in histograms:
        if n_dims == 2:
            histo_weights, histo_coords = as_sparse_measure(histo, device)
        elif n_dims == 1:
            histo_weights = histo.to(device)
            histo_coords = torch.linspace(0., histograms.shape[1], 
                                          histograms.shape[1])
            histo_coords = histo_coords.type(histograms.dtype).reshape(-1,1)
            histo_coords = histo_coords.to(device)
        loss_val = Loss(bary_weights, bary_coords, 
                        histo_weights, histo_coords)
        [gradient] = torch.autograd.grad(loss_val, [bary_coords])
        target = bary_coords - gradient / bary_weights.view(-1,1)
        target = target.detach()
        
        targets.append(target)
        print('.', end='')
    print(' Done. ')
    return targets

def compute_sink_div_bary(targets, weights, histo_size):
    n_dims = 1 if targets[0].shape[1] == 1 else 2
    bary = targets[0] * weights[0]
    for i in range(1,len(targets)):
        bary += targets[i] * weights[i]
    if n_dims == 2:
        bary = bin_barycenter(bary, histo_size)
    bary = bary.cpu().numpy()
    bary = bary / bary.sum()
    return bary
    
def compute_sink_div_barys(targets, n_barys, histo_size):
    barys = []
    w1_range = np.linspace(1., 0., n_barys)
    for w1 in w1_range:
        bary = compute_sink_div_bary(targets, [w1, (1.-w1)], histo_size)
        barys.append(bary)
    return np.stack(barys), w1_range

def get_pbary_bunet(model):
    def get_pbary_inner(ins, bweights):
        bweights = bweights.reshape(1,ins.shape[1],1,1)
        bweights = torch.tensor(bweights,dtype=ins.dtype)
        with torch.no_grad():
            pbary = model(ins.cuda(), bweights.cuda())[0][0].detach().cpu()
        return pbary.numpy()
    return get_pbary_inner

def get_pred_bary_function(model_id, model_class, dtype=np.float32):
    if model_class == 'BarycentricUNet':
        model_results_path = os.path.join('..', 'training_model', 'results', model_id)
        flags_path = os.path.join(model_results_path, 'flags.pkl')
        model_path = os.path.join(model_results_path, 'model.pth')
        FLAGS = load_flags(flags_path)
        model_args = [FLAGS[param_name] 
                      for param_name in FLAGS['model_params_names']]
        model = load_model_on_gpu(model_path, BarycentricUNet,
                                  model_args).eval()
        if dtype == np.float64:
            model = model.double()
        get_pbary = get_pbary_bunet(model)
    return get_pbary

def compute_model_barys(histograms, n_barys, histo_size, 
                        get_pbary, model_id, device):
    histograms = torch.tensor(histograms).unsqueeze(0)
    histograms = histograms.to(device)
    
    barys = []
    w1_range = np.linspace(1., 0., n_barys)
    for w1 in w1_range:
        bweights = np.array([w1, 1-w1])
        bary = get_pbary(histograms, bweights)
        barys.append(bary)

    return np.stack(barys), w1_range

def save_barycenters(barys, ws, bary_folder_path, chrom_space,
                     vmin=None, vmax=None):
    if not os.path.isdir(bary_folder_path):
        os.mkdir(bary_folder_path)
    
    # apply a threshold in order to filter out extreme values
    if vmin is None:
        vmin = np.percentile(barys, 1)
    if vmax is None:
        vmax = np.percentile(barys, 99)
    barys = np.clip(barys, vmin, vmax)

    for bary, w1 in zip(barys, ws):
        bary_file_path = os.path.join(bary_folder_path, 
                                      '{:.2f}.png'.format(w1))
        save_colored_chrom_histo(bary, chrom_space, bary_file_path)
        
def chrom_transfer_func(histo, bary, histo_coords, bary_coords, F, G, 
                        bins_chr_source, blur, device, histo_size):
    p = 2
    eps = blur**p
    bins_chr_source = torch.tensor(bins_chr_source).to(device)
    
    X_i = LazyTensor(histo_coords[None,:]) # M x 2 => 1 x M x 2
    Y_j = LazyTensor(bary_coords[:,None])  # M x 2 => M x 1 x 2
    C_ij = (1./p) * ((X_i-Y_j)**p).sum(-1) # M x M cost matrix
    
    F_i = LazyTensor(F[:,None,None]) # M => 1 x M : F is added to each line
    G_j = LazyTensor(G[None,:,None]) # M => M x 1 : G is added to each col
    
    I = LazyTensor((histo_size*bary_coords)[None,:,:])
    W_i = LazyTensor(histo[:,None,None]) # M => 1 x M : Wi is .* with each line
    W_j = LazyTensor(bary[None,:,None])  # M => 1 x M : Wj is .* with each col
    
    ot_plan = (((F_i + G_j - C_ij) / eps).exp() * (W_i * W_j))
    
    chrom_space_target = ot_plan * I
    
    chrom_space_target = chrom_space_target.sum(1) / histo.reshape(-1,1)
    
    bins_chr_target = chrom_space_target[bins_chr_source]
    
    bins_chr_target = bins_chr_target.clamp(0., histo_size-1.)
    bins_chr_target = bins_chr_target.cpu().numpy()
    
    return bins_chr_target

def apply_chrominance_transfer(bins_chr_source, range_a, range_b, histo,
                               bary, OT_solver, blur, device, histo_size):
    histo_coords = grid(histo_size, device, dtype=histo.dtype)
    bary_coords  = grid(histo_size, device, dtype=bary.dtype)
    
    F_i, G_j = OT_solver(histo, histo_coords, bary, bary_coords)
    
    bins_chr_target = chrom_transfer_func(histo, bary, histo_coords, 
                                          bary_coords, F_i, G_j, 
                                          bins_chr_source, blur, device,
                                          histo_size)
    
    bins_b_target = bins_chr_target[:,0]
    bins_a_target = bins_chr_target[:,1]
    
    a_target = bins_a_target / histo_size
    b_target = bins_b_target / histo_size
    
    # [0,1] => orig_chrom_space_range
    a_target = a_target * (range_a[1]-range_a[0]) + range_a[0]
    b_target = b_target * (range_b[1]-range_b[0]) + range_b[0]

    return a_target, b_target

def interpolate_luminances(luminances, weights, idx_lum_ref):
    lum_ref = luminances[idx_lum_ref]
    sigma_ref = np.argsort(lum_ref)
    lum_interp = torch.zeros(lum_ref.shape)
    
    lum_indices = list(range(len(luminances)))
    lum_indices.remove(idx_lum_ref)
    for i in lum_indices:
        lum_i = luminances[i]
        w_i = weights[i]
        sigma_i = np.argsort(lum_i)
        lum_i_sorted = lum_i[sigma_i]

        interp_lum_i_indices = np.linspace(0, len(lum_i)-1, len(lum_ref))
        sigma_inv_lum_i = interp1d(np.arange(len(lum_i)), lum_i_sorted) \
                                 (interp_lum_i_indices)
        sigma = lum_ref.copy()
        sigma[sigma_ref] = sigma_inv_lum_i
        
        lum_interp += w_i * sigma
    
    return lum_interp
    

def get_color_transfered_img(bins_chr_source, range_a, range_b, 
                             chrom_histo, chrom_bary, OT_solver, blur, 
                             device, histo_size, luminances, bweights,
                             img_shape, idx_lum_ref):
    a_target, b_target = apply_chrominance_transfer(
        bins_chr_source, range_a, range_b, chrom_histo,
        chrom_bary, OT_solver, blur, device, histo_size)

    lum = interpolate_luminances(luminances, bweights, idx_lum_ref)

    lab_target = np.stack((lum.reshape(img_shape[:-1]), 
                           a_target.reshape(img_shape[:-1]), 
                           b_target.reshape(img_shape[:-1])), axis=2)

    rgb_target = color.lab2rgb(lab_target)
    rgb_target = (rgb_target * 255.).astype(np.uint8)
    return rgb_target

def get_disp_chrom_histo(chrom_space):
    def disp_chrom_histo(ax, histo, vmin, vmax):
        histo = histo.clip(vmin, vmax)
        colored_histo = get_colored_chrom_histo(histo, chrom_space)
        ax.imshow(colored_histo)
    return disp_chrom_histo

def disp_color_img(ax, img, vmin, vmax):
    ax.imshow(img)

def apply_iterative_guided_filtering(orig_img, modif_img, niters=10, r=4, eps=0.02**2):
    orig_img = np.float32(orig_img) / 255.
    modif_img = np.float32(modif_img) / 255.
    
    ot_map = modif_img - orig_img
    
    filtered_ot_map = np.zeros(orig_img.shape)
    for i in range(niters):
        filtered_ot_map[:,:,0] = guidedFilter(orig_img, ot_map[:,:,0], r, eps)
        filtered_ot_map[:,:,1] = guidedFilter(orig_img, ot_map[:,:,1], r, eps)
        filtered_ot_map[:,:,2] = guidedFilter(orig_img, ot_map[:,:,2], r, eps)
        ot_map = np.float32(filtered_ot_map.copy())
    
    enhanced_img = ot_map + orig_img
    
    enhanced_img = (enhanced_img * 255.).clip(0.,255.).astype(np.uint8)
    
    return enhanced_img
    
def get_ptrans_img_func(bins_chr_source, range_a, range_b, chrom_histo,
                        OT_solver, blur, device, histo_size, luminances,
                        img_shape, get_bary, idx_lum_ref, orig_rgb_img,
                        with_filtering=True):
    def get_ptrans_img_inner(ins, ws):
        print('p - {}'.format(ws))
        chrom_bary = get_bary(ins, ws)
        chrom_bary = torch.tensor(chrom_bary)
        chrom_bary = chrom_bary.reshape(-1).to(device)
        trans_img = get_color_transfered_img(
            bins_chr_source, range_a, range_b, chrom_histo, 
            chrom_bary, OT_solver, blur, device, histo_size,
            luminances, ws, img_shape, idx_lum_ref)
        if (with_filtering):
            trans_img = apply_iterative_guided_filtering(orig_rgb_img, 
                                                         trans_img)
        return trans_img
    return get_ptrans_img_inner

def get_rtrans_img_func(bins_chr_source, range_a, range_b, chrom_histo,
                        OT_solver, blur, device, histo_size, luminances,
                        img_shape, get_bary, idx_lum_ref, orig_rgb_img,
                        with_filtering=True):
    def get_rtrans_img_inner(targets, ws, ins_size):
        print('r - {}'.format(ws))
        chrom_bary = get_bary(targets, ws, ins_size)
        chrom_bary = torch.tensor(chrom_bary)
        chrom_bary = chrom_bary.reshape(-1).to(device)
        trans_img = get_color_transfered_img(
            bins_chr_source, range_a, range_b, chrom_histo, 
            chrom_bary, OT_solver, blur, device, histo_size,
            luminances, ws, img_shape, idx_lum_ref)
        if (with_filtering):
            trans_img = apply_iterative_guided_filtering(orig_rgb_img, 
                                                         trans_img)
        return trans_img
    return get_rtrans_img_inner


def main():
    histo_size = 512
    input_folder = 'input_rgb_imgs'

    # when len(img_ids) == 4 or 6, we build a triangle (resp. a pentagon) in 
    # which each vertex corresponds to the chrominance/luminance style of an 
    # image (the 3 - or 5 - first images) and in which we apply color transfer
    # on another img (the 4th image / resp the 6th)
#     img_ids = ['clockmontague-1', 'clockmontague-2', 'clockmontague-3']
#     img_ids = ['red-5', 'green-2', 'street-11']
#     img_ids = ['baby1', 'baby2', 'baby3', 'baby4']
#     img_ids = ['cat1', 'cat2', 'cat3', 'cat4']
#     img_ids = ['landscape10', 'landscape13', 'landscape12', 'landscape17']
#     img_ids = ['landscape10', 'landscape13', 'landscape12', 'landscape14']
#     img_ids = ['landscape10', 'landscape13', 'landscape12', 'landscape1']
#     img_ids = ['landscape10', 'landscape13', 'landscape12', 'landscape8']
#     img_ids = ['cat1_uncropped', 'cat2', 'cat3', 'cat4']
#     img_ids = ['red-5', 'green-2', 'colors9', 'red-5']
    
    img_ids = ['landscape10', 'landscape13', 'landscape12', 
               'landscape8', 'landscape5', 'landscape17']
    
    n_iters = 2 # number of iterations used to build polygon figure. eg for triangles:
                # 0 iter => isobary
                # 1 iter => triangle around isobary
                # 2 iter => triangle around triangle around isobary
                # etc ...
    max_polygon_fig_res = 4096 # maximum resolution size for a side of the 
                                # barycentric polygon figure
    with_filtering = True  # determines whether iterative guided filtering
                           # is used to enhance color transfer results

    base_folder = 'chrominance_histograms'
    res_folder = os.path.join(base_folder, '_'.join(img_ids))
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    if not os.path.isdir(res_folder):
        os.mkdir(res_folder)
        
    n_barys = 11 # number of barycenters to compute
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = np.float32 # np.float32, np.float64
    
    # Sinkhorn divergence parameters
    sink_div_blur = 1e-2
    sink_div_scaling = 0.9
    
    # NN models parameters
    model_ids = [
'bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_dsv2',
# 'bunet_skipco_100000_31epoch_SGDR_nesterov_IN_0.0005_0.99_8_2_flickr'
    ]
    model_classes = [
        'BarycentricUNet',
#         'BarycentricUNet',
    ]
    
    # compute lab representations and (a,b) chrominance ranges
    rgb_imgs = get_images(img_ids, input_folder)
    
#     res=apply_iterative_guided_filtering(rgb_imgs[-1], rgb_imgs[-1], niters=10, r=4, eps=0.02**2)
    
#     import sys;sys.exit(-1)
    
    imgs_shapes = [img.shape for img in rgb_imgs]
    labs = get_labs(rgb_imgs)
    luminances, chrominances, range_a, range_b = labs
    
    # compute the chrominance space in which lie the computed lab
    # representations
    chrom_space = get_and_save_chrom_space(range_a, range_b, histo_size, 
                                           res_folder)
    
    # compute the chrominance 2D histograms and display them
    chrom_histos, all_bins_a, all_bins_b = get_chrom_histograms(
        chrominances, range_a, range_b, histo_size, dtype=dtype)
    vmin = np.percentile(chrom_histos, 1)
    vmax = np.percentile(chrom_histos, 99)
    
    # compute 1D luminance histograms
    lum_histos, all_bins_l = get_lum_histograms(luminances, histo_size,
                                                dtype=dtype)
    
    # save 2D chrominance histograms
    for histo, img_id in zip(chrom_histos, img_ids):
        histo_fname = '{}.png'.format(img_id)
        colored_histo_fpath = os.path.join(res_folder, histo_fname)
        print('Computing {}... '.format(colored_histo_fpath), end='')

        # apply a threshold in order to filter out extreme values
        histo = histo.clip(vmin, vmax)
        
        # save it as a colored histogram displayed in chrominance space
        # on a white background
        save_colored_chrom_histo(histo, chrom_space, colored_histo_fpath)
        
        print('Done.')
    
    # compute barycenters using Sinkhorn divergence (geomloss)
    # --------------------------------------------------------
    # compute geomloss targets between each histogram 
    # and the uniform distribution (those can then be 
    # directly combined to form barycenters)
    
    chrom_targets = compute_sink_div_targets(histo_size, device, chrom_histos,
                                             sink_div_blur, sink_div_scaling)
    
    lum_targets = compute_sink_div_targets(histo_size, device, lum_histos,
                                              sink_div_blur, sink_div_scaling)
    
    # compute barycenters using a neural network model
    # --------------------------------------------------------
    if (len(img_ids) == 4 or len(img_ids) == 6):
        chrom_histos = torch.tensor(chrom_histos).unsqueeze(0).to(device)
    
    for model_id, model_class in zip(model_ids, model_classes):
        suffix = '_filtered' if with_filtering else '_unfiltered'
        bary_folder_name = '{}_{}'.format(model_id, suffix)
        
        bary_folder_path = os.path.join(res_folder, bary_folder_name)
        print('Computing barycenters {}... '.format(bary_folder_path))
        if not os.path.isdir(bary_folder_path):
            os.mkdir(bary_folder_path)

        # load the model
        get_pbary = get_pred_bary_function(model_id, model_class, dtype)

        
        if (len(img_ids) >= 4):
            if (len(img_ids) == 4):
                polygon, barycentric_polygon, n_cells = get_barycentric_triangle( 
                                                             n_iters, show=False)
                i = 3
            elif (len(img_ids) == 6):
                polygon, barycentric_polygon, n_cells = get_barycentric_pentagon( 
                                                             n_iters, show=False)
                i = 5
            else:
                print('Error: usecase with {} images is not handled. '.format(len(img_ids)))
                sys.exit(-1)
            
            max_res_per_img = max_polygon_fig_res // (n_cells//2)
            
            print('Computing chrominance histograms barycentric triangle...')
            disp_chrom_histo = get_disp_chrom_histo(chrom_space)
            build_barycentric_polygon_figures(
                chrom_histos[0:1,:-1], chrom_targets[:-1], get_pbary, polygon,
                barycentric_polygon, n_cells, n_iters, 
                interpol_id='chrom_histo', results_path=bary_folder_path,
                disp_bary_func=disp_chrom_histo, max_res_per_img=max_res_per_img)
            
            n_imgs = len(imgs_shapes)
            histo_size = chrom_histos.shape[2]
            blur = 5e-2
            scaling = 0.99
            OT_solver = SamplesLoss('sinkhorn', p=2, blur=blur, 
                                    scaling=scaling, debias=False,
                                    potentials=True)
            
            print('Computing transfer color for image {}...'.format(
                                                            img_ids[i]))

            a_src, b_src = chrominances[i]
            img_shape = imgs_shapes[i]
            bins_a = all_bins_a[i]
            bins_b = all_bins_b[i]
            bins_chr_source = bins_a * histo_size + bins_b

            chrom_histo = chrom_histos[0,i].reshape(-1)
            chrom_histo = chrom_histo.to(device)

            orig_rgb_img = rgb_imgs[-1]
            
            get_ptrans_img = get_ptrans_img_func(
                                bins_chr_source, range_a, range_b, 
                                chrom_histo, OT_solver, blur, device, 
                                histo_size, luminances, img_shape,
                                get_pbary, i, orig_rgb_img, 
                                with_filtering=with_filtering)
            get_rtrans_img = get_rtrans_img_func(
                                bins_chr_source, range_a, range_b,
                                chrom_histo, OT_solver, blur, device,
                                histo_size, luminances, img_shape,
                                compute_sink_div_bary, i, orig_rgb_img, 
                                with_filtering=with_filtering)

            print('Computing color transfer barycentric polygon...')
            interpol_id = 'color_transfered_imgs_{}'.format(img_ids[i])
            build_barycentric_polygon_figures(
                chrom_histos[0:1,:-1], chrom_targets[:-1], get_ptrans_img, 
                polygon, barycentric_polygon, n_cells, n_iters, 
                interpol_id=interpol_id, results_path=bary_folder_path,
                disp_bary_func=disp_color_img,
                get_rbary=get_rtrans_img, max_res_per_img=max_res_per_img)
            
        print('Done.')

    
if __name__ == "__main__":
    main()

