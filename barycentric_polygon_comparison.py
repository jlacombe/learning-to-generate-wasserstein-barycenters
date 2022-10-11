import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
import cv2
from matplotlib.gridspec import GridSpec

# allows imports from parent directory
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

from bary_gen_util import *

def build_polygon(poly_coords, slopes, n_iters, init_side_len, 
                  side_step, top_vertex_step, recenter_step):
    side_len = init_side_len+1
    for i in range(1,n_iters+1):
        # recenter
        for j in range(len(poly_coords)):
            poly_coords[j][0] += recenter_step[0]
            poly_coords[j][1] += recenter_step[1]

        # add new coords
        x, y = poly_coords[-1]
        x = x + top_vertex_step
        vertices = []
        for slope in slopes:
            vertices.append([x,y])
            for j in range(side_len):
                x += slope[0]
                y += slope[1]
                poly_coords.append([x,y])

        side_len += side_step
    poly_coords = np.array(poly_coords)
    vertices = np.array(vertices)
    return poly_coords, vertices.tolist()

def get_polygon_matrix(img_side, n_iters, poly_coords, init_val):
    polygon = np.zeros((img_side, img_side))
    polygon[poly_coords[:,0],poly_coords[:,1]] = init_val
    return polygon

def get_adjacency_list(polygon, vertices, img_side):
    adj_list = {}
    for i in range(img_side):
        adj_list[i] = {}
        for j in range(img_side):
            if (polygon[i,j] != 0.):
                adj_list[i][j] = []
                possible_neighbours = [
                    (i-2,j-2),(i-2,j-1),(i-2,  j),(i-2,j+1),(i-2,j+2),

                    (  i,j-2),                              (  i,j+2),

                    (i+2,j-2),(i+2,j-1),(i+2,  j),(i+2,j+1),(i+2,j+2)]
                possible_neighbours = [(x,y) for (x,y) in possible_neighbours
                                             if x >= 0 and x < img_side
                                            and y >= 0 and y < img_side]
                for k1,k2 in possible_neighbours:
                    if (polygon[k1,k2] != 0. and (k1,k2) not in vertices):
                        adj_list[i][j].append([k1,k2])
    return adj_list

def build_barycentric_polygon(polygon, vertices, adj_list):
    barycentric_polygon = []
    for vertex in vertices:
        weighted_polygon = polygon.copy()
        for v in vertices:
            if np.all(v != vertex):
                weighted_polygon[tuple(v)] = 0.
        # breadth-first search. At each step, we decrease the associated barycentric weight
        to_visit = [vertex]
        visited = vertices.copy()
        while (to_visit != []):
            parent = to_visit.pop(0)
            for neighbour in adj_list[parent[0]][parent[1]]:
                if neighbour not in visited:
                    weighted_polygon[tuple(neighbour)] = max(weighted_polygon[tuple(parent)] - 1, 0.)
                    to_visit.append(neighbour)
                    visited.append(neighbour)
        barycentric_polygon.append(weighted_polygon)

    barycentric_polygon = np.stack(barycentric_polygon,axis=2)
    
    positive_vals_indices = np.where(barycentric_polygon > 0.)
    barycentric_polygon[positive_vals_indices] = barycentric_polygon[positive_vals_indices] / \
                                                 barycentric_polygon.sum(axis=2)[positive_vals_indices[0:2]]
    
    return barycentric_polygon

def get_barycentric_triangle(n_iters, show=True):
    triangle_coords = [[0,0]]
    triangle_slopes = [(2,1),(0,-2),(-2,1)]
    init_side_len = 2
    side_step = 3
    top_vertex_step = -4
    recenter_step = (4, 3)
    img_side = n_iters * 6 + 2
    init_val = init_side_len + (n_iters-1) * side_step + 1

    triangle_coords, vertices = build_polygon(triangle_coords, triangle_slopes, n_iters, init_side_len,
                                              side_step, top_vertex_step, recenter_step)
    triangle = get_polygon_matrix(img_side, n_iters, triangle_coords, init_val)
    if show:
        plt.figure();plt.imshow(triangle,cmap='Greys');plt.show()

    adj_list = get_adjacency_list(triangle, vertices, img_side)
    barycentric_triangle = build_barycentric_polygon(triangle, vertices, adj_list)
    if show:
        for i in range(barycentric_triangle.shape[-1]):
            plt.figure();plt.imshow(barycentric_triangle[:,:,i],cmap='Greys');plt.show()
    
    return triangle, barycentric_triangle, img_side

def get_barycentric_pentagon(n_iters, show=True):
    penta_coords = [[0,0]]
    penta_slopes = [(2,2),(2,-1),(0,-2),(-2,-1),(-2,2)]
    init_side_len = 0
    side_step = 1
    top_vertex_step = -2
    recenter_step = (2, 2)
    img_side = n_iters * 4 + 2
    init_val = init_side_len + (n_iters-1) * side_step + 2

    penta_coords, vertices = build_polygon(penta_coords, penta_slopes, n_iters, init_side_len,
                                           side_step, top_vertex_step, recenter_step)
    pentagon = get_polygon_matrix(img_side, n_iters, penta_coords, init_val)
    if show:
        plt.figure();plt.imshow(pentagon,cmap='Greys');plt.show()

    adj_list = get_adjacency_list(pentagon, vertices, img_side)
    barycentric_pentagon = build_barycentric_polygon(pentagon, vertices, adj_list)
    if show:
        for i in range(barycentric_pentagon.shape[-1]):
            plt.figure();plt.imshow(barycentric_pentagon[:,:,i],cmap='Greys');plt.show()
    
    return pentagon, barycentric_pentagon, img_side

def get_model_inputs_from_ids(inputs_folder_path, shapes_ids):
    fnames_inputs = [os.path.join(inputs_folder_path, str(shape_id)+'.png') for shape_id in shapes_ids]
    ins = [load_img(fname)[0] for fname in fnames_inputs]
    ins = torch.stack([x / x.sum() for x in ins]).unsqueeze(0)
    return ins

def get_geomloss_targets_from_ids(targets_folder_path, shapes_ids):
    fnames_targets = [str(shape_id)+'.pt' for shape_id in shapes_ids]
    targets = load_targets(fnames_targets, targets_folder_path)
    return targets

def disp_bary_grayscale(ax, bary, vmin, vmax):
    ax.imshow(bary, vmin=vmin, vmax=vmax, cmap='Greys')

def gen_bary_from_targets_pdf(targets, bweights, ins_size):
    rbary = gen_bary_from_targets(targets, bweights, ins_size).cpu().numpy()
    rbary /= rbary.sum()
    return rbary
    
def build_barycentric_polygon_figures(ins, targets, get_pbary, polygon, 
                                      barycentric_polygon, n_cells, 
                                      n_iters, interpol_id='cats', 
                                      results_path='.', 
                                      disp_bary_func=disp_bary_grayscale,
                                      get_rbary=gen_bary_from_targets_pdf,
                                      max_res_per_img=None):
    # we have to do generation & display in 2 steps in order to get the correct vmin & vmax
    # before display
    # barycenter generation
    vmin, vmax = np.inf, -np.inf
    if targets is not None:
        rbarys = [] 
    pbarys = []
    for i in range(n_cells):
        for j in range(0,n_cells):
            if (polygon[i][j] != 0.):
                bweights = barycentric_polygon[i,j]

                # with geomloss:
                if targets is not None:
                    rbary = get_rbary(targets, bweights, ins.shape[-1])

                # with model:
                pbary = get_pbary(ins, bweights)
                
                if targets is not None:
                    rbarys.append(rbary)
                pbarys.append(pbary)
    
    if (max_res_per_img is not None):
        # we rescale images in order to have the maximum desired size
        npix_row = pbary.shape[0]
        npix_col = pbary.shape[1]
        
        if (max(npix_row, npix_col) > max_res_per_img): # only rescale if needed
            if npix_row > npix_col:
                rescaled_cv2_shape = ((max_res_per_img * npix_col) // npix_row,
                                      max_res_per_img)
            else:
                rescaled_cv2_shape = (max_res_per_img,
                                      (max_res_per_img * npix_row) // npix_col)

            if targets is not None:
                rbarys = [cv2.resize(rbary, dsize=rescaled_cv2_shape) 
                          for rbary in rbarys]
            pbarys = [cv2.resize(pbary, dsize=rescaled_cv2_shape) 
                      for pbary in pbarys]
    
    row_cell_size = pbarys[0].shape[0]//2
    col_cell_size = pbarys[0].shape[1]//2
    
    if targets is not None:
        all_barys = np.concatenate(rbarys).reshape(-1)
    else:
        all_barys = np.concatenate(pbarys).reshape(-1)
    
    vmin = np.percentile(all_barys[all_barys>0.],5)
    vmax = np.percentile(all_barys[all_barys>0.],95)

    nrows = n_cells * row_cell_size
    ncols = n_cells * col_cell_size

    if targets is not None:
        fig1 = plt.figure(figsize=(ncols/100,nrows/100), dpi=100)
        gs1 = GridSpec(nrows, ncols, figure=fig1)
    fig2 = plt.figure(figsize=(ncols/100,nrows/100), dpi=100)
    gs2 = GridSpec(nrows, ncols, figure=fig2)
    fig3 = plt.figure(figsize=(ncols/100,nrows/100), dpi=100)
    gs3 = GridSpec(nrows, ncols, figure=fig3)
    format_bweights = ''
    for i in range(ins.shape[1]):
        format_bweights += '{}\n'
    for i in range(n_cells):
        for j in range(n_cells):
            if (polygon[i][j] != 0.):
                # figure row indices
                r1 = i     * row_cell_size
                r2 = (i+2) * row_cell_size
                # figure column indices
                c1 = j     * col_cell_size
                c2 = (j+2) * col_cell_size
                
                # with geomloss:
                if targets is not None:
                    rbary = rbarys.pop(0)
                    ax1 = fig1.add_subplot(gs1[r1:r2,c1:c2]); ax1.axis('off')
                    disp_bary_func(ax1, rbary, vmin, vmax)

                # with model:
                pbary = pbarys.pop(0)
                ax2 = fig2.add_subplot(gs2[r1:r2,c1:c2]); ax2.axis('off')
                disp_bary_func(ax2, pbary, vmin, vmax)

                # display barycentric weights:
                bweights = barycentric_polygon[i,j]
                ax3 = fig3.add_subplot(gs3[r1:r2,c1:c2]); ax3.axis('off')
                
                ax3.text(0.5, 0.5, format_bweights.format(*[round(bweight,2) for bweight in bweights]), ha='center', va='center', fontsize=25)
                # note: fontsize should not be constant

    # remove spaces & margins in order to have each image displayed as a img_size*img_size image
    # and to thus obtain a final (nrows*img_size)*(nrows*img_size) image. 
    if targets is not None:
        fig1.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    fig2.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    fig3.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    
    if targets is not None:
        fig1.savefig(os.path.join(results_path, '{}{}_niters={}_geomloss.png'.format(ins.shape[1], interpol_id, n_iters)))
    fig2.savefig(os.path.join(results_path, '{}{}_niters={}_model.png'.format(ins.shape[1], interpol_id, n_iters)))
    fig3.savefig(os.path.join(results_path, '{}{}_niters={}_weights.png'.format(ins.shape[1], interpol_id, n_iters)))
