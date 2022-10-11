import os
import sys
import torch
import tensorflow as tf
import numpy as np
from matplotlib.animation import PillowWriter
from skimage.transform import resize

# allows imports from parent directories
sys.path.insert(0, os.path.dirname(os.getcwd()))

from models import *
from model_util import *
from barycentric_polygon_comparison import *
from bary_gen_util import *
from anim_util import *
from io_util import analyse_args

def build_barycentric_line_figure(ins, targets, get_pbary, bweights, n_barys, 
                                  interpol_id='cats', results_path='.', 
                                  display_inputs=False, img_out_size=None):
    if targets is not None:
        rbarys = []
    if (display_inputs):
        merged_ins = ins[0,0].numpy().copy()
        for input_fig in ins[0,1:]:
            merged_ins = np.where(merged_ins <= 0., input_fig, merged_ins)
        merged_ins[merged_ins > 0.] = 1. # our goal here is just to see the inputs locations
        
    alpha = 0.9 if display_inputs else 1.0
    ninputs = ins.shape[1]
    img_size = ins.shape[-1]
        
    folder_path = os.path.join(results_path, '{}{}_line'.format(ninputs, interpol_id))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
    cmap = 'Blues' if display_inputs else 'Greys'
    pbarys = []
    for i in range(n_barys):
        # with geomloss:
        if targets is not None:
            bary_grid_size = int(math.sqrt(targets.shape[1]))
            rbary = gen_bary_from_targets(targets, bweights[i], bary_grid_size)
            rbary /= rbary.sum()
            rbary = rbary.numpy()
            rbarys.append(rbary)
        
        # with model
        pbary = get_pbary(ins, bweights[i])
        pbarys.append(pbary)
    
    if (targets is not None):
        if (img_out_size is not None):
            rbarys = [resize(rbary, img_out_size) for rbary in rbarys]
        elif (bary_grid_size != img_size):
            rbarys = [resize(rbary, (img_size, img_size)) for rbary in rbarys]
        rbarys = [rbary / rbary.sum() for rbary in rbarys]
    if (img_out_size is not None):
        pbarys = [resize(pbary, img_out_size) for pbary in pbarys]
        pbarys = [pbary / pbary.sum() for pbary in pbarys]
        
    img_size = rbarys[i].shape[-1]
    
    if targets is not None:
        all_barys = np.concatenate(rbarys).reshape(-1)
    else:
        all_barys = np.concatenate(pbarys).reshape(-1)
    vmin = np.percentile(all_barys[all_barys>0.],5)
    vmax = np.percentile(all_barys[all_barys>0.],95)
    
    nrows = 1+(0 if targets is None else 1)
    fig = plt.figure(figsize=(n_barys, nrows), dpi=img_size)
    gs = GridSpec(nrows, n_barys, figure=fig)

    for i in range(n_barys):
        pbary = pbarys.pop(0)
        
        if targets is None:
            ax = fig.add_subplot(gs[i]); ax.axis('off')
            if (display_inputs):
                ax.imshow(merged_ins, cmap='Greys')
            ax.imshow(pbary, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha)
            plt.figure(figsize=(1,1),dpi=512);plt.imshow(pbary,vmin=vmin,vmax=vmax,cmap='Greys');plt.axis('off');plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0);plt.savefig(os.path.join(folder_path,'pbary_{}.png').format(i));plt.close()
        else:
            rbary = rbarys.pop(0)
            ax1 = fig.add_subplot(gs[i]); ax1.axis('off')
            if (display_inputs):
                ax1.imshow(merged_ins, cmap='Greys')
            ax1.imshow(rbary, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha)
            ax2 = fig.add_subplot(gs[n_barys+i]); ax2.axis('off')
            if (display_inputs):
                ax2.imshow(merged_ins, cmap='Greys')
            ax2.imshow(pbary, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha)
            plt.figure(figsize=(1,1),dpi=512);plt.imshow(rbary,vmin=vmin,vmax=vmax,cmap='Greys');plt.axis('off');plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0);plt.savefig(os.path.join(folder_path,'rbary_{}.png').format(i));plt.close()
            plt.figure(figsize=(1,1),dpi=512);plt.imshow(pbary,vmin=vmin,vmax=vmax,cmap='Greys');plt.axis('off');plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0);plt.savefig(os.path.join(folder_path,'pbary_{}.png').format(i));plt.close()

    # remove spaces & margins in order to have each image displayed as a img_size*img_size image
    # and to thus obtain a final (nrows*img_size)*(nrows*img_size) image. 
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    
    if (display_inputs):
        folder_path += '_with_inputs'
    folder_path += '.png'
    print(folder_path)
    fig.savefig(folder_path)
    plt.close(fig)

def get_bary_evo(ins, step, get_pbary):
    ws = np.arange(0,1+step,step)
    bary_evo = []
    with torch.no_grad():
        for w in ws:
            bweights = np.array([1-w,w])
            
            pred_bary = get_pbary(ins, bweights)
            bary_evo.append(pred_bary)
    bary_evo = np.stack(bary_evo)
    return bary_evo

def get_geomloss_bary_evo(targets, step):
    ws = np.arange(0,1+step,step)
    bary_evo = []
    img_size = int(math.sqrt(targets[0].shape[0]))
    for w in ws:
        bweights = [1-w,w]
        bary = bweights[0] * targets[0] + bweights[1] * targets[1]
        bin_bary = bin_barycenter_numpy(bary, img_size).astype(np.float32)
        bin_bary /= bin_bary.sum()

        bary_evo.append(bin_bary)
    bary_evo = np.stack(bary_evo)
    return bary_evo

def get_random_bweights(n_barys, n_bweights, n_decimals=4):
    x = np.random.rand(n_barys, n_bweights)
    x = x / x.sum(axis=1, keepdims=True)
    x = np.round(x, 3)
    return x

def preprocess_anim_params(anim_params_str):
    anim_params = anim_params_str.split(',')
    anim_params[0] = float(anim_params[0])
    anim_params[1] = int(anim_params[1])
    anim_params[2] = int(anim_params[2])
    return anim_params

def get_pbary_bunet(model):
    def get_pbary_inner(ins, bweights):
        model_bweights = torch.FloatTensor(bweights.reshape(1,ins.shape[1],1,1))
        with torch.no_grad():
            pbary = model(ins.cuda(), model_bweights.cuda())[0][0].detach().cpu()
        return pbary.numpy()
    return get_pbary_inner

def get_pbary_dwe(feat, unfeat):
    def get_pbary_inner(ins, bweights):
        ins = np.array(ins[0].unsqueeze(-1))
        embeddings = feat.predict(ins)
        wbary_embedding = embeddings[0] * bweights[0]
        
        for i in range(1, ins.shape[0]):
            wbary_embedding += embeddings[i] * bweights[i]
            
        pbary = unfeat.predict(wbary_embedding[None])[0,:,:,0]
        
        return pbary
    return get_pbary_inner

def main():
    args = analyse_args([
        ['m',       'model_id', lambda x: x,            'model_id'],
        ['r',   'results_path', lambda x: x,            'results'],
        ['k',  'interpol_type', lambda x: x,            'line'], 
        ['c',    'model_class', lambda x: x,            'BarycentricUNet'], 
        ['f',   'input_folder', lambda x: x,            'input_imgs'], 
        ['l',   'input_fnames', lambda x: x.split(','), []],
        ['t',  'target_folder', lambda x: x,            None], # default None = don't generate geomloss 
                                                               # barycenters
        ['i',    'interpol_id', lambda x: x,            'interpolations'],
        ['n',        'n_iters', lambda x: int(x),       3], # note: n is used as the nb of barycenters when 
                                                            #       using type=line without defining bweights
        ['w',       'bweights', lambda x: np.array([[float(e) for e in y.split(',')] 
                                                              for y in x.split('|')]), None],
        ['a',    'anim_params', preprocess_anim_params, [0.01, 20, 30]],
        ['d', 'display_inputs', lambda x: x == 'True', False], # used when interpol_type == 'line'
        ['s',   'img_out_size', lambda x: [int(n) for n in x.split(',')], None]
    ])
    
    # loading model
    if args['model_class'] == 'BarycentricUNet':
        model_results_path = os.path.join('..', 'training_model', 'results', args['model_id'])
        flags_path = os.path.join(model_results_path, 'flags.pkl')
        model_path = os.path.join(model_results_path, 'model.pth')
        FLAGS = load_flags(flags_path)
        model = load_model_on_gpu(model_path, BarycentricUNet, 
                                  [FLAGS[param_name] for param_name in 
                                                         FLAGS['model_params_names']]).eval()
        get_pbary = get_pbary_bunet(model)
    elif args['model_class'] == 'DWE':
        model_results_path = os.path.join('..','dwe','models',args['model_id'])
        feat_path   = model_results_path + 'feat.hd5'
        unfeat_path = model_results_path + 'unfeat.hd5'
        feat   = tf.keras.models.load_model(feat_path)
        unfeat = tf.keras.models.load_model(unfeat_path)
        
        get_pbary = get_pbary_dwe(feat, unfeat)
    else:
        print('Error: undefined model class {}'.format(args['model_class']))
        sys.exit(-1)
    
    # loading inputs & geomloss targets (if provided)
    ins = get_model_inputs_from_ids(args['input_folder'], args['input_fnames'])
    if args['target_folder'] is None:
        targets = None
    else:
        targets = get_geomloss_targets_from_ids(args['target_folder'], args['input_fnames'])
    
    
    # generating barycenters
    if args['interpol_type'] in ['triangle', 'pentagon']:
        get_barycentric_polygon = get_barycentric_triangle if args['interpol_type'] == 'triangle' \
                                                           else get_barycentric_pentagon
        polygon, barycentric_polygon, img_side = get_barycentric_polygon(args['n_iters'],
                                                                         show=False)
            
        build_barycentric_polygon_figures(ins, targets, get_pbary, polygon, barycentric_polygon, 
                                          img_side, args['n_iters'], 
                                          interpol_id=args['interpol_id'], 
                                          results_path=args['results_path'])
    elif args['interpol_type'] == 'line':
        if args['bweights'] is None:
            n_barys = args['n_iters'] # when using interpol_type=line, n_iters argument 
                                      # corresponds to the nb of barycenters
            args['bweights'] = get_random_bweights(n_barys, len(args['input_fnames']))
        else:
            n_barys = len(args['bweights'])
        
        build_barycentric_line_figure(ins, targets, get_pbary, args['bweights'], 
                                      n_barys, interpol_id=args['interpol_id'], 
                                      results_path=args['results_path'], 
                                      display_inputs=args['display_inputs'],
                                      img_out_size=args['img_out_size'])
        np.savetxt(os.path.join(args['results_path'], 
                                '{}{}_line_bweights.csv'.format(ins.shape[1], 
                                                                args['interpol_id'])), 
                   args['bweights'], delimiter=',', fmt='%.3f')
    elif args['interpol_type'] == 'anim':
        step, interval, fps = args['anim_params']
        
        if targets is not None:
            geom_bary_evo = get_geomloss_bary_evo(targets.numpy(), step)
            vmin = np.percentile(geom_bary_evo[geom_bary_evo>0.],  5)
            vmax = np.percentile(geom_bary_evo[geom_bary_evo>0.], 95)
            anim = interpolate_between_2_shapes(geom_bary_evo, interval, vmin=vmin, vmax=vmax)
            anim_path = os.path.join(args['results_path'], 'anim_geomloss_{}_{}_{}fps.gif'.format(*args['input_fnames'], fps))
            anim.save(anim_path, writer=PillowWriter(fps=fps))
            
        model_bary_evo = get_bary_evo(ins, step, get_pbary)
        if targets is None:
            vmin = np.percentile(model_bary_evo[model_bary_evo>0.],  5)
            vmax = np.percentile(model_bary_evo[model_bary_evo>0.], 95)
        anim = interpolate_between_2_shapes(model_bary_evo, interval, vmin=vmin, vmax=vmax)
        anim.save(os.path.join(args['results_path'], 
                               'anim_model_{}_{}_{}fps.gif'.format(*args['input_fnames'], fps)), 
                  writer=PillowWriter(fps=fps))

if __name__ == "__main__":
    main()
