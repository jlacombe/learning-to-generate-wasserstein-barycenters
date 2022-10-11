import os
import sys
import tensorflow as tf
import torch

# allows imports from parent directories
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from models import *
from model_util import *
from custom_losses import *
from data_loading_util import *
from io_util import analyse_args

def get_pbary_bunet(model):
    def get_pbary_inner(ins, bweights):
        model_bweights = torch.FloatTensor(bweights.reshape(1,ins.shape[1],1,1))
        with torch.no_grad():
            pbary = model(ins.cuda(), model_bweights.cuda()).detach().cpu()
        return pbary
    return get_pbary_inner

def get_pbary_dwe(feat, unfeat):
    def get_pbary_inner(ins, bweights):
        ins = np.array(ins[0].unsqueeze(-1))
        embeddings = feat.predict(ins)
        
        wbary_embedding = embeddings[0] * bweights[0]
        
        for i in range(1, ins.shape[0]):
            wbary_embedding += embeddings[i] * bweights[i]
            
        pbary = unfeat.predict(wbary_embedding[None])[0,:,:,0]
        
        return torch.FloatTensor(pbary.reshape(1,1,*pbary.shape))
    return get_pbary_inner

def main():
    args = analyse_args([
        ['m',     'model_id', lambda x: x, 'model_id'],
        ['c',  'model_class', lambda x: x, 'BarycentricUNet'],
        ['r', 'results_path', lambda x: x, './results'],
        ['i',  'inputs_path', lambda x: x, '../datasets/input_contours.h5'],
        ['b',   'barys_path', lambda x: x, '../datasets/barycenters_contours.h5'],
        ['l',         'loss', lambda x: x, 'kldiv_loss'],
        ['n',      'n_barys', lambda x: int(x), 1000]
    ])
    
    # loading model
    if args['model_class'] == 'BarycentricUNet':
        model_results_path = os.path.join('..','training_model','results', args['model_id'])
        flags_path = os.path.join(model_results_path, 'flags.pkl')
        model_path = os.path.join(model_results_path, 'model.pth')
        FLAGS = load_flags(flags_path)
        model = load_model_on_gpu(model_path, BarycentricUNet, 
                                  [FLAGS[param_name] for param_name in 
                                                         FLAGS['model_params_names']]).eval()
        get_pbary = get_pbary_bunet(model)
    elif args['model_class'] == 'DWE':
        # manual GPU memory limitation for tensorflow
        MAX_GPU_MEM_TF = 4096 # maximal amount of GPU memory tensorflow can use
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MAX_GPU_MEM_TF)])
            except RuntimeError as e:
                print(e)
        
        model_results_path = os.path.join('..','dwe','models',args['model_id'])
        feat_path   = model_results_path + 'feat.hd5'
        unfeat_path = model_results_path + 'unfeat.hd5'
        feat   = tf.keras.models.load_model(feat_path)
        unfeat = tf.keras.models.load_model(unfeat_path)

        get_pbary = get_pbary_dwe(feat, unfeat)
    else:
        print('Error: undefined model class {}'.format(args['model_class']))
        sys.exit(-1)
    
    train_part, valid_part, test_part = 0.8, 0.1, 0.1
    n_data = 1000
    batch_size = 1
    dataloaders = load_geomloss_barys(args['inputs_path'], args['barys_path'], train_part, 
                                      valid_part, test_part, n_data, batch_size)
        
    if args['loss'] == 'l1_loss' :
        loss = l1_loss
    elif args['loss'] == 'kldiv_loss' :
        loss = kldiv_loss
    elif args['loss'] == 'mmd_loss' :
        loss = mmd_loss
    else :
        loss = l1_loss
    
    errs = []
    for i in range(args['n_barys']):
        ins, rbary, ws = dataloaders['test_loader'].dataset[i]
        ins = ins.unsqueeze(0)
        rbary = rbary.unsqueeze(0)
        ws = ws.reshape(-1).numpy()
        
        pbary = get_pbary(ins, ws)
        
        err = loss(pbary, rbary).item()
        errs.append(err)
    errs = np.array(errs)
    errs_fname = 'errs_{}_{}_{}.csv'.format(args['loss'], 
                                            args['n_barys'], 
                                            args['model_id'])
    errs_fpath = os.path.join(args['results_path'], errs_fname)
    np.savetxt(errs_fpath, errs, delimiter=',')
    
if __name__ == "__main__":
    main()
