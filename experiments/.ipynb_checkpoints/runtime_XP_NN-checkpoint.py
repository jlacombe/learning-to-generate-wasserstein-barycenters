import os
import sys
import time

# allows imports from parent directories
sys.path.insert(0, os.path.dirname(os.getcwd()))

from data_loading_util import *
from models import *
from model_util import *
from io_util import analyse_args

def eval_runtimes_nn(dataloader, model):
    data_time, model_time = 0., 0.
    t0 = time.time()
    t3 = time.time()
    with torch.no_grad():
        for i, (ins, _, ws) in enumerate(dataloader):
            ins = ins.cuda()
            ws = ws.cuda()
            t4 = time.time()
            data_time += (t4 - t3)

            t1 = time.time()
            _ = model(ins, ws)
            t2 = time.time()
            model_time += (t2 - t1)
            
            t3 = time.time()
    total_time = time.time() - t0
    return data_time, model_time, total_time

def print_and_trace_runtimes(runtimes, n_barys, batch_size, results_XPs_path, fname):
    data_time, model_time, total_time = runtimes
    print_and_trace('             data_time: {:.4f}s'.format(data_time), 
                    results_XPs_path, fname=fname)
    print_and_trace('      total model_time: {:.4f}s'.format(model_time), 
                    results_XPs_path, fname=fname)
    print_and_trace('model_time for 1 batch: {:.4f}s'.format(model_time/(n_barys/batch_size)), 
                    results_XPs_path, fname=fname)
    print_and_trace('model_time for 1 wbary: {:.4f}s'.format(model_time/n_barys), 
                    results_XPs_path, fname=fname)
    print_and_trace('            total_time: {:.4f}s'.format(total_time), 
                    results_XPs_path, fname=fname)

def main():
    args = analyse_args([
        ['m',     'model_id', lambda x: x,      'model_id'],
        ['r', 'results_path', lambda x: x,      '.'],
        ['n',      'n_barys', lambda x: int(x), 10],
        ['i', 'input_shapes_path', lambda x: x, os.path.join('..', 'datasets', 'input_contours.h5')],
        ['b',        'barys_path', lambda x: x, os.path.join('..', 'datasets', 'barycenters_contours.h5')]
    ])
    
    # loading model
    model_results_path = os.path.join('..', 'training_model', 'results', args['model_id'])
    flags_path = os.path.join(model_results_path, 'flags.pkl')
    model_path = os.path.join(model_results_path, 'model.pth')
    FLAGS = load_flags(flags_path)
    model = load_model_on_gpu(model_path, BarycentricUNet, [FLAGS[param_name] 
                                                           for param_name in FLAGS['model_params_names']]).eval()
    
    batch_sizes = [1,2,4,8]
    fname = 'runtimes_NN_{}.txt'.format(args['n_barys'])
    print_and_trace('NN RUNTIMES -- n_barys={}'.format(args['n_barys']), 
                    args['results_path'], fname=fname, mode='w')
    for batch_size in batch_sizes:
        print_and_trace('\nBS = {}'.format(batch_size), args['results_path'], fname=fname)
        ds = GeomlossBary(args['input_shapes_path'], args['barys_path'], (0, args['n_barys']))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)
        nn_runtimes = eval_runtimes_nn(loader, model)
        print_and_trace_runtimes(nn_runtimes, args['n_barys'], batch_size, args['results_path'], fname)
        
    
if __name__ == "__main__":
    main()
