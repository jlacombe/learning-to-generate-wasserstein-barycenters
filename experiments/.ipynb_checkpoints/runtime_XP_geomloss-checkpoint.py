# pykeops==1.4.0 + geomloss + torch==1.4.0 (nok with torch>=1.5)
import os
import sys
import getopt
import time
from geomloss import SamplesLoss

# allows imports from parent directory
sys.path.insert(0, os.path.dirname(os.getcwd()))

from data_loading_util import *
from bary_gen_util import *
from model_util import *
from io_util import analyse_args

def eval_runtimes_geom(dataloader, with_sparse_measures=False):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    M, N = 512, 512
    Loss = SamplesLoss("sinkhorn", blur=.01, scaling=.9, backend='multiscale')
    
    data_time, model_time = 0., 0.
    t0 = time.time()
    t3 = time.time()
    for i, (ins, _, ws) in enumerate(dataloader):
        if with_sparse_measures:
            input_measures = [as_sparse_measure(ins[0][i].to(device), device) for i in range(2)]
        else:
            input_measures = [as_measure(ins[0][i].to(device), M, device) for i in range(2)]
        ws = ws.reshape(-1).to(device)
        t4 = time.time()
        data_time += (t4 - t3)

        t1 = time.time()

        bary = gen_barycenter(input_measures, ws, Loss, M, N, device, 1)
        bin_bary = bin_barycenter(bary, M)
        bin_bary /= bin_bary.sum()

        t2 = time.time()
        model_time += (t2 - t1)

        t3 = time.time()
    total_time = time.time() - t0
    return data_time, model_time, total_time

def print_and_trace_runtimes(runtimes, n_barys, results_XPs_path, fname):
    data_time, model_time, total_time = runtimes
    print_and_trace('             data_time: {:.4f}s'.format(data_time),          results_XPs_path, fname=fname)
    print_and_trace('      total model_time: {:.4f}s'.format(model_time),         results_XPs_path, fname=fname)
    print_and_trace('model_time for 1 wbary: {:.4f}s'.format(model_time/n_barys), results_XPs_path, fname=fname)
    print_and_trace('            total_time: {:.4f}s'.format(total_time),         results_XPs_path, fname=fname)

def main():
    args = analyse_args([
        ['r', 'results_path', lambda x: x,      '.'],
        ['n',      'n_barys', lambda x: int(x), 10],
        ['i', 'input_shapes_path', lambda x: x, os.path.join('..', 'datasets', 'input_contours.h5')],
        ['b',        'barys_path', lambda x: x, os.path.join('..', 'datasets', 'barycenters_contours.h5')]
    ])
    
    ds = GeomlossBary(args['input_shapes_path'], args['barys_path'], (0, args['n_barys']))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
    
    fname = 'runtimes_geomloss_{}.txt'.format(args['n_barys'])
    
    print_and_trace('GEOMLOSS RUNTIMES -- n_barys={}'.format(args['n_barys']), 
                    args['results_path'], fname=fname, mode='w')
    print_and_trace('\nDENSE GEOMLOSS RUNTIMES:', args['results_path'], fname=fname)
    geom_runtimes = eval_runtimes_geom(loader)
    print_and_trace_runtimes(geom_runtimes, args['n_barys'], args['results_path'], fname)
    
    print_and_trace('\nSPARSE GEOMLOSS RUNTIMES:', args['results_path'], fname=fname)
    geom_sparse_runtimes = eval_runtimes_geom(loader, with_sparse_measures=True)
    print_and_trace_runtimes(geom_sparse_runtimes, args['n_barys'], args['results_path'], fname)
        
    
if __name__ == "__main__":
    main()
