import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# allows imports from parent directories
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from io_util import analyse_args


def main():
    args = analyse_args([
        ['m',    'model_ids', lambda x: x.split(','), ['bnet_model_id', 'dwe_model_id']],
        ['r', 'results_path', lambda x: x, './results'],
        ['t',  'axes_titles', lambda x: x.split(','), 'Ours,DWE'],
        ['l',         'loss', lambda x: x, 'kldiv_loss'],
        ['n',      'n_barys', lambda x: int(x), 1000]
    ])
    
    errs_fpath1 = os.path.join(args['results_path'], 'errs_{}_{}_{}.csv'.format(args['loss'], 
                                                                                args['n_barys'], 
                                                                                args['model_ids'][0]))
    errs1 = np.genfromtxt(errs_fpath1, delimiter=',')
    errs_fpath2 = os.path.join(args['results_path'], 'errs_{}_{}_{}.csv'.format(args['loss'],
                                                                                args['n_barys'], 
                                                                                args['model_ids'][1]))
    errs2 = np.genfromtxt(errs_fpath2, delimiter=',')
    
    err_max = max(errs1.max(),errs2.max())
    plt.figure()
    plt.plot(errs1, errs2, '+')
    plt.plot([0., err_max], [0., err_max], 'k')
    plt.xlabel(args['axes_titles'][0])
    plt.ylabel(args['axes_titles'][1])
    res_fpath = os.path.join(args['results_path'], 
                             'errs_{}_vs_{}__{}_{}.png'.format(*args['axes_titles'], 
                                                               args['loss'], args['n_barys']))
    plt.savefig(res_fpath)
    
if __name__ == "__main__":
    main()
