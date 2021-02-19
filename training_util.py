import sys
import os
import torch
import time
import matplotlib
from matplotlib.gridspec import GridSpec
from torch import optim

from model_util import *

def display_losses(iters, evo_train_loss, evo_eval_loss, save=True, 
                   folder_path=os.path.join('results', 'default'), show=True):
    plt.figure()
    plt.plot(iters, evo_train_loss, 'c--', label='train')
    plt.plot(iters, evo_eval_loss,  'c',   label='eval')
    plt.title('Evolution of evaluation loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.legend()
    if (save):
        plt.savefig(os.path.join(folder_path, 'eval_loss_evo.png'))
    if (show):
        plt.show()

def display_metric(iters, metric_name, evo_metric, save=True, 
                   folder_path=os.path.join('results', 'default'), 
                   log_scale=False, show=True):
    plt.figure()
    plt.plot(iters, evo_metric, 'c')
    plt.title('Evolution of {} on the evaluation dataset'.format(metric_name))
    plt.xlabel('Iteration')
    plt.ylabel('{} Value'.format(metric_name))
    if log_scale:
        plt.yscale('log')
    if (save):
        plt.savefig(os.path.join(folder_path, 'eval_{}.png'.format(metric_name)))
    if (show):
        plt.show()

def compare_real_pred(test_loader, model, n_samples=5, 
                      parent_folder_path=None, cmap='binary', 
                      with_inputs=True, show=True):
    display_dataset = test_loader.dataset
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_dpi_per_img = 4
    img_size = 512
    dpi = img_size // n_dpi_per_img
    col_mult = 4 if with_inputs else 2
    ncols = col_mult * n_dpi_per_img + 3
    nrows = n_samples * n_dpi_per_img + 1 + n_samples
    
    fig = plt.figure(figsize=(ncols, nrows), dpi=dpi)
    gs = GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    
    bbox = dict(boxstyle='round', fc='cyan', alpha=0.3)
    if (with_inputs):
        ax = fig.add_subplot(gs[0,0:n_dpi_per_img]); ax.axis('off')
        ax.text(.5, .5, 'input 1', horizontalalignment='center',
                verticalalignment='center', fontsize=25, bbox=bbox)
        ax = fig.add_subplot(gs[0,n_dpi_per_img:n_dpi_per_img*2]); ax.axis('off')
        ax.text(.5, .5, 'input 2', horizontalalignment='center',
                verticalalignment='center', fontsize=25, bbox=bbox)
        ax = fig.add_subplot(gs[0,n_dpi_per_img*2:n_dpi_per_img*3]); ax.axis('off')
        ax.text(.5, .5, 'real', horizontalalignment='center',
                verticalalignment='center', fontsize=25, bbox=bbox)
        ax = fig.add_subplot(gs[0,n_dpi_per_img*3:n_dpi_per_img*4]); ax.axis('off')
        ax.text(.5, .5, 'pred', horizontalalignment='center',
                verticalalignment='center', fontsize=25, bbox=bbox)
    else:
        ax = fig.add_subplot(gs[0,0:n_dpi_per_img]); ax.axis('off')
        ax.text(.5, .5, 'real', horizontalalignment='center',
                verticalalignment='center', fontsize=25, bbox=bbox)
        ax = fig.add_subplot(gs[0,n_dpi_per_img:n_dpi_per_img*2]); ax.axis('off')
        ax.text(.5, .5, 'pred', horizontalalignment='center',
                verticalalignment='center', fontsize=25, bbox=bbox)
    
    with torch.no_grad():
        for id_batch, i in zip(range(1,nrows), range(1,nrows,n_dpi_per_img+1)):
            batch = display_dataset[id_batch]
            inputs, real, weights = batch
            inputs = inputs.to(device)
            real = real.to(device)
            weights = weights.to(device)
            pred = model(inputs.unsqueeze(0), weights.unsqueeze(0))[0] # pass forward
            
            in1, in2 = inputs[0:2].cpu()
            real, pred = real[0].cpu(), pred[0].detach().cpu()
            
            if (with_inputs):
                all_real = torch.stack((in1,in2,real))
                vmin = np.percentile(all_real[all_real>0.], 5)
                vmax = np.percentile(all_real[all_real>0.], 95)
            else:
                vmin = np.percentile(real[real>0.], 5)
                vmax = np.percentile(real[real>0.], 95)

            if (with_inputs):
                ax = fig.add_subplot(gs[i:i+n_dpi_per_img,0:n_dpi_per_img]); ax.axis('off')
                ax.imshow(in1, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title('w1={:.4f}'.format(weights[0][0][0].item()), y=-0.15, fontsize=22)
                ax = fig.add_subplot(gs[i:i+n_dpi_per_img,n_dpi_per_img:n_dpi_per_img*2]); ax.axis('off')
                ax.imshow(in2, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title('w2={:.4f}'.format(weights[1][0][0].item()), y=-0.15, fontsize=22)
                ax = fig.add_subplot(gs[i:i+n_dpi_per_img,n_dpi_per_img*2:n_dpi_per_img*3]); ax.axis('off')
                pc = ax.imshow(real, cmap=cmap, vmin=vmin, vmax=vmax)
                ax = fig.add_subplot(gs[i:i+n_dpi_per_img,n_dpi_per_img*3:n_dpi_per_img*4]); ax.axis('off')
                ax.imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
                ax = fig.add_subplot(gs[i:i+n_dpi_per_img,n_dpi_per_img*4:n_dpi_per_img*4+1]); ax.axis('off')
            else:
                ax = fig.add_subplot(gs[i:i+n_dpi_per_img,0:n_dpi_per_img]); ax.axis('off')
                pc = ax.imshow(real, cmap=cmap, vmin=vmin, vmax=vmax)
                ax = fig.add_subplot(gs[i:i+n_dpi_per_img,n_dpi_per_img:n_dpi_per_img*2]); ax.axis('off')
                ax.imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
                ax = fig.add_subplot(gs[i:i+n_dpi_per_img,n_dpi_per_img*2:n_dpi_per_img*2+1]); ax.axis('off')
            
            ticks = np.linspace(vmin, vmax, 5)
            cbar = fig.colorbar(pc, ticks=ticks, pad=-5.)
            cbar.ax.tick_params(labelsize=22)
            cbar.update_ticks()

    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    
    if (parent_folder_path != None):
        if (with_inputs):
            fname = 'real_vs_pred_with_inputs.png'
        else:
            fname = 'real_vs_pred.png'
        fpath = os.path.join(parent_folder_path, fname)
        plt.savefig(fpath)
    if (show):
        plt.show()

def get_real_pred(batch, model, FLAGS, device, eval=False):
    inputs, real_bary, weights = batch
    inputs = inputs.to(device)
    real_bary = real_bary.to(device)
    weights = weights.to(device)
    if (not eval):
        inputs.requires_grad = True
        weights.requires_grad = True
    pred_bary = model(inputs, weights) # pass forward
    return real_bary, pred_bary

def get_loss_val(batch, model, FLAGS, device, eval=False):
    real_bary, pred_bary = get_real_pred(batch, model, FLAGS, device, eval=eval)
    loss_val = FLAGS['loss'](pred_bary, real_bary)
    return loss_val

def eval_model(FLAGS, test_loader, model, device):
    mean_loss = 0.
    n_mean_metrics = len(FLAGS['eval_metrics'])
    mean_metrics = [0. for _ in range(n_mean_metrics)]
    with torch.no_grad():        
        for batch in test_loader:
            real_bary, pred_bary = get_real_pred(batch, model, FLAGS, device, eval=True)
            loss = FLAGS['loss'](pred_bary, real_bary)
            for i in range(n_mean_metrics):
                mean_metrics[i] += FLAGS['eval_metrics'][i](pred_bary, real_bary)
            mean_loss += loss.item()
    
    mean_loss /= len(test_loader)
    mean_metrics = [metric / len(test_loader) for metric in mean_metrics]
    return mean_loss, mean_metrics

def test_and_visualize_model(FLAGS, best_model, loaders, device, model_fold_path, save, 
                             cmap, n_iter_interval, n_batchs, evo_train_loss, evo_eval_loss, 
                             evos_eval_metrics, evo_lr=None, show=True):
    mean_test_loss, mean_test_metrics = eval_model(FLAGS, loaders['test_loader'], best_model, device)
    print_and_trace('Test err: {:9}'.format(mean_test_loss), model_fold_path)

    for metric_name, metric_val in zip(FLAGS['eval_metrics_names'], mean_test_metrics):
        print_and_trace('Test {} metric: {:9}'.format(metric_name, metric_val), model_fold_path)

    best_model = best_model.module

    if (save):
        compare_real_pred(loaders['test_loader'], best_model, 
                          parent_folder_path=model_fold_path, cmap=cmap, 
                          with_inputs=False, show=show)
        compare_real_pred(loaders['test_loader'], best_model, 
                          parent_folder_path=model_fold_path, cmap=cmap, 
                          with_inputs=True, show=show)
    
    if 'stop_iter' in FLAGS:
        iters = range(n_iter_interval, FLAGS['stop_iter'] + 1, n_iter_interval)
    else:
        iters = range(n_iter_interval, (n_batchs*FLAGS['epochs']) + 1, n_iter_interval)
    folder_path = model_fold_path if save else None
    display_losses(iters, evo_train_loss, evo_eval_loss, save=save, folder_path=folder_path, show=show)

    for metric_name, evo_metric in zip(FLAGS['eval_metrics_names'], evos_eval_metrics):
        display_metric(iters, metric_name, evo_metric, save=save, 
                       folder_path=folder_path, show=show)

    if evo_lr is not None:
        n_iter_lr_interval =  max(1, int(n_batchs * FLAGS['scheduler_interval']))
        if 'stop_iter' in FLAGS:
            iters_lr = range(n_iter_lr_interval, FLAGS['stop_iter'] + 1, n_iter_lr_interval)
        else:
            iters_lr = range(n_iter_lr_interval, (n_batchs*FLAGS['epochs']) + 1, n_iter_lr_interval)
        display_metric(iters_lr, 'lr', evo_lr, save=save, 
                       folder_path=folder_path, log_scale=True, show=show)
    
    return mean_test_loss

def train_model(FLAGS, loaders, model, save=True, 
                parent_res_fold_path='results', res_fold_name=None,
                cmap='binary', show=True):
    tinit = time.time()
    
    if not show:
        matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    
    # initialize the results' folder
    model_fold_path = init_model_folder(parent_res_fold_path, res_fold_name)
    
    # display model's parameters
    print_and_trace(flags_to_str(FLAGS) + '\n', model_fold_path)
    
    # display total number of weights
    n_weigths = str(get_nparams_model(model))
    print_and_trace('Number of weights: {}'.format(n_weigths), model_fold_path)
        
    # move the model from cpu to gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = nn.DataParallel(model) # allows to use multiple GPUs
    # define the optimizer:
    optimizer = FLAGS['optimizer_params'][0](model.parameters(), 
                                             *FLAGS['optimizer_params'][1:])
    
    if 'scheduler_params' in FLAGS:
        scheduler = FLAGS['scheduler_params'][0](optimizer, *FLAGS['scheduler_params'][1:])
    
    DISPLAY_PREC = 10     # number of decimals to display
    evo_train_loss = [] # evolution of the loss on the training set
    evo_eval_loss = []  # evolution of the loss on the evaluation set
    n_mean_metrics = len(FLAGS['eval_metrics'])
    evos_eval_metrics = [[] for _ in range(n_mean_metrics)]  # evolution of the metrics on the evaluation set
    if 'scheduler_params' in FLAGS:
        evo_lr = [] # evolution of the learning rate value accross evaluation iterations
    else:
        evo_lr = None
    
    # evaluation every n_iter_interval 
    n_batchs = len(loaders['train_loader'])
    
    n_iter_interval =  max(1, int(n_batchs * FLAGS['eval_interval']))
    print_and_trace('n_iter_interval={}'.format(n_iter_interval), model_fold_path)
    if 'scheduler_params' in FLAGS:
        n_iter_lr_interval =  max(1, int(n_batchs * FLAGS['scheduler_interval']))
        print_and_trace('n_iter_lr_interval={}'.format(n_iter_lr_interval), model_fold_path)
    if 'stop_iter' in FLAGS:
        print_and_trace('stop_iter={}'.format(FLAGS['stop_iter']), model_fold_path)
    
    n_tot_iter = n_batchs * FLAGS['epochs']
    print_and_trace('Total number of iterations: {}'.format(n_tot_iter), model_fold_path)
    
    # trace title
    title = 'iter    -- train err              -- valid err              -- time (s)'
    print_and_trace(title, model_fold_path)
    csv_results = ','.join([x.strip() for x in title.split('--')]) + '\n'
    
    # define the format used for tracing iterations
    iter_format  = '{:7} -- {:.20f} -- {:.20f} -- {:8}'
    
    # we track the best iter at which the model was the best
    # (for instance it could be the iter where we had the lowest test error)
    best_iter = -1
    
    mean_train_loss = 0.
    
    deltat = time.time() # used to measure the time between 2 evaluations
    model.train() # set model to train mode
    for nepoch in range(1, FLAGS['epochs']+1): # training loop
        for (n_iter, batch) in enumerate(loaders['train_loader']):
            if 'stop_iter' in FLAGS and ((nepoch-1)*(n_iter_interval*10) + n_iter) >= FLAGS['stop_iter']:
                break
            
            train_loss = get_loss_val(batch, model, FLAGS, device)
            
            optimizer.zero_grad()

            train_loss.backward()  # compute the gradient using backpropagation
            optimizer.step() # update the weigths of the cnn using the gradient
            
            mean_train_loss += train_loss.item()
            
            # evaluation
            if ((n_iter+1) % n_iter_interval == 0.):
                model.eval() # set model to evaluation mode
                
                mean_train_loss /= n_iter_interval
                evo_train_loss.append(mean_train_loss)

                # compute mean eval loss over all the validation dataset
                mean_eval_loss, mean_eval_metrics = eval_model(FLAGS, loaders['valid_loader'], 
                                                               model, device)
                evo_eval_loss.append(mean_eval_loss)
                for i in range(len(evos_eval_metrics)):
                    evos_eval_metrics[i].append(mean_eval_metrics[i])

                # trace current iteration
                cur_trace_line = iter_format.format(
                    str((n_iter+1)), 
                    mean_train_loss,
                    mean_eval_loss,
                    str(round(time.time() - deltat, 3))
                )
                print_and_trace(cur_trace_line, model_fold_path)
                csv_results += ','.join([x.strip() for x in cur_trace_line.split('--')]) + '\n'
                    
                # save model
                save_model(model.module, FLAGS, model_fold_path)
                
                mean_train_loss = 0.
                model.train() # set model to train mode
                deltat = time.time()
            
            # update lr
            if 'scheduler_params' in FLAGS and ((n_iter+1) % n_iter_lr_interval == 0.):
                cur_lr = optimizer.param_groups[0]['lr']
                evo_lr.append(cur_lr)
                if isinstance(FLAGS['scheduler_params'], optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(mean_eval_loss)
                else:
                    scheduler.step()

    model.eval() # set model to eval mode
    
    if (save):
        with open(os.path.join(model_fold_path, 'results.csv'), 'w') as f:
            f.write(csv_results)

    
    mean_test_loss = test_and_visualize_model(FLAGS, model, loaders, device, model_fold_path, 
                                              save, cmap, n_iter_interval, n_batchs, evo_train_loss, 
                                              evo_eval_loss, evos_eval_metrics, evo_lr=evo_lr, show=show)
    
    print_and_trace('Elapsed Time: {:.2f}s'.format(time.time() - tinit), model_fold_path)
    return model, model_fold_path, mean_test_loss
