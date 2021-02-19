#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:43:17 2018

@author: mducoffe, rflammary, ncourty
"""

import numpy as np
import scipy as sp
import os
import h5py
from tensorflow.keras.utils import Sequence
from scipy import io
from tensorflow.keras.datasets import mnist

#%%
MNIST='mnist'
RANDSHAPES='randshapes'
CAT='cat'
CRAB='crab'
FACE='face'
REPO='data'
CONTOURS_28X28='input_shapes_28x28_v2'
#%%

def get_pairwise_index(dataset_name='mnist',repo='data', train=True):
    i=0
    var = 'train' if train else 'test'
    tmp_filename = os.path.join(REPO, '{}_{}_'.format(dataset_name, var))
    assert os.path.isfile(tmp_filename+'{}.mat'.format(i)), 'error: no emd recorded for {}ing {}'.format(var, dataset_name)
     
    while os.path.isfile(tmp_filename+'{}.mat'.format(i)): i+=1
    nmax=i
     
    ytot=np.zeros((0))
    i1=np.zeros((0))
    i2=np.zeros((0))

    for i in range(nmax):
        data=io.loadmat(tmp_filename+'{}.mat'.format(i))
        iss=data['is'].ravel()
        it=data['it'].ravel()
        D=data['D'].ravel()
        i1=np.append(i1,iss)
        i2=np.append(i2,it)
        ytot=np.append(ytot,D)
         
    i1=i1.astype(int)
    i2=i2.astype(int)
    
    return i1, i2, ytot

#%%
def get_data(dataset_name='mnist', repo='data', labels=False):
    
    assert dataset_name in [MNIST, CAT, CRAB, FACE, CONTOURS_28X28], 'unknown dataset {}'.format(dataset_name)
    
    if dataset_name==MNIST:
        n = 28
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        X_train =x_train.reshape((len(x_train),-1))*1.0
        # normalize each sample
        X_train/=X_train.sum(1).reshape((-1,1))
        X_train=X_train.reshape((-1,n,n,1))
        
        X_test =x_test.reshape((len(x_test),-1))*1.0
        # normalize each sample
        X_test/=X_test.sum(1).reshape((-1,1))
        X_test=X_test.reshape((-1,n,n,1))
        
        # splitting into training and validation
        i1_train, i2_train, emd_train = get_pairwise_index(dataset_name, repo, train=True)
        i1_test, i2_test, emd_test = get_pairwise_index(dataset_name, repo, train=False)
        
        N = len(i1_train)
        n_train = (int)(0.8*N)
        
        if not(labels):
            data_train = (X_train[i1_train[:n_train]], X_train[i2_train[:n_train]], emd_train[:n_train])
            data_valid = (X_train[i1_train[n_train:]], X_train[i2_train[n_train:]], emd_train[n_train:])
            data_test = (X_train[i1_test], X_train[i2_test], emd_test)
        else:
            data_train = (X_train[i1_train[:n_train]], X_train[i2_train[:n_train]], emd_train[:n_train], \
                          y_train[i1_train[:n_train]], y_train[i2_train[:n_train]])
            data_valid = (X_train[i1_train[n_train:]], X_train[i2_train[n_train:]], emd_train[n_train:], \
                          y_train[i1_train[n_train:]], y_train[i2_train[n_train:]])
            data_test = (X_train[i1_test], X_train[i2_test], emd_test, y_test[i1_test], y_test[i2_test])
            
    if dataset_name in [CAT, CRAB, FACE, CONTOURS_28X28]:
        n=28
        if dataset_name in [CAT, CRAB, FACE]:
            #assert files are alerady download, otherwise download them
            url_path = "https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap"

            assert os.path.isfile(os.path.join(REPO, '{}.npy'.format(dataset_name))), \
                "file not found: please download it at '{}' and put it in './{}/{}.npy'".format(url_path, REPO, dataset_name)
            
        X = np.load(os.path.join(REPO, '{}.npy'.format(dataset_name)))
        X = X.reshape((len(X),-1))*1.0
        X /=X.sum(1).reshape((-1,1))
        X=X.reshape((-1,n,n,1))
        
        # split into train, and test
        N = len(X)
        n_test = (int)(0.2*N)
        n_train = N - 2*n_test
        X_train = X[:n_train]
        X_test = X[n_train:]
        
        # splitting into training and validation
        i1_train, i2_train, emd_train = get_pairwise_index(dataset_name, repo, train=True)
        i1_test, i2_test, emd_test = get_pairwise_index(dataset_name, repo, train=False)
        
        N = len(i1_train)
        n_train = (int)(0.8*N)
        
        data_train = (X_train[i1_train[:n_train]], X_train[i2_train[:n_train]], emd_train[:n_train])
        data_valid = (X_train[i1_train[n_train:]], X_train[i2_train[n_train:]], emd_train[n_train:])
        data_test = (X_test[i1_test], X_test[i2_test], emd_test)
    
    return data_train, data_valid, data_test


class RandShapesWDistSeq(Sequence):
    def __init__(self, shapes_fpath, wdists_fpath, idx_interval, batch_size=32):
        super(RandShapesWDistSeq, self).__init__()
        
        self.batch_size = batch_size
        self.shapes_fpath = shapes_fpath
        self.wdists_fpath = wdists_fpath
        self.idx_interval = idx_interval
        
        self.shapes_file = h5py.File(shapes_fpath, 'r')
        self.wdists_file = h5py.File(wdists_fpath, 'r')
        
        self.inputs_dset = self.shapes_file['input_shapes']
        self.wdists_dset = self.wdists_file['wdists']
        self.ids_dset    = self.wdists_file['input_ids']
        
        self.n_wdists = idx_interval[1] - idx_interval[0]

    def __getitem__(self, index):
        index0 = self.idx_interval[0] + index
        wdist = self.wdists_dset[index0:index0+self.batch_size].astype(np.float)
        all_input_ids = self.ids_dset[index0:index0+self.batch_size]
        input_ids1 = all_input_ids[:,0]
        input_ids2 = all_input_ids[:,1]
        x1 = [self.inputs_dset[input_id].astype(np.float).reshape((512,512,1))
                   for input_id in input_ids1]
        x1 = np.array([x / x.sum() for x in x1])
        x2 = [self.inputs_dset[input_id].astype(np.float).reshape((512,512,1))
                   for input_id in input_ids2]
        x2 = np.array([x / x.sum() for x in x2])
        wdist = np.array(wdist)
        y = wdist
        return [x1, x2], [y, x1, x2, x1, x2]
        
    def __len__(self):
        return self.n_wdists


#%%
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--dataset_name', type=str, default='cat', help='dataset name')
    parser.add_argument('--repo', type=str, default=REPO, help='repository to stock the dataset')
    args = parser.parse_args()
                                                                   
    dataset_name=args.dataset_name
    repo=args.repo

    get_data(dataset_name, repo)
    get_pairwise_index(dataset_name, repo)
    