#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:59:48 2018

@author: mducoffe, rflammary, ncourty
"""
import os
import h5py
import tensorflow as tf
import atexit
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger
from tensorflow_addons.layers import InstanceNormalization
from dataset import get_data, MNIST, RANDSHAPES, REPO, RandShapesWDistSeq

MODEL='models'
MODEL_ORIG = 'model_orig'
MODEL_512x512 = 'model_512x512'

def euclidean_distance(vects):
    x, y = vects
    return K.sum(K.square(x - y), axis=(1), keepdims=True)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def sparsity_constraint(y_true, y_pred):
    return K.mean(K.sum(K.sqrt(y_pred+ K.epsilon()), axis=(1,2,3)), axis=0)

def kullback_leibler_divergence_(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.sum(y_true * K.log(y_true / y_pred), axis=(1,2,3)), axis=-1)

def add_block_conv_inorm_relu(model, filters, kernel_size, input_shape=None):
    if input_shape is not None:
        model.add(Conv2D(filters, kernel_size, padding='same', input_shape=input_shape))
    else:
        model.add(Conv2D(filters, kernel_size, padding='same'))
    model.add(InstanceNormalization())
    model.add(Activation('relu'))
    
def add_block_2conv_avgpool(model, filters, kernel_size, pool_size, input_shape=None):
    if input_shape is not None:
        add_block_conv_inorm_relu(model, filters, kernel_size, input_shape=input_shape)
    else:
        add_block_conv_inorm_relu(model, filters, kernel_size)
    add_block_conv_inorm_relu(model, filters, kernel_size)
    model.add(AveragePooling2D(pool_size=pool_size))
    
def add_block_2conv_upsample(model, filters1, filters2, kernel_size, up_size):
    model.add(UpSampling2D(size=up_size, interpolation='nearest'))
    add_block_conv_inorm_relu(model, filters1, kernel_size)
    add_block_conv_inorm_relu(model, filters2, kernel_size)

def build_model(image_shape=(28,28), embedding_size=50, model_id=MODEL_ORIG):
    s = image_shape[-1]
    feat=Sequential()
    
    if model_id == MODEL_ORIG:
        feat.add(Conv2D(20,(3,3),
                activation='relu',padding='same',
                input_shape=(s, s, 1)))
        feat.add(Conv2D(5,(5,5),activation='relu',padding='same'))
        feat.add(Flatten())
        feat.add(Dense(100))
        feat.add(Dense(embedding_size))
    else:
        nf = 16                                      # input size: 1 x 512 x 512 (262 144)
        add_block_conv_inorm_relu(feat,  nf, (3,3), 
                                  input_shape=(s, s, 1))    # =>  16 x 512 x 512
        add_block_2conv_avgpool(feat,    nf, (3,3), (2,2))  # =>  16 x 256 x 256
        add_block_2conv_avgpool(feat,  nf*2, (3,3), (2,2))  # =>  32 x 128 x 128
        add_block_2conv_avgpool(feat,  nf*4, (3,3), (2,2))  # =>  64 x  64 x  64
        add_block_2conv_avgpool(feat,  nf*8, (3,3), (2,2))  # => 128 x  32 x  32
        add_block_2conv_avgpool(feat, nf*16, (3,3), (2,2))  # => 256 x  16 x  16
        
        add_block_conv_inorm_relu(feat, nf*16, (3,3))       # => 256 x  16 x  16
        add_block_conv_inorm_relu(feat, nf*16, (3,3))       # => 256 x  16 x  16
        feat.add(Flatten())                                 # => 65 536
        
    
    inp1=Input(shape=(s,s,1))
    inp2=Input(shape=(s,s,1))
    
    feat1=feat(inp1)
    feat2=feat(inp2)
    
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([feat1, feat2])

    feat.compile('sgd','mse')
    
    model=Model([inp1,inp2],distance)
    model.compile('adam','mse')
    
    unfeat=Sequential()
    input_dim = feat.get_output_shape_at(0)[-1]
    
    if model_id == MODEL_ORIG:
        unfeat.add(Dense(100, input_shape=(input_dim,), activation='relu'))
        unfeat.add(Dense(5*s*s, activation='relu'))
        unfeat.add(Reshape((s,s,5)))
        unfeat.add(Conv2D(10,(5,5),activation='relu', padding='same'))
        unfeat.add(Conv2D(1,(3,3),activation='linear', padding='same'))
    else:
        nf = 16                                              # input size: 65 536
        unfeat.add(Reshape((16,16,nf*16)))                            # => 256 x  16 x  16
        add_block_conv_inorm_relu(unfeat, nf*16, (3,3))               # => 256 x  16 x  16
        add_block_conv_inorm_relu(unfeat, nf*16, (3,3))               # => 256 x  16 x  16
        
        add_block_2conv_upsample(unfeat, nf*16, nf*8, (3,3), (2,2))   # => 128 x  32 x  32
        add_block_2conv_upsample(unfeat , nf*8, nf*4, (3,3), (2,2))   # =>  64 x  64 x  64
        add_block_2conv_upsample(unfeat,  nf*4, nf*2, (3,3), (2,2))   # =>  32 x 128 x 128
        add_block_2conv_upsample(unfeat,  nf*2,   nf, (3,3), (2,2))   # =>  16 x 256 x 256
        
        unfeat.add(UpSampling2D(size=(2,2), interpolation='nearest')) # =>  16 x 512 x 512
        add_block_conv_inorm_relu(unfeat, nf, (3,3))                  # =>  16 x 512 x 512
        add_block_conv_inorm_relu(unfeat,  1, (3,3))                  # =>   1 x 512 x 512
        
    unfeat.add(Flatten())
    unfeat.add(Activation('softmax')) # samples are probabilities
    unfeat.add(Reshape((s,s,1)))

    uf1=unfeat(feat1)
    uf2=unfeat(feat2)

    unfeat.compile('adam','kullback_leibler_divergence')
    
    model2=Model([inp1,inp2],[distance, uf1,uf2, uf1, uf2])
    model2.compile('adam',['mse', kullback_leibler_divergence_,kullback_leibler_divergence_,
                           sparsity_constraint, sparsity_constraint],
                           loss_weights=[1, 1e1, 1e1, 1e-3, 1e-3])
    
    return {'feat':feat, 'emd':model,'unfeat':unfeat,'dwe':model2}


def train_DWE(dataset_name=MNIST, repo=REPO, embedding_size=50, image_shape=(28,28),\
              batch_size=100, epochs=100):
    train, valid, test=get_data(dataset_name, repo)
    validation_data=([valid[0],valid[1]],[valid[2], valid[0], valid[1], valid[0], valid[1]])
    test_data=([test[0],test[1]],[test[2], test[0], test[1], test[0], test[1]])
    n_train=len(train[0])
    steps_per_epoch=int(n_train/batch_size)
    def myGenerator():
        #loading data
        while 1:
            for i in range(steps_per_epoch):
                index=range(i*batch_size, (i+1)*batch_size)
                x1,x2,y=(train[0][index], train[1][index], train[2][index])
                yield [x1, x2], [y, x1, x2, x1, x2]
    if image_shape == (28,28):
        model_id = MODEL_ORIG
    else:
        model_id = MODEL_512x512
        
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        dict_models=build_model(image_shape, embedding_size, model_id=model_id)
        model = dict_models['dwe']
    
    earlystop=EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    saveweights=ModelCheckpoint('{}/{}_autoencoder'.format(MODEL,dataset_name), 
                                monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    csv_logger = CSVLogger('dwe_training.log')
    
    model.fit(myGenerator(),steps_per_epoch=steps_per_epoch,
              epochs=epochs,validation_data=validation_data,
              callbacks=[earlystop, saveweights, csv_logger])

    model.evaluate(test_data[0], test_data[1])
    
    for key in dict_models:
        dict_models[key].save('{}/{}_{}.hd5'.format(MODEL, dataset_name, key))
    
    # explicitely close the pool
    atexit.register(strategy._extended._collective_ops._pool.close)

def train_DWE_randshapes(shapes_fpath, wdists_fpath, embedding_size=50, image_shape=(512,512), 
                         batch_size=100, epochs=100, n_data=100000, 
                         train_part=0.8, valid_part=0.1, test_part=0.1):
    n_train = round(n_data * train_part)
    n_valid = round(n_data * valid_part)
    n_test  = round(n_data * test_part)

    train_idx_interval = (0, n_train)
    valid_idx_interval = (n_train, n_train+n_valid)
    test_idx_interval  = (n_train+n_valid, n_train+n_valid+n_test)

    train_seq = RandShapesWDistSeq(shapes_fpath, wdists_fpath, train_idx_interval, batch_size=batch_size)
    valid_seq = RandShapesWDistSeq(shapes_fpath, wdists_fpath, valid_idx_interval, batch_size=batch_size)
    test_seq  = RandShapesWDistSeq(shapes_fpath, wdists_fpath, test_idx_interval,  batch_size=batch_size)
    
    if image_shape == (28,28):
        model_id = MODEL_ORIG
    else:
        model_id = MODEL_512x512
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        dict_models=build_model(image_shape, embedding_size, model_id=model_id)
        model = dict_models['dwe']
    
    earlystop=EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    saveweights=ModelCheckpoint('{}/{}_autoencoder'.format(MODEL,dataset_name), monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    csv_logger = CSVLogger('dwe_training.log')
    
    steps_per_epoch=int(n_train/batch_size)
    
    model.fit(train_seq,steps_per_epoch=steps_per_epoch,
              validation_data=valid_seq, epochs=epochs,
              callbacks=[earlystop, saveweights, csv_logger])

    model.evaluate(test_seq)
    
    for key in dict_models:
        dict_models[key].save('{}/{}_{}.hd5'.format(MODEL, dataset_name, key))
    
    # explicitely close the pool
    atexit.register(strategy._extended._collective_ops._pool.close)
    
#%%    
if __name__=="__main__":
    
    import tensorflow as tf
    import tensorflow.keras.backend as tfback
    print("tf.__version__ is", tf.__version__)
    print("tf.keras.__version__ is:", tf.keras.__version__)
    
    import argparse
    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--dataset_name', type=str, default='cat', 
                        help='dataset name')
    parser.add_argument('--repo', type=str, default=REPO, help='repository')
    parser.add_argument('--shapes_fpath', type=str, default=REPO, 
                        help='shapes repository (for dataset randshapes)')
    parser.add_argument('--wdists_fpath', type=str, default=REPO, 
                        help='wdist repository (for dataset randshapes)')
    parser.add_argument('--n_data', type=int, default=100000, 
                        help='#data (for dataset randshapes)')
    parser.add_argument('--embedding_size', type=int, default=50, 
                        help='embedding size')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    
    args = parser.parse_args()
    
    dataset_name=args.dataset_name
    repo=args.repo
    shapes_fpath=args.shapes_fpath
    wdists_fpath=args.wdists_fpath
    n_data=args.n_data
    embedding_size=args.embedding_size
    batch_size=args.batch_size
    epochs=args.epochs
    
    if dataset_name == RANDSHAPES:
        train_DWE_randshapes(shapes_fpath, wdists_fpath, embedding_size, 
                             batch_size=batch_size, epochs=epochs, n_data=n_data)
    else:
        train_DWE(dataset_name, repo, embedding_size, 
                  batch_size=batch_size, epochs=epochs)