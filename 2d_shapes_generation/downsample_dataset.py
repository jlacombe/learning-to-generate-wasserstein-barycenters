import os
import sys
import h5py
import time
import numpy as np
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(os.getcwd()))
from io_util import analyse_args

def main():
    t1 = time.time()
    args = analyse_args([
        ['d',     'dataset_path', lambda x: str(x), os.path.join('.', 'input_shapes.h5')],
        ['n', 'new_dataset_path', lambda x: str(x), os.path.join('.', 'new_input_shapes.h5')],
        ['s',             'size', lambda x: [int(i) for i in x.split(',')], [28,28]],
        ['b',       'batch_size', lambda x: int(x), 100]
    ])
    dsname, ext = args['new_dataset_path'].split('/')[-1].split('.')
    
    if (ext not in ['h5', 'npy']):
        print('Unhandled datatype {}'.format(ext))
        sys.exit(-1)

    with h5py.File(args['dataset_path'], 'r') as f_in:
        inputs_dset = f_in['input_shapes']
        n_images = inputs_dset.shape[0]
        
        if (ext == 'h5'):
            f_out = h5py.File(args['new_dataset_path'], 'w')
            inputs_28x28_dset = f_out.create_dataset('input_shapes', (n_images,*args['size']), 
                                                     compression='gzip', shuffle=True, dtype=np.uint8, 
                                                     chunks=(1,*args['size']))
        
        print('Downsampling images...')
        new_imgs = []
        for i in range(n_images):#args['batch_size']):
            # resize image to the given shape
            new_img = resize(inputs_dset[i].astype(np.float64)/255., tuple(args['size']))

            # each image is normalized to have its min value = 0 and its max value = 255
            new_img = (new_img/new_img.max()) * 255.
            new_img = new_img.astype(np.uint8)

            new_imgs.append(new_img)
            if ((i+1) % (n_images//20) == 0):
                print('{}/{}'.format(i+1,n_images))
        new_imgs = np.stack(new_imgs)
            
        if (ext == 'h5'):
            inputs_28x28_dset[:] = new_imgs
            f_out.close()
        elif (ext == 'npy'):
            np.save(args['new_dataset_path'], new_imgs.reshape((n_images, -1)))
    
    print('Done.')
    deltat = time.time() - t1
    print('Elapsed Time: {:.2f}s to downsample {} images'.format(deltat, n_images))
    with open('{}_downsample_time.txt'.format(dsname), 'w') as f:
        f.write('Elapsed Time: {:.2f}s to downsample {} images'.format(deltat, n_images))
    
if __name__ == "__main__":
    main()
