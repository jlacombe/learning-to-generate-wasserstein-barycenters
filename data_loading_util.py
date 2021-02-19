import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

class GeomlossBary(Dataset):
    def __init__(self, shapes_folder_path, barys_folder_path, idx_interval):
        super(GeomlossBary, self).__init__()
        
        self.shapes_folder_path = shapes_folder_path
        self.barys_folder_path = barys_folder_path
        self.idx_interval = idx_interval
        
        self.input_shapes_file = h5py.File(shapes_folder_path, 'r')
        self.barys_file = h5py.File(barys_folder_path, 'r')
        
        self.inputs_dset = self.input_shapes_file['input_shapes']
        self.barys_dset    = self.barys_file['barycenters']
        self.ids_dset      = self.barys_file['input_ids']
        self.bweights_dset = self.barys_file['bweights']
        
        self.n_barys = idx_interval[1] - idx_interval[0]
        self.eps = 1e-10
        
    def __getitem__(self, index):
        index = self.idx_interval[0] + index
        bary = self.barys_dset[index].astype(np.float)
        bweights = torch.FloatTensor(self.bweights_dset[index])
        bweights = bweights.reshape(-1,1,1)
        input_ids = self.ids_dset[index]
        inputs = [self.inputs_dset[input_id].astype(np.float) for input_id in input_ids]
        
        bary = torch.FloatTensor(bary / bary.sum()).unsqueeze(0)
        inputs = torch.stack([torch.FloatTensor(x / x.sum()) for x in inputs])
        
        return inputs, bary, bweights

    def __len__(self):
        return self.n_barys

def load_geomloss_barys(shapes_folder_path, barys_folder_path, train_part, 
                           valid_part, test_part, n_data, batch_size):
    with h5py.File(barys_folder_path, 'r') as f:
        barys_dset = f['barycenters']
        n_total_barys = barys_dset.shape[0]
    n_train = round(n_data * train_part)
    n_valid = round(n_data * valid_part)
    n_test  = round(n_data * test_part)
    
    train_idx_interval = (0, n_train)
    valid_idx_interval = (n_train, n_train+n_valid)
    test_idx_interval  = (n_train+n_valid, n_train+n_valid+n_test)
    
    train_dataset = GeomlossBary(shapes_folder_path, barys_folder_path, train_idx_interval)
    valid_dataset = GeomlossBary(shapes_folder_path, barys_folder_path, valid_idx_interval)
    test_dataset  = GeomlossBary(shapes_folder_path, barys_folder_path, test_idx_interval)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,  num_workers=1)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=1)
    
    return {'train_loader': train_loader,
            'valid_loader': valid_loader,
             'test_loader': test_loader}
