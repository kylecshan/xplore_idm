import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import h5py
from osgeo import gdal

def load_file(folder, file):
    ds = gdal.Open(os.path.join(folder, file))
    return torch.Tensor(ds.ReadAsArray())

def split_image(x):
    return x[:-1, :, :], x[-1, :, :]

def random_crop(x, K):
    hstart = np.random.randint(x.shape[1]-K)
    wstart = np.random.randint(x.shape[2]-K)
    return x[:, hstart:(hstart+K), wstart:wstart+K]

def random_rot_flip(x):
    output = x
    if np.random.rand() < 0.5: # vertical flip
        output = torch.flip(output, [1])
    if np.random.rand() < 0.5: # horizontal flip
        output = torch.flip(output, [2])
    if np.random.rand() < 0.5: # transpose
        output = torch.transpose(output, 1, 2)
    return output

# # A dataset which simply loads the .tif files
# class FullImageDataset_DEPRECATED(torch.utils.data.Dataset):
#     def __init__(self, folder, dhsgps_file, normalize=True, img_prefix='image_'):
#         self.folder = folder
#         if normalize:
#             means = torch.Tensor([10.4416, 15.2296, 26.3324, 
#                                   30.9281, 73.2675, 70.4694, 
#                                   49.4693, 70.2683, 62.4014])
#             stds = torch.Tensor([6.0010,  7.3479, 10.5185,
#                                  17.6855, 16.4320, 28.8515,
#                                  27.2334,  8.5265, 8.6082])
#             self.norm = transforms.Normalize(means, stds)
#         else:
#             self.norm = None
            
#         if dhsgps_file is not None:
#             self.healthcare = torch.Tensor(pd.read_csv(dhsgps_file).to_numpy())
#             self.img_prefix = img_prefix
#         else:
#             self.healthcare = None
#             self.data_files = os.listdir(folder)
#             self.data_files.sort()

#     def __getitem__(self, idx):
#         if self.healthcare is None:
#             img_file = self.data_files[idx]
#             vacc_stats = None
#         else:
#             cluster_id = int(self.healthcare[idx, 0])
#             img_file = self.img_prefix + str(cluster_id).rjust(3, '0') + '.tif'
#             vacc_stats = self.healthcare[idx, :]
#         output = load_file(self.folder, img_file)
#         if self.norm is not None:
#             output[:9, :, :] = self.norm(output[:9, :, :])
#         output[torch.isnan(output)] = 0
#         return output, vacc_stats

#     def __len__(self):
#         if self.healthcare is None:
#             return len(self.data_files)
#         else:
#             return self.healthcare.shape[0]

# A dataset which loads the hdf5 data
class FullImageDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, dhsgps_file, normalize=True):
        self.h5 = h5py.File(h5_file, 'r')
        if normalize:
            means = torch.Tensor([10.1707, 14.8429, 25.7929, 
                                  30.0618, 73.5301, 69.8245, 
                                  48.3126, 69.7498, 62.0024])
            stds = torch.Tensor([6.0652,  7.4589, 10.7411, 
                                 18.1305, 17.0735, 29.9055, 
                                 28.0655,  8.7038, 8.7830])
            self.norm = transforms.Normalize(means, stds)
        else:
            self.norm = None
            
        self.healthcare = pd.read_csv(dhsgps_file).to_numpy()

    def __getitem__(self, idx):
        vacc_stats = torch.Tensor(self.healthcare[idx, :])
        assert(self.h5['cluster_id'][idx] == vacc_stats[0])
        
        output = torch.Tensor(self.h5['satellite'][idx])
        if self.norm is not None:
            output[:9, :, :] = self.norm(output[:9, :, :])
        output[torch.isnan(output)] = 0
        return output, vacc_stats

    def __len__(self):
        if self.healthcare is None:
            return len(self.data_files)
        else:
            return self.healthcare.shape[0]

# This dataset center-crops the 25km square images to
# 10km square images (K = 333).
class TestDataset(FullImageDataset):
    def __init__(self, h5_file, dhsgps_file, K=333, normalize=True):
        super(TestDataset, self).__init__(h5_file, dhsgps_file, normalize=normalize)
        self.K = K
        
    def __getitem__(self, idx):
        full, vacc_stats = super(TestDataset, self).__getitem__(idx)
        hstart = (full.shape[1]-self.K)//2
        wstart = (full.shape[2]-self.K)//2
        output = full[:, hstart:(hstart+self.K), wstart:wstart+self.K]
        landsat, light = split_image(output)
        light = ((light > 4).long() + (light > 16).long()).median()
        return landsat, light, vacc_stats
    
# # This dataset extends TestDataset to also include healthcare
# # outcomes.
# class HealthcareDataset(TestDataset):
#     def __init__(self, folder, dhsgps_file, K=333):
#         super(HealthcareDataset, self).__init__(folder, K)
#         self.healthcare = pd.read_csv(dhsgps_file).to_numpy()
        
#     def __getitem__(self, idx):
#         landsat, light = super(HealthcareDataset, self).__getitem__(idx)
#         healthcare = torch.Tensor(self.healthcare[idx, 1:12])
#         return landsat, light, healthcare
    
    
# This dataset random-crops the 25km square images to
# 10km square images (K = 333) and randomly applies
# rotation or a flip.
class TrainDataset(FullImageDataset):
    def __init__(self, h5_file, dhsgps_file, K=333):
        super(TrainDataset, self).__init__(h5_file, dhsgps_file)
        self.K = K
        
    def __getitem__(self, idx):
        full, vacc_stats = super(TrainDataset, self).__getitem__(idx)
        output = random_crop(full, self.K)
        output = random_rot_flip(output)
        landsat, light = split_image(output)
        light = ((light > 4).long() + (light > 16).long()).median()
        return landsat, light, vacc_stats