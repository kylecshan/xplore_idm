import torch
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
import pandas as pd
import os
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

# A dataset which simply loads the .tif files
class FullImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.data_files = os.listdir(folder)
        self.data_files.sort()

    def __getitem__(self, idx):
        return load_file(self.folder, self.data_files[idx])

    def __len__(self):
        return len(self.data_files)

# This dataset center-crops the 25km square images to
# 10km square images (K = 333).
class TestDataset(FullImageDataset):
    def __init__(self, folder, K=333):
        super(TestDataset, self).__init__(folder)
        self.K = K
        
    def __getitem__(self, idx):
        full = super(TestDataset, self).__getitem__(idx)
        hstart = (full.shape[1]-self.K)//2
        wstart = (full.shape[2]-self.K)//2
        output = full[:, hstart:(hstart+self.K), wstart:wstart+self.K]
        landsat, light = split_image(output)
        light = ((light > 2).long() + (light > 34).long()).median()
        return landsat, light
    
# This dataset extends TestDataset to also include healthcare
# outcomes.
class HealthcareDataset(TestDataset):
    def __init__(self, folder, dhsgps_file, K=333):
        super(HealthcareDataset, self).__init__(folder, K)
        self.healthcare = pd.read_csv(dhsgps_file).to_numpy()
        
    def __getitem__(self, idx):
        landsat, light = super(HealthcareDataset, self).__getitem__(idx)
        healthcare = torch.Tensor(self.healthcare[idx, 1:12])
        return landsat, light, healthcare
    
    
# This dataset random-crops the 25km square images to
# 10km square images (K = 333) and randomly applies
# rotation or a flip.
class TrainDataset(FullImageDataset):
    def __init__(self, folder, K=333):
        super(TrainDataset, self).__init__(folder)
        self.K = K
        
    def __getitem__(self, idx):
        full = super(TrainDataset, self).__getitem__(idx)
        output = random_crop(full, self.K)
        output = random_rot_flip(output)
        landsat, light = split_image(output)
        light = ((light > 4).long() + (light > 20).long()).median()
        return landsat, light