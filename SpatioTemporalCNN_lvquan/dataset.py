import scipy.io as sio
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LeftVentricleDataset(Dataset):

    def __init__(self, indices, k_slices = 3, transform = None):

        self.indices = indices
        self.k_slices = k_slices # slices size
        self.transform = transform

        data = sio.loadmat('/home/adebus/storage/cardiac-dig.mat')
        images_LV = data['images_LV']
        endo_LV = data['endo_LV']
        epi_LV = data['epi_LV']
        lv_phase = data['lv_phase']
        rwt = data['rwt']
        areas = data['areas']
        dims = data['dims']

        # Number of patients
        subjects = 145

        # 20 is frames number

        n = self.k_slices

        # Shape: (Total images, depth, H, W)
        imgs = np.zeros([(subjects * 20),n,80,80], dtype='f8')

        for i in range(subjects):
            v = images_LV[:, :, 20*i:20*i+20]
            v_i = np.zeros(20 + ((n//2)*2), dtype=int)
            v_i[0:20+(n//2)] = np.arange(-(n//2),20, dtype=int)
            v_i[20+(n//2):] = np.arange(0,(n//2), dtype=int)
            for j in range(20):
                for k in range(n):
                    imgs[j+20*i, k, :, :] = v[:, :, v_i[j+k]]

        # 1 - MRI Left Ventricle
        self.images_LV = torch.from_numpy(imgs[indices, :, :, :])

        # 2 - Cardiac phase
        self.lv_phase = torch.from_numpy(lv_phase.flatten('C'))
        self.lv_phase = self.lv_phase[self.indices]

        # 3 - RWT (6 indices)
        self.rwt = torch.from_numpy(rwt)
        self.rwt = self.rwt[:, self.indices]

        # 4 - Areas (cavity and myocardium)
        self.areas = torch.from_numpy(areas)
        self.areas = self.areas[:, self.indices]

        # 5 - Dims
        self.dims = torch.from_numpy(dims)
        self.dims = self.dims[:, self.indices]

        # 6 - Endocardium
        # Shape: (Total images, depth, H, W)
        imgs_endo = np.zeros([(subjects * 20),n,80,80], dtype=int)
        for i in range(subjects):
            v = endo_LV[:, :, 20*i:20*i+20]
            v_i = np.zeros(20 + ((n//2)*2), dtype=int)
            v_i[0:20+(n//2)] = np.arange(-(n//2),20, dtype=int)
            v_i[20+(n//2):] = np.arange(0,(n//2), dtype=int)
            for j in range(20):
                for k in range(n):
                    imgs_endo[j+20*i, k, :, :] = v[:, :, v_i[j+k]]

        self.endo_LV = torch.from_numpy(imgs_endo[indices, :, :, :])

        # 7 - Epicardium
        # Shape: (Total images, depth, H, W)
        imgs_epi = np.zeros([(subjects * 20),n,80,80], dtype=int)
        for i in range(subjects):
            v = epi_LV[:, :, 20*i:20*i+20]
            v_i = np.zeros(20 + ((n//2)*2), dtype=int)
            v_i[0:20+(n//2)] = np.arange(-(n//2),20, dtype=int)
            v_i[20+(n//2):] = np.arange(0,(n//2), dtype=int)
            for j in range(20):
                for k in range(n):
                    imgs_epi[j+20*i, k, :, :] = v[:, :, v_i[j+k]]

        self.epi_LV = torch.from_numpy(imgs_epi[indices, :, :, :])


    def __getitem__(self, index):

        image_lv = self.images_LV[index, :, :, :]

        phase_lv = self.lv_phase[index]

        rwt = self.rwt[:, index]

        areas = self.areas[:, index]

        dims = self.dims[:, index]

        endo_lv = self.endo_LV[index, :, :, :]

        epi_lv = self.epi_LV[index, :, :, :]

        sample = {'images_lv': image_lv, 'phase_lv': phase_lv,
                  'rwt_lv': rwt, 'areas_lv': areas, 'dims_lv': dims, 'endo_lv': endo_lv, 'epi_lv': epi_lv}

        return sample

    def __len__(self):
        return self.images_LV.__len__()
