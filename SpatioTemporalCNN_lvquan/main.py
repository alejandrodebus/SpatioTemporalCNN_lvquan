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
from torchvision.utils import save_image
import os

from dataset import LeftVentricleDataset
from utils import k_folds
from utils import save_data
from model_cnn3d import CNN3D
from model_phase_classification import PhaseClassification

experiment = 1
directory = './m_spa_tem_e{}/'.format(experiment)

if not os.path.exists(directory):
    os.makedirs(directory)

loss_train = []
loss_test = []
loss_train_fold = []
loss_test_fold = []

def train(model_1, model_2, optimizer_1, optimizer_2, epoch, device, train_loader, log_interval):
    model_1.train()
    model_2.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        img = data['images_lv']
        img = img.type(torch.FloatTensor).to(device)

        img_0 = img[:, 0, :, :]
        img_0 = torch.unsqueeze(img_0, 1)
        img_1 = img[:, 1, :, :]
        img_1 = torch.unsqueeze(img_1, 1)
        img_2 = img[:, 2, :, :]
        img_2 = torch.unsqueeze(img_2, 1)
        img_3 = img[:, 3, :, :]
        img_3 = torch.unsqueeze(img_3, 1)
        img_4 = img[:, 4, :, :]
        img_4 = torch.unsqueeze(img_4, 1)

        rwt = data['rwt_lv']
        rwt = rwt.type(torch.FloatTensor).to(device)

        dims = data['dims_lv']
        dims = dims.type(torch.FloatTensor).to(device)

        areas = data['areas_lv']
        areas = areas.type(torch.FloatTensor).to(device)

        phase = data['phase_lv']
        phase = phase.type(torch.FloatTensor).to(device)

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        # Forward
        output_1, in_model2 = model_1(img_0, img_1, img_2, img_3, img_4)
        loss_1 = F.mse_loss(output_1[:,0:6,:,:,:], rwt) + F.mse_loss(output_1[:,6:9,:,:,:], dims) + F.mse_loss(output_1[:,9:11,:,:,:], areas)

        output_2 = model_2(in_model2)
        loss_2 = loss_1 + F.cross_entropy(output_2, phase)

        # Backward
        #loss_1.backward() # No es necesario
        loss_2.backward()
        optimizer_1.step()
        optimizer_2.step()

        train_loss += loss_2.item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_2.item()))

    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss {:.6f}'.format(train_loss))

def test(model_1, model_2, device, test_loader):
    model_1.eval()
    model_2.eval()
    test_loss_1 = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img = data['images_lv']
            img = img.type(torch.FloatTensor).to(device)

            img_0 = img[:, 0, :, :]
            img_0 = torch.unsqueeze(img_0, 1)
            img_1 = img[:, 1, :, :]
            img_1 = torch.unsqueeze(img_1, 1)
            img_2 = img[:, 2, :, :]
            img_2 = torch.unsqueeze(img_2, 1)
            img_3 = img[:, 3, :, :]
            img_3 = torch.unsqueeze(img_3, 1)
            img_4 = img[:, 4, :, :]
            img_4 = torch.unsqueeze(img_4, 1)

            rwt = data['rwt_lv']
            rwt = rwt.type(torch.FloatTensor).to(device)

            dims = data['dims_lv']
            dims = dims.type(torch.FloatTensor).to(device)

            areas = data['areas_lv']
            areas = areas.type(torch.FloatTensor).to(device)

            phase = data['phase_lv']
            phase = phase.type(torch.FloatTensor).to(device)

            output_1, in_model2 = model_1(img_0, img_1, img_2, img_3, img_4)
            output_2 = model_2(in_model2)

            test_loss_1 = F.mse_loss(output_1[:,0:6,:,:,:], rwt, size_average=False).item() + F.mse_loss(output_1[:,6:9,:,:,:], dims, size_average=False).item() + F.mse_loss(output_1[:,9:11,:,:,:], areas, size_average=False).item()
            test_loss = test_loss_1 + F.cross_entropy(output_2, phase)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss))
    loss_test.append(test_loss)

def main():

    torch.manual_seed(1)

    global loss_train
    global loss_test
    global loss_train_fold
    global loss_test_fold

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    learning_rate = 0.0001
    num_epochs = 700
    model_1 = Sandwich().to(device)
    optimizer_1 = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.5, weight_decay = 0.005)

    model_2 = PhaseClassification().to(device)
    optimizer_2 = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.6, weight_decay = 0.005)

    log_interval = 80

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    batch_size_train = 20
    batch_size_test = 20

    num_folds = 5

    cont_fold = 1
    for train_idx, test_idx in k_folds(n_splits = num_folds):
        print('###################### FOLD {} ######################'.format(cont_fold))
        dataset_lv_train = LeftVentricleDataset(indices = train_idx, k_slices = 5)
        dataset_lv_test = LeftVentricleDataset(indices = test_idx, k_slices = 5)
        train_loader = torch.utils.data.DataLoader(dataset = dataset_lv_train, batch_size = batch_size_train, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset = dataset_lv_test, batch_size = batch_size_test, **kwargs)
        loss_train = []
        loss_test = []
        for epoch in range(1, num_epochs + 1):
            train(model_1, model_2 optimizer_1, optimizer_2, epoch, device, train_loader, log_interval)
            test(model_1, model_2, device, test_loader)
        cont_fold+=1
        loss_train_fold.append(loss_train)
        loss_test_fold.append(loss_test)

    print('Guardando datos...')
    torch.save(model, './m_spa_tem_e{}/model_trained.pt'.format(experiment))
    save_data(loss_train_fold, './m_spa_tem_e{}/loss_train.pickle'.format(experiment))
    save_data(loss_test_fold, './m_spa_tem_e{}/loss_test.pickle'.format(experiment))



if __name__ == '__main__':
    main()
