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
import pickle
import os

from model_encoder_cnn import EncoderCNN

experiment = 1
directory = './m_spa_tem_pha_e{}'.format(experiment)

if not os.path.exists(directory):
    os.makedirs(directory)

loss_train = []
loss_test = []
loss_train_fold = []
loss_test_fold = []
lista_accuracy = []
lista_accuracy_fold = []

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")



def train(model, optimizer, epoch, device, train_loader, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        img = data['images_lv']
        #img = torch.unsqueeze(img, 1) #
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

        target = data['phase_lv']
        target = target.type(torch.LongTensor).to(device)

        optimizer.zero_grad()

        # Forward
        output = model(img_0, img_1, img_2, img_3, img_4)
        loss = F.cross_entropy(output, target)

        train_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss

        # Backward
        loss.backward()
        optimizer.step()


        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss {:.4f}'.format(train_loss))
    loss_train.append(train_loss)

def test(model, device, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img = data['images_lv']
            #img = torch.unsqueeze(img, 1) #
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

            target = data['phase_lv']
            target = target.type(torch.LongTensor).to(device)

            output = model(img_0, img_1, img_2, img_3, img_4)

            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    lista_accuracy.append(100. * correct / len(test_loader.dataset))
    loss_test.append(test_loss)

def main():

    torch.manual_seed(1)

    global loss_train
    global loss_test
    global loss_train_fold
    global loss_test_fold
    global lista_accuracy
    global lista_accuracy_fold

    batch_size_train = 2
    batch_size_test = 2

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    learning_rate = 0.0003
    num_epochs = 100
    model = Sandwich().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 0.005)

    log_interval = 40

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
        lista_accuracy = []
        for epoch in range(1, num_epochs + 1):
            train(model, optimizer, epoch, device, train_loader, log_interval)
            test(model, device, test_loader)
        cont_fold+=1
        loss_train_fold.append(loss_train)
        loss_test_fold.append(loss_test)
        lista_accuracy_fold.append(lista_accuracy)

    print('Guardando datos...')
    torch.save(model, './m_spa_tem_pha_e{}/model_trained.pt'.format(experiment))
    save_data(loss_train_fold, './m_spa_tem_pha_e{}/loss_train.pickle'.format(experiment))
    save_data(loss_test_fold, './m_spa_tem_pha_e{}/loss_test.pickle'.format(experiment))
    save_data(lista_accuracy_fold, './m_spa_tem_pha_e{}/lista_accuracy.pickle'.format(experiment))


if __name__ == '__main__':
    main()
