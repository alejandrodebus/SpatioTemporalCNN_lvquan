import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):

    def __init__(self):
        super(EncoderCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size = 5, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size = 5, stride = 1, padding = 1)

        self.pool1 = nn.MaxPool2d(kernel_size = 5, stride = 2, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size = 5, stride = 2, padding = 0)

        self.batch_norm1 = nn.BatchNorm2d(20)
        self.batch_norm2 = nn.BatchNorm2d(40)


    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        return x
