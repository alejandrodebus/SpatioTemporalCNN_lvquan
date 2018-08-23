import torch
import torch.nn as nn
import torch.nn.functional as F
from model_encoder_cnn import EncoderCNN

class CNN3D(nn.Module):

    def __init__(self):
        super(CNN3D, self).__init__()

        self.cnn0 = EncoderCNN()
        self.cnn1 = EncoderCNN()
        self.cnn2 = EncoderCNN()
        self.cnn3 = EncoderCNN()
        self.cnn4 = EncoderCNN()

        self.conv3d_3 = nn.Conv3d(40, 200, kernel_size = (3,5,5), stride = 1, padding = (0,1,1))
        self.conv3d_4 = nn.Conv3d(200, 200, kernel_size = (2,5,5), stride = 1, padding = (0,1,1))

        self.pool3 = nn.MaxPool3d(kernel_size = (1,2,2), stride = 2, padding = (0,1,1))

        self.batch_norm3 = nn.BatchNorm3d(200)
        self.batch_norm4 = nn.BatchNorm3d(200)

        self.conv_reg5 = nn.Conv3d(200, 200, kernel_size = (3, 3, 3), stride = 1, padding = 1)
        self.conv_reg6 = nn.Conv3d(200, 11, kernel_size = (1, 6, 6), stride = 1, padding = 0)

    def forward(self, v, w, x, y, z):

        v = self.cnn0(v) # (t-2)
        w = self.cnn1(w) # (t-1)
        x = self.cnn2(x) # (t)
        y = self.cnn3(y) # (t+1)
        z = self.cnn4(z) # (t+2)

        out = torch.stack([v,w,x,y,z],2)

        out = self.conv3d_3(out)
        out = self.batch_norm3(out)
        out = F.relu(out)
        out = self.pool3(out)

        out = self.conv3d_4(out)
        out = self.batch_norm4(out)
        out = F.relu(out)

        out_int = out

        out = self.conv_reg5(out)
        out = self.conv_reg6(out)

        out = out.view(out.size(0), -1)

        return out, out_int
