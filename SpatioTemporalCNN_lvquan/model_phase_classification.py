import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseClassification(nn.Module):

    def __init__(self):
        super(PhaseClassification, self).__init__()

        self.conv3d_5 = nn.Conv3d(40, 40, kernel_size = (1,5,5), stride = 1, padding = (0,1,1))

        self.batch_norm5 = nn.BatchNorm3d(40)

        self.fc6 = nn.Linear(640,640)
        self.fc7 = nn.Linear(640,2)

    def forward(self, x):

        x = self.conv3d_5(x)
        x = self.batch_norm5(x)
        x = F.relu(x)

        x = out.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return x
