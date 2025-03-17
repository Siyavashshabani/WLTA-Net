import torch.nn as nn
import torch.nn.functional as F
import math
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)


class ConvEncoder(nn.Module):
    def __init__(self, c=1, n=48, dropout=0.5, norm='gn', num_classes=15):
        super(ConvEncoder, self).__init__()
        self.convd1 = ConvD(c,     n, dropout, norm, first=True)
        self.convd2 = ConvD(n,     n, dropout, norm)
        self.convd3 = ConvD(n,     2*n, dropout, norm)
        self.convd4 = ConvD(2*n, 4*n, dropout, norm)
        self.convd5 = ConvD(4*n,8*n, dropout, norm)

    def forward(self, x):
        x0 = self.convd1(x)
        x1 = self.convd2(x0)
        x2 = self.convd3(x1)    
        x3 = self.convd4(x2)
        x4 = self.convd5(x3)
        return x0,x1,x2,x3,x4

#import torch
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#cuda0 = torch.device('cuda:0')
#x = torch.rand((2, 4, 32, 32, 32), device=cuda0)
#model = Unet()
#model.cuda()
#y = model(x)
#print(y.shape)
