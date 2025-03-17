import torch
import os
from UNet.unet import ConvEncoder
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda0 = torch.device('cuda:0')
x = torch.rand((2, 1, 96, 96, 96), device=cuda0)
model = ConvEncoder()
model.cuda()
y = model(x)
print(y.shape)
