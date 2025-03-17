from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
import copy
import logging
import math
import torch
import torch.nn as nn
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock


class DecoderBranch(nn.Module):
    def __init__(self,  
                feature_size, 
                out_channels):
        super(DecoderBranch, self).__init__()

        self.dec_0 = UnetrUpBlock(
            spatial_dims=3,
            in_channels= feature_size,
            out_channels= feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.dec_1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels= 2*feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.dec_2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=4* feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.dec_3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x_0, x_1, x_2, x_3, x_4):
        print(x_0.shape, x_1.shape)
        out_dec_3 = self.dec_3(x_0, x_1)
        out_dec_2 = self.dec_2(out_dec_3, x_2)
        out_dec_1 = self.dec_1(out_dec_2, x_3)
        out_dec_0 = self.dec_0(out_dec_1, x_4)
        out       = self.out(out_dec_0)
        return out