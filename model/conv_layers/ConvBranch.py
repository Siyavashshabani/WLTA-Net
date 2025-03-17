from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
import copy
import logging
import math
import torch
import torch.nn as nn
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock


class ConvBranch(nn.Module):
    def __init__(self,  
                        feature_size, 
                        n_channels_0,
                        n_channels_1,
                        n_channels_2,
                        n_channels_3,
                        n_channels_4):
        super(ConvBranch, self).__init__()


        # print(  feature_size, 
        #         n_channels_0,
        #         n_channels_1,
        #         n_channels_2,
        #         n_channels_3,
        #         n_channels_4)
        
        self.conv_0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels =n_channels_0,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        
        self.conv_1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels =n_channels_1,
            out_channels=n_channels_1,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.conv_2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels =n_channels_2,
            out_channels=n_channels_2,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        
        self.conv_3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels =n_channels_3,
            out_channels=n_channels_3,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.conv_4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels =n_channels_4,
            out_channels=n_channels_4,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

    def forward(self, x_0, x_1, x_2, x_3, x_4):     
        print(x_0.shape, x_1.shape, x_2.shape, x_3.shape, x_4.shape)   
        out_0 =self.conv_0(x_0)
        out_1 =self.conv_1(x_1)
        out_2 =self.conv_2(x_2)
        out_3 =self.conv_3(x_3)
        out_4 =self.conv_4(x_4)        
        return  out_0, out_1, out_2, out_3, out_4       