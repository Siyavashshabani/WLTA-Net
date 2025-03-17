import torch
import torch.nn as nn
from embeddings2D import Embeddings2D
from embeddings3D import Embeddings24_48_96
from embedding3DHybrid import Embeddings3D

import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
import copy
import logging
import math
import torch





# # Define the Transformer class
# class EmbeddingBranch(nn.Module):
#     def __init__(self, config, 
#                 input_dim_0, n_channels_0, 
#                 input_dim_1, n_channels_1, 
#                 input_dim_2, n_channels_2,
#                 input_dim_3, n_channels_3,
#                 input_dim_4, n_channels_4,
#                 target_dim, n_patches ):

#         super(EmbeddingBranch, self).__init__()
#         print()

#         self.embeddings_0 = Embeddings3D(config,input_dim_0, n_channels_0, n_patches)
#         self.embeddings_1 = Embeddings3D(config,input_dim_1, n_channels_1, n_patches)
#         self.embeddings_2 = Embeddings3D(config,input_dim_2, n_channels_2, n_patches)
#         self.embeddings_3 = Embeddings3D(config,input_dim_3, n_channels_3, n_patches, upsample=None )
#         self.embeddings_4 = Embeddings3D(config, input_dim_4, n_channels_4, n_patches, upsample=True )

#     def forward(self, input_0, input_1, input_2, input_3, input_4):
#         # print("Start the CrossViT")
#         # Obtain the embeddings and features
#         # embedding_2d = self.embeddings2d(input_2d)
#         # print("embedding_2d:", embedding_2d.shape)

#         # Obtain the embeddings and features

#         embedding_0 = self.embeddings_0(input_0)
#         print("embedding_0:", embedding_0.shape)   

#         embedding_1 = self.embeddings_1(input_1)
#         print("embedding_1:", embedding_1.shape)   

#         embedding_2 = self.embeddings_2(input_2)
#         print("embedding_2:", embedding_2.shape)  

#         embedding_3 = self.embeddings_3(input_3)
#         print("embedding_3:", embedding_3.shape)   

#         embedding_4 = self.embeddings_4(input_4)
#         print("embedding_4:", embedding_4.shape)  



#         # print("reshaped_encoded-----------------------", reshaped_encoded.shape)
#         # Pass through the reconstruction block to obtain the desired shape
#         # reconstructed_output = self.reconstruction_block(reshaped_encoded)

#         return embedding_0, embedding_1, embedding_2, embedding_3, embedding_4
    
    



# Define the Transformer class
class EmbeddingHybridBranch(nn.Module):
    def __init__(self, config, 
                input_dim_0, n_channels_0, 
                input_dim_1, n_channels_1, 
                input_dim_2, n_channels_2,
                input_dim_3, n_channels_3,
                input_dim_4, n_channels_4,
                n_patches ):

        super(EmbeddingHybridBranch, self).__init__()
        print()
        self.embeddings_0 = Embeddings3D(config, input_dim_0, n_channels_0, n_patches)
        self.embeddings_1 = Embeddings3D(config, input_dim_1, n_channels_1, n_patches)
        self.embeddings_2 = Embeddings3D(config, input_dim_2, n_channels_2, n_patches)
        self.embeddings_3 = Embeddings3D(config, input_dim_3, n_channels_3, n_patches)
        self.embeddings_4 = Embeddings3D(config, input_dim_4, n_channels_4, n_patches)

    def forward(self, input_0, input_1, input_2, input_3, input_4):

        embedding_0 = self.embeddings_0(input_0)
        # print("embedding_0:", embedding_0.shape)   

        embedding_1 = self.embeddings_1(input_1)
        # print("embedding_1:", embedding_1.shape)   

        embedding_2 = self.embeddings_2(input_2)
        # print("embedding_2:", embedding_2.shape)  

        embedding_3 = self.embeddings_3(input_3)
        # print("embedding_3:", embedding_3.shape)   

        embedding_4 = self.embeddings_4(input_4)
        # print("embedding_4:", embedding_4.shape)  

        return embedding_0, embedding_1, embedding_2, embedding_3, embedding_4
    
    




