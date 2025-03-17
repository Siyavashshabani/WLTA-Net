from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
# import configs as configs
from torch.distributions.normal import Normal
logger = logging.getLogger(__name__)
import ml_collections


from embedding import EmbeddingBranch

def test_transformer():
    # Configuration for sizes

    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8, 8)})
    config.patches.grid = (8, 8, 8)
    config.hidden_size = 256
    config.transformer = ml_collections.ConfigDict()
    config.mlp_dim = 3072
    config.transformer.num_heads = 8
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.patch_size = 8
    config.n_channels = 48
    config.encoder_channels = 32
    config.down_factor = 2
    config.down_num = 2
    # config.decoder_channels = (96, 48, 32, 32, 16)
    # config.skip_channels = (32, 32, 32, 32, 16)
    # config.n_dims = 3
    # config.n_skip = 5
    config.target_size = 4




    ############ Run the 3D Transoforemr 
    n_patches = 64
    input_dim_0 = 96  # Assuming 'img_size' represents a feature size in your embedding
    n_channels_0 = 48
    config.input_dim_3d = input_dim_0
    input_0 = torch.randn(2, n_channels_0, input_dim_0, input_dim_0, input_dim_0)   # Batch size of 10
    print("input_0 shape-------------------", input_0.shape)



    input_dim_1 = 48  # Assuming 'img_size' represents a feature size in your embedding
    n_channels_1 = 48
    config.input_dim_3d = input_dim_1
    input_1 = torch.randn(2, n_channels_1, input_dim_1, input_dim_1, input_dim_1)   # Batch size of 10
    print("input_1 shape-------------------", input_1.shape)
    
    input_dim_2 = 24  # Assuming 'img_size' represents a feature size in your embedding
    n_channels_2 = 96
    config.input_dim_3d = input_dim_2
    input_2 = torch.randn(2, n_channels_2, input_dim_2, input_dim_2,input_dim_2)   # Batch size of 10
    print("input_2 shape-------------------", input_2.shape)
    
    input_dim_3 = 12  # Assuming 'img_size' represents a feature size in your embedding
    n_channels_3 = 192
    config.input_dim_3d = input_dim_3
    input_3 = torch.randn(2, n_channels_3, input_dim_3, input_dim_3, input_dim_3)   # Batch size of 10
    print("input_3 shape-------------------", input_3.shape)
    
    input_dim_4 = 6  # Assuming 'img_size' represents a feature size in your embedding
    n_channels_4 = 384
    config.input_dim_3d = input_dim_4
    input_4 = torch.randn(2, n_channels_4, input_dim_4, input_dim_4,input_dim_4)   # Batch size of 10
    print("input_4 shape-------------------", input_4.shape)
    

    
    
    # Create the CrossViT instance
    target_dim = input_dim_0
    embedding_branch = EmbeddingBranch(config, 
                                       input_dim_0, n_channels_0, 
                                       input_dim_1, n_channels_1, 
                                       input_dim_2, n_channels_2,
                                       input_dim_3, n_channels_3,
                                       input_dim_4, n_channels_4,
                                       target_dim, n_patches)

    # Pass the inputs through the model
    embeded_0, embeded_1, embeded_2, embeded_3, embeded_4 = embedding_branch(input_0, input_1, input_2, input_3, input_4)
    
    print("Output shape of crossVit", embeded_1.shape)
    # Count parameters for embedding_0
    embedding_0_params = sum(p.numel() for p in embedding_branch.embeddings_0.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in embedding_0: {embedding_0_params}")

    # Count parameters for embedding_1
    embedding_1_params = sum(p.numel() for p in embedding_branch.embeddings_1.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in embedding_1: {embedding_1_params}")

    # Count parameters for embedding_2
    embedding_2_params = sum(p.numel() for p in embedding_branch.embeddings_2.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in embedding_2: {embedding_2_params}")

    # Count parameters for embedding_3
    embedding_3_params = sum(p.numel() for p in embedding_branch.embeddings_3.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in embedding_3: {embedding_3_params}")

    # Count parameters for embedding_4
    embedding_4_params = sum(p.numel() for p in embedding_branch.embeddings_4.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in embedding_4: {embedding_4_params}")



if __name__ == "__main__":
    test_transformer()
