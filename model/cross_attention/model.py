from cross_attention.cross_attention import CrossAttentionBranch, CrossAttentionOutBranch
from SwinViT.SwinViT import SwinTransformer
from conv_layers.ConvBranch import ConvBranch
import copy
import logging
import math
import torch
from embedding.embedding import EmbeddingHybridBranch
from decoder.decoder import DecoderBranch

import torch.nn as nn
from typing_extensions import Final
import numpy  as np 
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils.deprecate_utils import deprecated_arg
from collections.abc import Sequence
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm

class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x
MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}




################################################################################## add conv branch to network  
class CAtDecConvNet(nn.Module):

    patch_size: Final[int] = 2

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
        "checks on the input size are run during forward().",
    )

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )
    def __init__(
        self,
        config,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        
        ## input shapes 
        input_dim_0: int, n_channels_0: int,
        input_dim_1: int, n_channels_1: int,
        input_dim_2: int, n_channels_2: int,
        input_dim_3: int, n_channels_3: int,
        input_dim_4: int, n_channels_4: int,
        n_patches: int,
                
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str = "merging",
        use_v2: bool = False,
    ):
        super(CAtDecConvNet, self).__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)


        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        ##convoluitional branch
        
        self.conv_branch = ConvBranch(feature_size ,n_channels_0, n_channels_1,n_channels_2,n_channels_3,n_channels_4)

        ## embedding layer 
        self.embedding_branch = EmbeddingHybridBranch(config, 
                input_dim_0, feature_size,
                input_dim_1, n_channels_1,
                input_dim_2, n_channels_2,
                input_dim_3, n_channels_3,
                input_dim_4, n_channels_4,
                n_patches
                )
        
        ## cross attention layers 
        self.cross_attention_branch = CrossAttentionOutBranch(config, vis= True)
        
        ## final layer 
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        
        

    def forward(self, x_in):
        
        ## encoder 
        hidden_states = self.encoder(x_in, self.normalize)
               
        ## conv layers
        x_0, x_1, x_2, x_3, x_4 = self.conv_branch(x_in, hidden_states[0], hidden_states[1], hidden_states[2], hidden_states[3])
        
        print("pass the conv",x_0.shape, x_1.shape, x_2.shape, x_3.shape, x_4.shape )
        
        ## Embedding path 
        embeded_0, embeded_1, embeded_2, embeded_3, embeded_4 = self.embedding_branch(x_0, 
                                                                                      x_1, 
                                                                                      x_2, 
                                                                                      x_3, 
                                                                                      x_4 )
        
        ## cross attention layers 
        out_up_03, out_up_12, out_up_21, final_out = self.cross_attention_branch(embeded_0, embeded_1, embeded_2, embeded_3, embeded_4)
        print("pass the cross attention",hidden_states[3].shape,out_up_03.shape, out_up_12.shape, out_up_21.shape, final_out.shape)    
        logits = self.out(final_out )
        # print("logits-------------------", logits.shape)
        
        return logits
    
 #############################################################################################
from cross_attention.triple_cross_attention import TripleCrossAttentionBranch
from cross_attention.direct_triple_cross_attention import DirectTripleCrossAttentionBranch
from UNet.unet import ConvEncoder

class WLTANet(nn.Module):

    patch_size: Final[int] = 2

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
        "checks on the input size are run during forward().",
    )

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )
    def __init__(
        self,
        config,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        
        ## input shapes 
        input_dim_0: int, n_channels_0: int,
        input_dim_1: int, n_channels_1: int,
        input_dim_2: int, n_channels_2: int,
        input_dim_3: int, n_channels_3: int,
        input_dim_4: int, n_channels_4: int,
        n_patches: int,
                
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str = "merging",
        use_v2: bool = False,
    ):
        super(WLTANet, self).__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)


        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        self.encoder_backbone = config["encoder_backbone"] 
        if self.encoder_backbone == "Tr":
            self.encoder = SwinTransformer(
                in_chans=in_channels,
                embed_dim=feature_size,
                window_size=window_size,
                patch_size=patch_sizes,
                depths=depths,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dropout_path_rate,
                norm_layer=nn.LayerNorm,
                use_checkpoint=use_checkpoint,
                spatial_dims=spatial_dims,
                downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
                use_v2=use_v2,
            )
            ##convoluitional branch
            self.conv_branch = ConvBranch(feature_size ,n_channels_0, n_channels_1,n_channels_2,n_channels_3,n_channels_4)  
                      
        elif self.encoder_backbone == "Conv":
            self.encoder =  ConvEncoder()


        else:
            raise ValueError("Backnone type is no correct!")


        ## embedding layer 
        self.embedding_branch = EmbeddingHybridBranch(config, 
                input_dim_0, feature_size,
                input_dim_1, n_channels_1,
                input_dim_2, n_channels_2,
                input_dim_3, n_channels_3,
                input_dim_4, n_channels_4,
                n_patches
                )
        if config["att_mechanism"]=="Triple":
            self.cross_attention_branch = TripleCrossAttentionBranch(config, vis= True)
            
        elif config["att_mechanism"]=="DirectTriple":
            self.cross_attention_branch = DirectTripleCrossAttentionBranch(config, vis= True)  
             
        else:
            self.cross_attention_branch = CrossAttentionOutBranch(config, vis= True)
        
        ## decoder branch  
        self.decoder_branch = DecoderBranch(feature_size, out_channels)
        
    def forward(self, x_in):
        
        ## encoder 
        ## conv layers
        if self.encoder_backbone=="Tr":
            hidden_states = self.encoder(x_in)
            x_0, x_1, x_2, x_3, x_4 =  x_in, hidden_states[0], hidden_states[1], hidden_states[2], hidden_states[3]
            x_0, x_1, x_2, x_3, x_4 = self.conv_branch(x_0, x_1, x_2, x_3, x_4)
            
        elif self.encoder_backbone=="Conv":
            x_0, x_1, x_2, x_3, x_4 = self.encoder(x_in)
             
        ## Embedding path 
        embeded_0, embeded_1, embeded_2, embeded_3, embeded_4 = self.embedding_branch(x_0, 
                                                                                      x_1, 
                                                                                      x_2, 
                                                                                      x_3, 
                                                                                      x_4 )
        
        ## cross attention layers 
        out_3, out_2, out_1, out_0 = self.cross_attention_branch(embeded_0, embeded_1, embeded_2, embeded_3, embeded_4)
        # print("pass the cross attention",hidden_states[3].shape,out_3.shape, out_2.shape, out_1.shape, out_0.shape)    
            
        ## decoder branch 
        logits = self.decoder_branch(x_4, out_3, out_2, out_1, out_0)
        
        return logits
 


################################################################## shallow converter blocks 

