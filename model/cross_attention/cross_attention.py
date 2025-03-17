import torch
import torch.nn as nn
from embeddings2D import Embeddings2D
from embeddings3D import Embeddings3D
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
import copy
import logging
import math
import torch
from embedding.embedding import EmbeddingHybridBranch

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"



def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config["transformer"]["num_heads"]
        self.attention_head_size = int(config["hidden_size"] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config["hidden_size"], self.all_head_size)
        self.key = Linear(config["hidden_size"], self.all_head_size)
        self.value = Linear(config["hidden_size"], self.all_head_size)

        self.out = Linear(config["hidden_size"], config["hidden_size"])
        self.attn_dropout = Dropout(config["transformer"]["attention_dropout_rate"])
        self.proj_dropout = Dropout(config["transformer"]["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states_2d, hidden_states_3d):
        mixed_query_layer = self.query(hidden_states_3d)
        mixed_key_layer = self.key(hidden_states_3d)
        mixed_value_layer = self.value(hidden_states_2d)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # print("context_layer shape-------",context_layer.shape)
        # print("self.all_head_size-------",self.all_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # print("context_layer----------------- :", context_layer.shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp_CrossViT(nn.Module):
    def __init__(self, config):
        super(Mlp_CrossViT, self).__init__()
        self.fc1 = Linear(config["hidden_size"], 3072)
        self.fc2 = Linear(3072, config["hidden_size"])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config["transformer"]["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.attention_norm_2d = LayerNorm(config["hidden_size"], eps=1e-6)
        self.attention_norm_3d = LayerNorm(config["hidden_size"], eps=1e-6)

        self.ffn_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        self.ffn = Mlp_CrossViT(config)
        self.attn = Attention(config, vis)

    def forward(self, hidden_states_2d, hidden_states_3d):
        h = hidden_states_3d.clone()

        hidden_states_2d = self.attention_norm_2d(hidden_states_2d)
        hidden_states_3d = self.attention_norm_3d(hidden_states_3d)

        hidden_states_combined, weights = self.attn(hidden_states_2d, hidden_states_3d)
        hidden_states_combined = hidden_states_combined + h

        h = hidden_states_combined.clone()
        hidden_states_combined = self.ffn_norm(hidden_states_combined)
        hidden_states_combined = self.ffn(hidden_states_combined)
        hidden_states_combined = hidden_states_combined + h
        return hidden_states_combined, weights

class CrossAttention(nn.Module):
    def __init__(self, config, vis):
        super(CrossAttention, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        for _ in range(config["transformer"]["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states_1, hidden_states_2):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states_1, hidden_states_2)
            # hidden_states_2d, hidden_states_3d = hidden_states

            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
    
    
    

class CrossAttentionBranch(nn.Module):
    def __init__(self, config, vis):
        super(CrossAttentionBranch, self).__init__()
        
        ## first layer 
        self.CA_00 = CrossAttention(config, vis= True)
        self.CA_01 = CrossAttention(config, vis= True)
        self.CA_02 = CrossAttention(config, vis= True)
        self.CA_03 = CrossAttention(config, vis= True)

        ## second layer 
        self.CA_10 = CrossAttention(config, vis= True)
        self.CA_11 = CrossAttention(config, vis= True)
        self.CA_12 = CrossAttention(config, vis= True)

        ## third layer 
        self.CA_20 = CrossAttention(config, vis= True)
        self.CA_21 = CrossAttention(config, vis= True)

        ## fourth layer 
        self.CA_30 = CrossAttention(config, vis= True)

        self.reconstruction_block = nn.Sequential(
            nn.Linear(64 * config["hidden_size"], 12 * 6 * 6 * 6),  # Linear layer to reduce elements
            nn.ReLU(),
            nn.Unflatten(1, (12, 6, 6, 6)),       # Reshape to [1, 12, 3, 3, 3]
            nn.ConvTranspose3d(12, 48, kernel_size=4, stride=2, padding=1),  # Upsample and increase channels to 48
            nn.Upsample(size=(96, 96, 96), mode='trilinear', align_corners=True)  # Final upsample to [1, 48, 96, 96, 96]
        )

        
        ## add a layer for number of classes here 


    def forward(self, em_0,em_1, em_2,em_3,em_4):
        
        ## first layer 
        out00, _ = self.CA_00(em_0, em_1)
        out01, _ = self.CA_01(em_1, em_2)
        out02, _ = self.CA_02(em_2, em_3)
        out03, _ = self.CA_03(em_3, em_4)
        
        ## second layer 
        out10, _ = self.CA_10(out00, out01)
        out11, _ = self.CA_11(out01, out02)
        out12, _ = self.CA_12(out02, out03)


        ## third layer 
        out20, _ = self.CA_20(out10, out11)
        out21, _ = self.CA_21(out11, out12)

        ## fourth layer 
        out30, _ = self.CA_30(out20, out21)
        
        ## upsampling layer soon 
        print("out30.shape-----------------------:",out30.shape)
        batch_size, n_patches, hidden_size = out30.shape
        reshaped_encoded = out30.view(batch_size, -1)  # Flatten to shape [1, 27 * 120]

        final_out = self.reconstruction_block(reshaped_encoded)
        print("final_out shape", final_out.shape)
        return final_out



class CrossAttentionOutBranch(nn.Module):
    def __init__(self, config, vis, upsacale_branch= True):
        super(CrossAttentionOutBranch, self).__init__()
        
        ## first layer 
        self.CA_00 = CrossAttention(config, vis= True)
        self.CA_01 = CrossAttention(config, vis= True)
        self.CA_02 = CrossAttention(config, vis= True)
        self.CA_03 = CrossAttention(config, vis= True)

        ## second layer 
        self.CA_10 = CrossAttention(config, vis= True)
        self.CA_11 = CrossAttention(config, vis= True)
        self.CA_12 = CrossAttention(config, vis= True)

        ## third layer 
        self.CA_20 = CrossAttention(config, vis= True)
        self.CA_21 = CrossAttention(config, vis= True)

        ## fourth layer 
        self.CA_30 = CrossAttention(config, vis= True)
        self.reconstruction_block = nn.Sequential(
            nn.Linear(64 * config["hidden_size"], 12 * 3 * 3 * 3),  # Linear layer to reduce elements
            nn.ReLU(),
            nn.Unflatten(1, (12, 3, 3, 3)),       # Reshape to [1, 12, 3, 3, 3]
            nn.ConvTranspose3d(12, 48, kernel_size=4, stride=2, padding=1),  # Upsample and increase channels to 48
            nn.Upsample(size=(96, 96, 96), mode='trilinear', align_corners=True)  # Final upsample to [1, 48, 96, 96, 96]
        )
        if config["shallow_rec"]==True: 
            self.upscale_21 = nn.Sequential(
            nn.Linear(64 * config["hidden_size"], 12 * 3 * 3 * 3),  # Linear layer to reduce elements
            nn.ReLU(),
            nn.Unflatten(1, (12, 3, 3, 3)),       # Reshape to [1, 12, 3, 3, 3]
            nn.ConvTranspose3d(12, 48, kernel_size=4, stride=2, padding=1),  # Upsample and increase channels to 48
            nn.Upsample(size=(48, 48, 48), mode='trilinear', align_corners=True),  # Final upsample to [1, 48, 96, 96, 96]
            nn.Dropout(0.2)
            )
            self.upscale_03 = nn.Sequential(
            nn.Linear(64 * config["hidden_size"], 12 * 3 * 3 * 3),  # Linear layer to reduce elements
            nn.ReLU(),
            nn.Unflatten(1, (12, 3, 3, 3)),       # Reshape to [1, 12, 3, 3, 3]
            nn.ConvTranspose3d(12, 96, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose3d(96, 192, kernel_size=4, stride=2, padding=1),
            # Upsample and increase channels to 48
            nn.Upsample(size=(12, 12, 12), mode='trilinear', align_corners=True),  # Final upsample to [1, 48, 96, 96, 96]
            nn.Dropout(0.2)
            )
            
            ## 10 scale 
            self.upscale_12 = nn.Sequential(
            nn.Linear(64 * config["hidden_size"], 12 * 3 * 3 * 3),  # Linear layer to reduce elements
            nn.ReLU(),
            nn.Unflatten(1, (12, 3, 3, 3)),       # Reshape to [1, 12, 3, 3, 3]
            nn.ConvTranspose3d(12, 48, kernel_size=4, stride=2, padding=1),  # Upsample and increase channels to 48
            nn.ConvTranspose3d(48, 96, kernel_size=4, stride=2, padding=1),  # Upsample and increase channels to 48
            nn.Upsample(size=(24, 24, 24), mode='trilinear', align_corners=True),  # Final upsample to [1, 48, 96, 96, 96]
            nn.Dropout(0.2)
            )        
         
        else:
            ## 00 scale 
            self.upscale_03 = nn.Sequential(
            nn.Linear(64 * config["hidden_size"], 48 * 3 * 3 * 3),  # Linear layer to reduce elements
            nn.ReLU(),
            nn.Unflatten(1, (48, 3, 3, 3)),       # Reshape to [1, 12, 3, 3, 3]
            nn.ConvTranspose3d(48, 192, kernel_size=4, stride=2, padding=1),  # Upsample and increase channels to 48
            nn.Upsample(size=(12, 12, 12), mode='trilinear', align_corners=True),  # Final upsample to [1, 48, 96, 96, 96]
            nn.Dropout(0.2)
            )
            
            ## 10 scale 
            self.upscale_12 = nn.Sequential(
            nn.Linear(64 * config["hidden_size"], 24 * 3 * 3 * 3),  # Linear layer to reduce elements
            nn.ReLU(),
            nn.Unflatten(1, (24, 3, 3, 3)),       # Reshape to [1, 12, 3, 3, 3]
            nn.ConvTranspose3d(24, 96, kernel_size=4, stride=2, padding=1),  # Upsample and increase channels to 48
            nn.Upsample(size=(24, 24, 24), mode='trilinear', align_corners=True),  # Final upsample to [1, 48, 96, 96, 96]
            nn.Dropout(0.2)
            )

            ## 20 scale 
            self.upscale_21 = nn.Sequential(
            nn.Linear(64 * config["hidden_size"], 12 * 3 * 3 * 3),  # Linear layer to reduce elements
            nn.ReLU(),
            nn.Unflatten(1, (12, 3, 3, 3)),       # Reshape to [1, 12, 3, 3, 3]
            nn.ConvTranspose3d(12, 48, kernel_size=4, stride=2, padding=1),  # Upsample and increase channels to 48
            nn.Upsample(size=(48, 48, 48), mode='trilinear', align_corners=True),  # Final upsample to [1, 48, 96, 96, 96]
            nn.Dropout(0.2)
            )


        
        ## add a layer for number of classes here 


    def forward(self, em_0,em_1, em_2,em_3,em_4):
        
        ## first layer 
        out00, _ = self.CA_00(em_0, em_1)
        out01, _ = self.CA_01(em_1, em_2)
        out02, _ = self.CA_02(em_2, em_3)
        out03, _ = self.CA_03(em_3, em_4)
        
        ## second layer 
        out10, _ = self.CA_10(out00, out01)
        out11, _ = self.CA_11(out01, out02)
        out12, _ = self.CA_12(out02, out03)


        ## third layer 
        out20, _ = self.CA_20(out10, out11)
        out21, _ = self.CA_21(out11, out12)

        ## fourth layer 
        out30, _ = self.CA_30(out20, out21)
        
        ## upsampling layer soon 
        # if upsacale_branch == True: 
        out_up_03 = self.upscale_03(self.reshape_encoded(out03) )
        out_up_12 = self.upscale_12(self.reshape_encoded(out12) )
        out_up_21 = self.upscale_21(self.reshape_encoded(out21) )
        final_out = self.reconstruction_block(self.reshape_encoded(out30))                    
        return out_up_03, out_up_12, out_up_21, final_out      
        # else:
        #     ### final layer
        #     reshaped_encoded = self.reshape_encoded(out30)
        #     final_out = self.reconstruction_block(reshaped_encoded)
        #     print("final_out shape", final_out.shape)
        #     return final_out

    def reshape_encoded(self, tensor):
        """
        Reshapes a tensor of shape [batch_size, n_patches, hidden_size]
        into a flattened shape [batch_size, n_patches * hidden_size].

        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, n_patches, hidden_size].

        Returns:
            torch.Tensor: Reshaped tensor of shape [batch_size, n_patches * hidden_size].
        """
        batch_size, n_patches, hidden_size = tensor.shape
        reshaped_tensor = tensor.view(batch_size, -1)
        return reshaped_tensor


#########################################################################################
from typing_extensions import Final
from monai.utils.deprecate_utils import deprecated_arg
from collections.abc import Sequence
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
import numpy as np 
from SwinViT.SwinViT import SwinTransformer
rearrange, _ = optional_import("einops", name="rearrange")
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

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


################################################################################## CAtDEcNet
from conv_layers.ConvBranch import ConvBranch

class CAtDecNet(nn.Module):

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
        super(CAtDecNet, self).__init__()
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

        self.conv_0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        ## embedding layer 
        self.embedding_branch = EmbeddingHybridBranch(config, 
                input_dim_0, n_channels_0,
                input_dim_1, n_channels_1,
                input_dim_2, n_channels_2,
                input_dim_3, n_channels_3,
                input_dim_4, n_channels_4,
                n_patches
                )
        
        ## cross attention layers 
        self.cross_attention_branch = CrossAttentionBranch(config, vis= True)
        
        
        ## final layer 
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x_in):
        
        ## encoder 
        hidden_states = self.encoder(x_in, self.normalize)
        
        x_in_1 = self.conv_0(x_in)
        
        print(x_in_1.shape, hidden_states[0].shape, hidden_states[1].shape, hidden_states[2].shape, hidden_states[3].shape )
        ## Embedding path 
        embeded_0, embeded_1, embeded_2, embeded_3, embeded_4 = self.embedding_branch(x_in_1, 
                                                                                    hidden_states[0], 
                                                                                    hidden_states[1], 
                                                                                    hidden_states[2], 
                                                                                    hidden_states[3] )
        
        ## cross attention layers 
        output = self.cross_attention_branch(embeded_0, embeded_1, embeded_2, embeded_3, embeded_4)
        
        logits = self.out(output )
        # print("logits-------------------", logits.shape)
        
        return logits
    
   
