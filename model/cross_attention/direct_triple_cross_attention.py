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


class SelfAttention(nn.Module):
    def __init__(self, config, vis):
        super(SelfAttention, self).__init__()
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

    def forward(self, hidden_states_i):
        mixed_query_layer = self.query(hidden_states_i)
        mixed_key_layer = self.key(hidden_states_i)
        mixed_value_layer = self.value(hidden_states_i)

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





############################################################################## direct twice cross attention 

class BlockDirectTwice(nn.Module):
    def __init__(self, config, vis):
        super(BlockDirectTwice, self).__init__()
        self.hidden_size = config["hidden_size"]
        
        ## norm of em_0,em_1, em_2 inputs
        self.attention_norm_0 = LayerNorm(config["hidden_size"], eps=1e-6)
        self.attention_norm_1 = LayerNorm(config["hidden_size"], eps=1e-6)

        ## Attention of em_0 and em_1
        self.attn_00 = SelfAttention(config, vis)
        self.attn_01 = SelfAttention(config, vis)

        self.ffn_norm_01 = LayerNorm(config["hidden_size"], eps=1e-6)
        self.ffn_01 = Mlp_CrossViT(config)
        self.comb = CombinerTwice(config)  

    def forward(self, hidden_states_0, hidden_states_1):
        
        ## norm of input embeddings(em_0, em_1, em_2)
        h_0 = hidden_states_0.clone()
        h_1 = hidden_states_1.clone()

        hidden_states_0 = self.attention_norm_0(hidden_states_0)
        hidden_states_1 = self.attention_norm_1(hidden_states_1)

        ## attention for each embedding  
        hidden_states_00, _ = self.attn_00(hidden_states_0)
        hidden_states_01, _ = self.attn_01(hidden_states_1)
        
        ## skip connections 
        hidden_states_00 = hidden_states_00 + h_0
        hidden_states_01 = hidden_states_01 + h_1
        
        ## getting the maximum or weights 
        hidden_states = self.comb(hidden_states_00, hidden_states_01 )
        
        ## norm and skip connections
        h = hidden_states.clone()
        hidden_states = self.ffn_norm_01(hidden_states)
        hidden_states = self.ffn_01(hidden_states)
        hidden_states = hidden_states + h
        
        return hidden_states
 
class DirectTwiceCrossAttention(nn.Module):
    def __init__(self, config, vis=True):
        super(DirectTwiceCrossAttention, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        for _ in range(config["transformer"]["num_layers"]):
            layer = BlockDirectTwice(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states_0, hidden_states_1):
        # attn_weights = []
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states_0, hidden_states_1 )
            
            # if self.vis:
                # attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded 

class CombinerTwice(nn.Module):
    def __init__(self, config):
        """
        Initializes the HiddenStateCombiner class.

        Args:
            use_max (bool): Determines which combination method to use in the forward pass.
                           True for `combine_max`, False for `combine_weights`.
        """
        super(CombinerTwice, self).__init__()  # Properly initialize nn.Module
        self.type_combine = config["type_combine"] 
        self.up_hyper = config["up_hyper"]
        self.middle_hyper = config["middle_hyper"]
        self.down_hyper = config["down_hyper"]
        
        self.weights = nn.Parameter(torch.zeros(3))
        self.weights.data = torch.tensor([1.0, 1.0 ])
        
    def combine_max(self, hidden_states_00, hidden_states_01):
        hidden_states = torch.max(hidden_states_00, hidden_states_01)
        # print("flag for max")
        return hidden_states

    def combine_weights(self, hidden_states_00, hidden_states_01):
        hidden_states = self.up_hyper*hidden_states_00 + self.middle_hyper*hidden_states_01
        # print("flag for weights")
        return hidden_states

    def forward(self, hidden_states_00, hidden_states_01 ):
        if self.type_combine=="max":
            return self.combine_max(hidden_states_00, hidden_states_01)
        elif self.type_combine=="weights":
            return self.combine_weights(hidden_states_00, hidden_states_01)
        elif self.type_combine=="learnable":
            # print("self.weights----------------------------", self.weights)
            return (
                self.weights[0] * hidden_states_00 +
                self.weights[1] * hidden_states_01 
            )
        else:
            raise ValueError(f"Unknown combination type: {self.type_combine}")


        
############################################################################## direct triple cross attention 
class BlockDirectTriple(nn.Module):
    def __init__(self, config, vis):
        super(BlockDirectTriple, self).__init__()
        self.hidden_size = config["hidden_size"]
        
        ## norm of em_0,em_1, em_2 inputs
        self.attention_norm_0 = LayerNorm(config["hidden_size"], eps=1e-6)
        self.attention_norm_1 = LayerNorm(config["hidden_size"], eps=1e-6)
        self.attention_norm_2 = LayerNorm(config["hidden_size"], eps=1e-6)

        ## Attention of em_0 and em_1
        self.attn_00 = SelfAttention(config, vis)
        self.attn_01 = SelfAttention(config, vis)
        self.attn_02 = SelfAttention(config, vis)

        self.ffn_norm_01 = LayerNorm(config["hidden_size"], eps=1e-6)
        self.ffn_01 = Mlp_CrossViT(config)
        self.comb = Combiner(config)  

    def forward(self, hidden_states_0, hidden_states_1, hidden_states_2):
        
        ## norm of input embeddings(em_0, em_1, em_2)
        h_0 = hidden_states_0.clone()
        h_1 = hidden_states_1.clone()
        h_2 = hidden_states_2.clone()

        hidden_states_0 = self.attention_norm_0(hidden_states_0)
        hidden_states_1 = self.attention_norm_1(hidden_states_1)
        hidden_states_2 = self.attention_norm_2(hidden_states_2)

        ## attention for each embedding  
        hidden_states_00, _ = self.attn_00(hidden_states_0)
        hidden_states_01, _ = self.attn_01(hidden_states_1)
        hidden_states_02, _ = self.attn_02(hidden_states_2)
        
        ## skip connections 
        hidden_states_00 = hidden_states_00 + h_0
        hidden_states_01 = hidden_states_01 + h_1
        hidden_states_02 = hidden_states_02 + h_2
        
        ## getting the maximum or weights 
        hidden_states = self.comb(hidden_states_00, hidden_states_01, hidden_states_02 )
        
        ## norm and skip connections
        h = hidden_states.clone()
        hidden_states = self.ffn_norm_01(hidden_states)
        hidden_states = self.ffn_01(hidden_states)
        hidden_states = hidden_states + h
        
        return hidden_states

class DirectTripleCrossAttention(nn.Module):
    def __init__(self, config, vis=True):
        super(DirectTripleCrossAttention, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        for _ in range(config["transformer"]["num_layers"]):
            layer = BlockDirectTriple(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states_0, hidden_states_1, hidden_states_2):
        # attn_weights = []
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states_0, hidden_states_1, hidden_states_2)
            
            # if self.vis:
                # attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded

class Combiner(nn.Module):
    def __init__(self, config):
        """
        Initializes the HiddenStateCombiner class.

        Args:
            use_max (bool): Determines which combination method to use in the forward pass.
                           True for `combine_max`, False for `combine_weights`.
        """
        super(Combiner, self).__init__()  # Properly initialize nn.Module
        self.type_combine = config["type_combine"] 
        self.up_hyper = config["up_hyper"]
        self.middle_hyper = config["middle_hyper"]
        self.down_hyper = config["down_hyper"]
        
        self.weights = nn.Parameter(torch.zeros(3))
        self.weights.data = torch.tensor([1.0, 1.0, 1.0])
            
    def combine_max(self, hidden_states_00, hidden_states_01, hidden_states_02):
        hidden_states = torch.max(hidden_states_02, torch.max(hidden_states_00, hidden_states_01))
        # print("flag for max")
        return hidden_states

    def combine_weights(self, hidden_states_00, hidden_states_01, hidden_states_02):
        hidden_states = self.up_hyper*hidden_states_00 + self.middle_hyper*hidden_states_01+ self.down_hyper*hidden_states_02
        # print("flag for weights")
        return hidden_states

    def forward(self, hidden_states_00, hidden_states_01, hidden_states_02):
        if self.type_combine=="max":
            return self.combine_max(hidden_states_00, hidden_states_01, hidden_states_02)
        elif self.type_combine=="weights":
            return self.combine_weights(hidden_states_00, hidden_states_01, hidden_states_02)
        elif self.type_combine=="learnable":
            # print("self.weights----------------------------", self.weights)
            return (
                self.weights[0] * hidden_states_00 +
                self.weights[1] * hidden_states_01 +
                self.weights[2] * hidden_states_02
            )
        else:
            raise ValueError(f"Unknown combination type: {self.type_combine}")


    
from cross_attention.cross_attention import CrossAttention

class DirectTripleCrossAttentionBranch(nn.Module):
    def __init__(self, config, vis, upsacale_branch= True):
        super(DirectTripleCrossAttentionBranch, self).__init__()
        
        ## first layer 
        if config["first_row"]=="cross":
            self.CA_00 = CrossAttention(config, vis= True)
            self.CA_10 = CrossAttention(config, vis= True)
            self.CA_20 = CrossAttention(config, vis= True)
            self.CA_30 = CrossAttention(config, vis= True)
            
        elif config["first_row"]=="self":
            self.CA_00 = DirectTwiceCrossAttention(config, vis= True )
            self.CA_10 = DirectTwiceCrossAttention(config, vis= True )
            self.CA_20 = DirectTwiceCrossAttention(config, vis= True )
            self.CA_30 = DirectTwiceCrossAttention(config, vis= True)

        else:
            raise ValueError(f"Unknown config for first_row attention")
     
        self.CA_01 = DirectTripleCrossAttention(config, vis= True)
        self.CA_02 = DirectTripleCrossAttention(config, vis= True)
        self.CA_03 = DirectTripleCrossAttention(config, vis= True)

        ## second layer 
        self.CA_11 = DirectTripleCrossAttention(config, vis= True)
        self.CA_12 = DirectTripleCrossAttention(config, vis= True)

        ## third layer 
        self.CA_21 = DirectTripleCrossAttention(config, vis= True)

        ## fourth layer 
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
        out00    = self.CA_00(em_0, em_1)
        out01    = self.CA_01(em_0, em_1, em_2)
        out02    = self.CA_02(em_1, em_2, em_3)
        out03    = self.CA_03(em_2, em_3, em_4)
        ## second layer 
        out10   = self.CA_10(out00, out01)
        out11   = self.CA_11(out00, out01, out02)
        out12   = self.CA_12(out01, out02, out03)


        ## third layer 
        out20    = self.CA_20(out10, out11)
        out21    = self.CA_21(out10, out11, out12)

        ## fourth layer 
        out30    = self.CA_30(out20, out21)
        
        ## upsampling layer soon 
        # if upsacale_branch == True: 
        out_up_03 = self.upscale_03(self.reshape_encoded(out03) )
        out_up_12 = self.upscale_12(self.reshape_encoded(out12) )
        out_up_21 = self.upscale_21(self.reshape_encoded(out21) )
        final_out = self.reconstruction_block(self.reshape_encoded(out30))    
        # print("'---------------------------------------------------------------")                
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

    
    
