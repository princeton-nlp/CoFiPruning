""" PyTorch OPT model."""
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import *
import torch.nn.functional as F

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

def prune_layer_norm(layernorm, index):
    layernorm.weight = torch.nn.parameter.Parameter(
        layernorm.weight.index_select(0, index))
    layernorm.bias = torch.nn.parameter.Parameter(
        layernorm.bias.index_select(0, index))
    layernorm.normalized_shape = (len(index),)

def turn_head_z(head_z, head_layer_z):
    head_z = head_z.squeeze().clone()
    if head_layer_z is not None:
        head_z *= head_layer_z
    to_prune_heads = torch.where(head_z == 0)[0].view(-1).tolist()
    return to_prune_heads

def turn_mlp_z(intermediate_z, mlp_z):
    intermediate_z_layer = intermediate_z.squeeze().clone()
    if mlp_z is not None:
        intermediate_z_layer *= mlp_z
    keep_intermediate_dims = torch.where(intermediate_z_layer != 0)[0].tolist()
    return keep_intermediate_dims 

def turn_hidden_z(hidden_z):
    kept_hidden_dims = hidden_z.squeeze().nonzero().squeeze()
    return kept_hidden_dims

    
class CoFiLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input, hidden_z=None):
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(
                input, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(
                compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            output = input.clone()
            output[..., remaining_index] = normed_input
        else:
            output = F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        return output
    

class CoFiOPTAttention(OPTAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)
        self.pruned_heads = set()

     # override 
    def prune_heads(self, head_z, head_layer_z):
        # update params #
        head_z_for_update = torch.repeat_interleave(head_z, self.head_dim)
        self.v_proj.weight.data = self.v_proj.weight.data.transpose(0, 1).mul(head_z_for_update).transpose(0, 1)
        self.v_proj.bias.data = self.v_proj.bias.data.mul(head_z_for_update)
        if head_layer_z is not None:
            self.out_proj.weight.data = self.out_proj.weight.data.transpose(0, 1).mul(head_layer_z).transpose(0, 1)
            self.out_proj.bias.data = self.out_proj.bias.data.mul(head_layer_z)
        #################
            
        to_prune_heads = turn_head_z(head_z, head_layer_z)
        len_to_prune_heads = len(to_prune_heads)
        if len_to_prune_heads == 0:
            print(f"    Heads: {self.num_heads} -> {self.num_heads}")
            return

        heads, index = find_pruneable_heads_and_indices(
            to_prune_heads, self.num_heads, self.head_dim, self.pruned_heads
        )
        
        # Prune linear layers
        # setting layers to be None if all the heads are pruned
        if len(index) == 0:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None
        else:

            self.q_proj = prune_linear_layer(self.q_proj, index)
            self.k_proj = prune_linear_layer(self.k_proj , index)
            self.v_proj = prune_linear_layer(self.v_proj , index)
            self.out_proj = prune_linear_layer(self.out_proj, index, dim=1)

        print(f"    Heads: {self.num_heads} -> {self.num_heads - len(heads)}")

        # Update hyper params and store pruned heads
        self.num_heads = self.num_heads - len(heads)
        self.all_head_size = self.head_dim * self.num_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def prune_hidden_states(self, hidden_z): 
        keep_dims = turn_hidden_z(hidden_z)
        if self.q_proj is not None:
            # update_params #
            self.q_proj.weight.data = self.q_proj.weight.data.mul(hidden_z)
            self.k_proj.weight.data = self.k_proj.weight.data.mul(hidden_z)
            self.v_proj.weight.data = self.v_proj.weight.data.mul(hidden_z)
            self.out_proj.weight.data = self.out_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
            self.out_proj.bias.data = self.out_proj.bias.data.mul(hidden_z)
            ################
            self.q_proj = prune_linear_layer(self.q_proj, keep_dims, dim=1)
            self.k_proj = prune_linear_layer(self.k_proj, keep_dims, dim=1)
            self.v_proj = prune_linear_layer(self.v_proj, keep_dims, dim=1)
            self.out_proj = prune_linear_layer(self.out_proj, keep_dims)
            print(f"    Attention hidden dim: {self.embed_dim} -> {len(keep_dims)}")
        self.embed_dim = len(keep_dims)
    
    # TODO: change to prune_modules for Attention
    def prune_modules(self, zs):
        head_z = zs["head_z"]
        head_layer_z = zs["head_layer_z"]
        head_z * head_layer_z
        
    # def update_params_for_heads()   
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        head_z: Optional[torch.Tensor]=None,
        head_layer_z: Optional[torch.Tensor]=None, 
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        if self.v_proj is None:
            return (None, None, None)
        
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        
        #### Added for CoFi ####
        if head_z is not None:
            attn_output *= head_z
        #######################
        
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.num_heads * self.head_dim)

        # during training, when head_z.sum() == 0, attn_output 
        attn_output = self.out_proj(attn_output)
       
        #### Added for CoFi ####  
        # not sure how to process
        # during training, when head_z.sum() == 0, the attn_output != 0 bc of bias in self.out_proj     
        if head_layer_z is not None:
            attn_output *= head_layer_z
        #######################

        return attn_output, attn_weights_reshaped, past_key_value
    

class CoFiOPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CoFiOPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            # bias=config.enable_bias, # deleted for version control
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = CoFiLayerNorm(self.embed_dim) # , elementwise_affine=config.layer_norm_elementwise_affine
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim) # , bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim) # , bias=config.enable_bias)
        self.final_layer_norm = CoFiLayerNorm(self.embed_dim) # , elementwise_affine=config.layer_norm_elementwise_affine)
        self.config = config

    def prune_heads(self, head_z, head_layer_z):
        self.self_attn.prune_heads(head_z, head_layer_z)
        
    def prune_mlps(self, intermediate_z, mlp_z):
        # update params #
        self.fc2.weight.data = self.fc2.weight.data.mul(intermediate_z.squeeze(0))
        if mlp_z is not None:
            self.fc2.weight.data = self.fc2.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)
            self.fc2.bias.data = self.fc2.bias.data.mul(mlp_z)
        #################
        keep_dim = turn_mlp_z(intermediate_z, mlp_z)
        device = self.fc1.weight.device
        if len(keep_dim) == self.fc1.weight.shape[0]:
            print(f"    FFN intermediate dim: {self.config.ffn_dim} -> {len(keep_dim)}")
            return 
            
        if len(keep_dim) == 0:
            self.fc1 = None; self.fc2 = None
        else:
            keep_dim_index = torch.tensor(keep_dim).long().to(device)
            self.fc1 = prune_linear_layer(self.fc1, keep_dim_index, dim=0)
            self.fc2 = prune_linear_layer(self.fc2, keep_dim_index, dim=1)
        print(f"    FFN intermediate dim: {self.config.ffn_dim} -> {len(keep_dim)}")
    
    def prune_hidden_states(self, hidden_z):
        keep_dims = turn_hidden_z(hidden_z)
        self.self_attn.prune_hidden_states(hidden_z)
        prune_layer_norm(self.self_attn_layer_norm, keep_dims)
        
        if self.fc1 is not None:
            # update params #
            self.fc1.weight.data = self.fc1.weight.data.mul(hidden_z)
            self.fc2.weight.data = self.fc2.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
            self.fc2.bias.data = self.fc2.bias.data.mul(hidden_z)
            #################
            self.fc1 = prune_linear_layer(self.fc1, keep_dims, dim=1)
            self.fc2 = prune_linear_layer(self.fc2, keep_dims, dim=0)
            prune_layer_norm(self.final_layer_norm, keep_dims)
        print(f"    FFN hidden dim: {self.config.hidden_size} -> {len(keep_dims)}")
        self.embed_dim = len(keep_dims)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        head_z: Optional[torch.Tensor]=None,
        head_layer_z: Optional[torch.Tensor]=None,
        hidden_z: Optional[torch.Tensor]=None,
        intermediate_z: Optional[torch.Tensor]=None,
        mlp_z: Optional[torch.Tensor]=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
            
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            # previous layer should have process hidden_z before layer norm
            hidden_states = self.self_attn_layer_norm(hidden_states, hidden_z=hidden_z)

            #### Added for CoFi ####
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        #######################
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            head_z=head_z,
            head_layer_z=head_layer_z
        )

        #### Added for CoFi ####
        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)
        #######################
        
        if hidden_states is None:    
            hidden_states = residual
        else:
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states, hidden_z=hidden_z)

            #### Added for CoFi ####
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            #######################
        
        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        if self.fc1 is not None:
            # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
            if self.do_layer_norm_before:
                hidden_states = self.final_layer_norm(hidden_states, hidden_z=hidden_z)
            
                #### Added for CoFi ####
                if hidden_z is not None:
                    hidden_states = hidden_states.mul(hidden_z)
                #######################
            
            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)

            #### Added for CoFi ####
            if intermediate_z is not None:
                hidden_states = hidden_states.mul(intermediate_z)
            #######################
            
            hidden_states = self.fc2(hidden_states)

            #### Added for CoFi ####
            if mlp_z is not None:
                hidden_states = hidden_states.mul(mlp_z)
            
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            #######################

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = (residual + hidden_states)

            # 350m applies layer norm AFTER attention
            if not self.do_layer_norm_before:
                hidden_states = self.final_layer_norm(hidden_states)
        
        hidden_states = hidden_states.view(hidden_states_shape)
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# class CoFiEmbeddings(nn.Module):

class CoFiOPTDecoder(OPTDecoder):
    def __init__(self, config: OPTConfig):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = CoFiLayerNorm(config.hidden_size)
        else:
            self.final_layer_norm = None
            
        self.layers = nn.ModuleList([CoFiOPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    def prune_modules(self, zs):
        hidden_zs = zs.get("hidden_z", None)
        head_zs = zs.get("head_z", None)
        head_layer_zs = zs.get("head_layer_z", None)
        intermediate_zs = zs.get("intermediate_z", None)
        mlp_zs = zs.get("mlp_z", None)
        
        if hidden_zs is not None:
            # update params #
            # self.embed_tokens.weight.data = self.embed_tokens.weight.data.mul(hidden_zs)
            self.embed_positions.weight.data = self.embed_positions.weight.data.mul(hidden_zs) 
            ################
            
            kept_hidden_dims = turn_hidden_z(hidden_zs)
            self.embed_tokens.weight = torch.nn.parameter.Parameter(
                self.embed_tokens.weight.index_select(1, kept_hidden_dims).clone())
            self.embed_tokens.embedding_dim = len(kept_hidden_dims)
            self.embed_positions.weight = torch.nn.parameter.Parameter(
                self.embed_positions.weight.index_select(1, kept_hidden_dims).clone())
            self.embed_positions.embedding_dim = len(kept_hidden_dims)
            print(f"embedding dimensions: {self.embed_tokens.embedding_dim} -> {len(kept_hidden_dims)}")
            
            if self.final_layer_norm is not None:
                prune_layer_norm(self.final_layer_norm, kept_hidden_dims)

            if self.project_out is not None:
                # update params #
                self.project_out.weight.data *= hidden_zs
                #################

                prune_linear_layer(self.project_out, kept_hidden_dims, dim=1)
                print("project_out dimensions: {self.project_out.in_features} -> {len(kept_hidden_dims)}")
                
        for i, layer in enumerate(self.layers):
            print(f"Pruning layer {i} :")
            head_z = head_zs[i] if head_zs is not None else None
            head_layer_z = head_layer_zs[i] if head_layer_zs is not None else None
            intermediate_z = intermediate_zs[i] if intermediate_zs is not None else None
            mlp_z = mlp_zs[i] if mlp_zs is not None else None
            layer.prune_heads(head_z, head_layer_z)
            layer.prune_mlps(intermediate_z, mlp_z)
            layer.prune_hidden_states(hidden_zs)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # added masks
        head_z: Optional[torch.Tensor]=None,
        head_layer_z: Optional[torch.Tensor]=None,
        intermediate_z: Optional[torch.Tensor]=None,
        mlp_z: Optional[torch.Tensor]=None,
        hidden_z: Optional[torch.Tensor]=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        ## Added for CoFi ####
        if hidden_z is not None:
            inputs_embeds = inputs_embeds.mul(hidden_z)
            pos_embeds = pos_embeds.mul(hidden_z)
        #####################
        
        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                intermediate_z=intermediate_z[idx] if intermediate_z is not None else None,
                head_z=head_z[idx] if head_z is not None else None,
                mlp_z=mlp_z[idx] if mlp_z is not None else None,
                head_layer_z=head_layer_z[idx] if head_layer_z is not None else None,
                hidden_z=hidden_z
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states, hidden_z=hidden_z)
            
            
            #### added for CoFi ####
            if hidden_z is not None:
                hidden_states = hidden_states * hidden_z
            ########################

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CoFiOPTModel(OPTModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = CoFiOPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # added masks
        head_z: Optional[torch.Tensor]=None,
        head_layer_z: Optional[torch.Tensor]=None,
        intermediate_z: Optional[torch.Tensor]=None,
        mlp_z: Optional[torch.Tensor]=None,
        hidden_z: Optional[torch.Tensor]=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class CoFiOPTForCausalLM(OPTForCausalLM):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = CoFiOPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def prune_modules(self, zs):
        if "hidden_z" in zs:
            hidden_zs = zs["hidden_z"]
            kept_hidden_dims = turn_hidden_z(hidden_zs)
            if self.model.decoder.project_out is None:
                # lm_head and embed_tokens are tied
                # update params #
                self.lm_head.weight.data = self.lm_head.weight.data.mul(hidden_zs)
                #################

                self.lm_head = prune_linear_layer(self.lm_head, kept_hidden_dims, dim=1)
                print(f"lm_head input dimension: {self.config.word_embed_proj_dim} -> {len(kept_hidden_dims)}")
        self.model.decoder.prune_modules(zs)
            
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
         # added masks
        head_z: Optional[torch.Tensor]=None,
        head_layer_z: Optional[torch.Tensor]=None,
        intermediate_z: Optional[torch.Tensor]=None,
        mlp_z: Optional[torch.Tensor]=None,
        hidden_z: Optional[torch.Tensor]=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
