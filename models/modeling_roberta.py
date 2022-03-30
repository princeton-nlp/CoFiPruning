# allows removing layers of heads and mlps
import pdb

import transformers

__version__ = transformers.__version__

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.file_utils import ModelOutput

if __version__.startswith("3"):
    from transformers.modeling_roberta import (RobertaForSequenceClassification,
                                               RobertaForMaskedLM,
                                               RobertaModel,
                                               RobertaEncoder,
                                               RobertaLayer,
                                               RobertaAttention,
                                               RobertaSelfAttention,
                                               RobertaSelfOutput,
                                               RobertaOutput,
                                               MaskedLMOutput,
                                               create_position_ids_from_input_ids,
                                               RobertaEmbeddings)
else:
    from transformers.models.roberta.modeling_roberta import (RobertaForSequenceClassification,
                                                              RobertaForMaskedLM,
                                                              RobertaModel,
                                                              RobertaEncoder,
                                                              RobertaLayer,
                                                              RobertaAttention,
                                                              RobertaSelfAttention,
                                                              RobertaSelfOutput,
                                                              RobertaOutput,
                                                              MaskedLMOutput,
                                                              create_position_ids_from_input_ids,
                                                              RobertaEmbeddings)


import torch
from torch import nn
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
import math
import logging
from typing import Dict, List
import numpy as np
from models.modeling_bert import CoFiLayerNorm 

logger = logging.getLogger(__name__)

BertLayerNorm = CoFiLayerNorm 


class CoFiRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = NewRobertaModel(config)

        self.do_layer_distill = getattr(config, "do_layer_distill", False)
        self.do_emb_distill = getattr(config, "do_emb_distill", False)
        self.do_mha_layer_distill = getattr(config, "do_mha_layer_distill", False)

        if self.do_mha_layer_distill:
            self.mha_layer_transformation = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.mha_layer_transformation = None

        if self.do_layer_distill:
            self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.layer_transformation = None
        if self.do_emb_distill:
            self.emb_transformation = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.emb_transformation = None

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            logging=False,
            dims_boundary=None,
            ov_dims_boundary=None,
            qk_z=None,
            vo_z=None,
            intermediate_z=None,
            head_z=None,
            head_layer_z=None,
            mlp_z=None,
            inference=False,
            hidden_z=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logging=logging,
            dims_boundary=dims_boundary,
            ov_dims_boundary=ov_dims_boundary,
            qk_z=qk_z,
            vo_z=vo_z,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            inference=inference,
            hidden_z=hidden_z
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prune_indices(self, indices_to_prune: Dict[int, List[int]], vo_indices_to_prune: Dict[int, List[int]] = None,
                      q_input_index=None, k_input_index=None,
                      v_input_index=None, o_output_index=None):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (:obj:`int`) and associated values being the list
                of heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will
                prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        self.base_model._prune_indices(indices_to_prune, vo_indices_to_prune, q_input_index, k_input_index,
                                       v_input_index, o_output_index)


class NewRobertaBertForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = NewRobertaModel(config)

        self.do_layer_distill = getattr(config, "do_layer_distill", False)
        self.do_emb_distill = getattr(config, "do_emb_distill", False)
        self.do_mha_layer_distill = getattr(config, "do_mha_layer_distill", False)

        if self.do_mha_layer_distill:
            self.mha_layer_transformation = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.mha_layer_transformation = None

        if self.do_layer_distill:
            self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.layer_transformation = None
        if self.do_emb_distill:
            self.emb_transformation = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.emb_transformation = None

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            logging=False,
            dims_boundary=None,
            ov_dims_boundary=None,
            qk_z=None,
            vo_z=None,
            intermediate_z=None,
            head_z=None,
            head_layer_z=None,
            mlp_z=None,
            inference=False,
            hidden_z=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logging=logging,
            dims_boundary=dims_boundary,
            ov_dims_boundary=ov_dims_boundary,
            qk_z=qk_z,
            vo_z=vo_z,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            inference=inference,
            hidden_z=hidden_z
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        prediction_scores = prediction_scores[labels != -100]
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prune_indices(self, indices_to_prune: Dict[int, List[int]], vo_indices_to_prune: Dict[int, List[int]] = None,
                      q_input_index=None, k_input_index=None,
                      v_input_index=None, o_output_index=None):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (:obj:`int`) and associated values being the list
                of heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will
                prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        self.base_model._prune_indices(indices_to_prune, vo_indices_to_prune, q_input_index, k_input_index,
                                       v_input_index, o_output_index)


class NewRobertaEmbeddings(RobertaEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, hidden_z=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # Copied from transformers.modeling_bert.BertEmbeddings.forward
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)
        embeddings = self.LayerNorm(embeddings, hidden_z)
        embeddings = self.dropout(embeddings)

        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)
        return embeddings


class NewRobertaModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = NewRobertaEncoder(config)
        self.embeddings = NewRobertaEmbeddings(config)
        self.embedding_transformer = None
        if getattr(config, "transform_embedding", False):
            self.embedding_transformer = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            logging=False,
            qk_z=None,
            vo_z=None,
            intermediate_z=None,
            head_z=None,
            mlp_z=None,
            head_layer_z=None,
            dims_boundary=None,
            ov_dims_boundary=None,
            inference=False,
            hidden_z=None
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            hidden_z=hidden_z
        )

        if self.embedding_transformer is not None:
            embedding_output = self.embedding_transformer(embedding_output)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            qk_z=qk_z,
            vo_z=vo_z,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            inference=inference,
            hidden_z=hidden_z
        )
        # self.encoder_outputs = encoder_outputs
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return NewBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            attention_layers=encoder_outputs.attention_layers
        )

    def _prune_indices(self, qk_indice_to_prune, vo_indice_to_prune=None, q_input_index=None, k_input_index=None,
                       v_input_index=None, o_output_index=None):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, indices in qk_indice_to_prune.items():
            qk_indices = indices
            if vo_indice_to_prune is not None:
                vo_indices = vo_indice_to_prune[layer]
            else:
                vo_indices = None
            self.encoder.layer[layer].attention.prune_indices(qk_indices, vo_indices)


class NewRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([NewRobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            dims_boundary=None,
            ov_dims_boundary=None,
            qk_z=None,
            vo_z=None,
            intermediate_z=None,
            head_z=None,
            mlp_z=None,
            head_layer_z=None,
            inference=False,
            hidden_z=None
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_attention_outputs = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    dims_boundary=dims_boundary[i] if dims_boundary is not None else None,
                    ov_dims_boundary=ov_dims_boundary[i] if ov_dims_boundary is not None else None,
                    qk_z=qk_z[i] if qk_z is not None else None,
                    vo_z=vo_z[i] if vo_z is not None else None,
                    intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                    head_z=head_z[i] if head_z is not None else None,
                    mlp_z=mlp_z[i] if mlp_z is not None else None,
                    head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                    inference=inference,
                    hidden_z=hidden_z
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                all_attention_outputs = all_attention_outputs + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_attention_outputs] if v is not None)
        return NewBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions,
            attention_layers=all_attention_outputs
        )


class NewRobertaLayer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = NewRobertaAttention(config)
        self.output = NewRobertaOutput(config)
        self.config = config

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            dims_boundary=None,
            ov_dims_boundary=None,
            qk_z=None,
            vo_z=None,
            intermediate_z=None,
            head_z=None,
            mlp_z=None,
            head_layer_z=None,
            inference=False,
            hidden_z=None
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            dims_boundary=dims_boundary,
            ov_dims_boundary=ov_dims_boundary,
            qk_z=qk_z,
            vo_z=vo_z,
            head_z=head_z,
            head_layer_z=head_layer_z,
            inference=inference,
            hidden_z=hidden_z
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # self.attention_output = attention_output

        if self.intermediate.dense is None:
            layer_output = attention_output
        else:
            self.intermediate_z = intermediate_z
            self.mlp_z = mlp_z
            self.inference = inference
            self.hidden_z = hidden_z
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
        # self.layer_output = layer_output
        outputs = (layer_output,) + outputs + (attention_output,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        if self.intermediate_z is not None:
            intermediate_output = intermediate_output.mul(self.intermediate_z)
        layer_output = self.output(intermediate_output, attention_output, self.mlp_z, self.hidden_z,
                                   inference=self.inference)
        return layer_output


class NewRobertaAttention(RobertaAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = NewRobertaSelfAttention(config)
        self.output = NewRobertaSelfOutput(config)

        self.config = config

    def prune_indices(self, qk_index, vo_index=None):

        if type(qk_index) == list or isinstance(qk_index, np.ndarray):
            qk_index = torch.LongTensor(qk_index)

        if type(vo_index) == list or isinstance(vo_index, np.ndarray):
            vo_index = torch.LongTensor(vo_index)

        if vo_index is None:
            vo_index = qk_index

        # Prune linear layers
        if len(qk_index) == 0:
            self.self.query = None
            self.self.key = None
        else:
            self.self.query = prune_linear_layer(self.self.query, qk_index)
            self.self.key = prune_linear_layer(self.self.key, qk_index)

        if len(vo_index) == 0:
            self.self.value = None
            self.output.dense = None
        else:
            self.self.value = prune_linear_layer(self.self.value, vo_index)
            self.output.dense = prune_linear_layer(self.output.dense, vo_index, dim=1)

        # print(f"query: {self.self.query.weight.shape if self.self.query is not None else [0, 0]}")
        # print(f"key: {self.self.key.weight.shape if self.self.key is not None else [0, 0]}")
        # print(f"value: {self.self.value.weight.shape if self.self.value is not None else [0, 0]}")
        # print(f"dense: {self.output.dense.weight.shape if self.output.dense is not None else [0, 0]}")

    def prune_heads(self, heads):
        len_heads = len(heads)
        if len_heads == 0:
            return

        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        if len(index) == 0:
            self.self.query = None
            self.self.key = None
            self.self.value = None
            self.output.dense = None
        else:
            self.self.query = prune_linear_layer(self.self.query, index)
            self.self.key = prune_linear_layer(self.self.key, index)
            self.self.value = prune_linear_layer(self.self.value, index)
            self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            dims_boundary=None,
            ov_dims_boundary=None,
            qk_z=None,
            vo_z=None,
            head_z=None,
            head_layer_z=None,
            mlp_z=None,
            inference=False,
            hidden_z=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            qk_z=qk_z,
            vo_z=vo_z,
            head_z=head_z,
            head_layer_z=head_layer_z
        )

        attention_output = self.output(self_outputs[0], hidden_states, head_layer_z=head_layer_z, hidden_z=hidden_z,
                                       inference=inference)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class NewRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, dims_boundary=None, pad=False, type="qk"):
        # head int setting
        # semi head setting, the rest of the head dimensions has been padded
        x_shape = x.size()
        last_dim = x_shape[-1]
        size_per_head = last_dim // self.num_attention_heads
        new_x_shape = x_shape[:-1] + (self.num_attention_heads, size_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                head_mask=None,
                dims_boundary=None,
                ov_dims_boundary=None,
                qk_z=None,
                vo_z=None,
                head_z=None,
                head_layer_z=None,
                ):
        if self.value is None:
            return (None, None) if output_attentions else (None,)

        # if self.value.weight.sum() == 0: # only for updated full model
        #     return (None, None) if output_attentions else (None, )

        if self.query is None:
            mixed_query_layer = None
        else:
            query_hidden_states = hidden_states
            mixed_query_layer = self.query(query_hidden_states)

            if getattr(self, "qk_s", None) is not None:
                mixed_query_layer = mixed_query_layer.mul(self.qk_s)
            if qk_z is not None:
                mixed_query_layer = mixed_query_layer.mul(qk_z)
            # self.mixed_query_layer = mixed_query_layer

            key_hidden_states = hidden_states
            mixed_key_layer = self.key(key_hidden_states)

            value_hidden_states = hidden_states
            mixed_value_layer = self.value(value_hidden_states)
            if vo_z is not None:
                mixed_value_layer = mixed_value_layer.mul(vo_z)

        # batch * sequence_length * dim => batch * sequence_length
        batch_size, seq_length, _ = hidden_states.shape

        if not hasattr(self, "ones"):
            self.ones = torch.ones(batch_size, seq_length, seq_length).float().to(hidden_states.device)

        if mixed_query_layer is not None:
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        else:
            attention_scores = self.ones[:batch_size]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(mixed_value_layer, dims_boundary=ov_dims_boundary, pad=False, type="vo")
        context_layer = torch.matmul(attention_probs, value_layer)
        if head_z is not None:
            context_layer *= head_z

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (context_layer.shape[-1] * context_layer.shape[-2],)
        context_layer = context_layer.view(*new_context_layer_shape)

        # from https://github.com/pmichel31415/pytorch-pretrained-BERT/blob/paul/pytorch_pretrained_bert/modeling.py line 306
        if getattr(self, "vo_s", None) is not None:
            context_layer = context_layer.mul(self.vo_s)
        # self.context_layer = context_layer

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class NewRobertaSelfOutput(RobertaSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor, head_layer_z=None, hidden_z=None, inference=False):
        if hidden_states is None:
            return input_tensor
        batch_size, seq_length, _ = input_tensor.shape
        hidden_states = self.dense(hidden_states)
        if head_layer_z is not None:
            hidden_states = hidden_states.mul(head_layer_z)
        if not inference and hidden_states.sum().eq(0).item():
            hidden_states = hidden_states + input_tensor
        else:
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor, hidden_z)
            # self.hidden_states = hidden_states
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        return hidden_states


class NewRobertaOutput(RobertaOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor, mlp_z, hidden_z=None, inference=False):
        hidden_states = self.dense(hidden_states)
        if mlp_z is not None:
            hidden_states *= mlp_z
        if not inference and hidden_states.sum().eq(0).item():
            return hidden_states + input_tensor
        else:
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor, hidden_z)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        return hidden_states


@dataclass
class NewQuestionAnsweringModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_layers: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class NewBaseModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_layers: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class NewBaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_layers: Optional[Tuple[torch.FloatTensor]] = None
