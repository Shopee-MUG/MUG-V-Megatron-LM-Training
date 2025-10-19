# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Zhongyi Fan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is modified from megatron/core/transformer/transformer_layer.py to support:
# - MUGDiT layers

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mcore_patch.fusions.fused_bias_dropout_gate import get_bias_dropout_gate_add
from mcore_patch.transformer_layer import TransformerLayer
from mugdit_modulate import t2i_modulate
from torch import Tensor
from torch.cuda import default_stream
from torch.nn.parameter import Parameter

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.jit import jit_fuser
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.utils import (
    deprecate_inference_params,
    is_te_min_version,
    log_single_rank,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    import apex

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm


@dataclass
class MUGDiTLayerSubmodules(TransformerLayerSubmodules):
    """MUGDiTLayer submodule type class.

    """
    scale_shift_table: Union[ModuleSpec, type] = IdentityFuncOp


class MUGDiTLayer(TransformerLayer):
    """A single MUGDiT layer.

    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MUGDiTLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            model_comm_pgs=model_comm_pgs,
            vp_stage=vp_stage,
        )

        # BUG: We have moved the scale-shift table to the super class i.e. Transformer layer.
        # This ensures that the parameter declaration order aligns with the forward pass usage order,
        # preventing issues such as incorrect parameter gathering and checkpointing logic bugs.
        # This behavior will be revised once we merge the fused BiasAdd-Dropout-Residual-Modulate-LayerNorm.
        # At that point, the scale-shift table will be integrated into the standard Transformer block in a more logical and cohesive manner.

        # # Init scale_shift_table
        # self.scale_shift_table = build_module(
        #     submodules.scale_shift_table,
        #     self.config.hidden_size,
        #     6,
        #     config=self.config,
        # )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        mask_timesteps=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the transformer layer.

        This method implements the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            mask_timesteps: mask_selected_timesteps. [batch, 6, time, 1, hidden]
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_params (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                output (Tensor): Transformed hidden states of shape [s, b, h].
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        L, B, H = hidden_states.shape

        # Residual connection.
        residual = hidden_states

        # prepare MUGDiT modulate parameters
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.scale_shift_table(mask_timesteps).unbind(1)

        # Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states, shift_msa, scale_msa)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=None,
            sequence_len_offset=sequence_len_offset,
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, gate_msa, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states
        # residual = input_layernorm_output

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        pre_cross_attn_layernorm_output = rearrange(pre_cross_attn_layernorm_output, "L B C -> (B L) C")
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params, # NOTE: We use varlen_qkv for cross-attention.
        )
        attention_output_with_bias = (
            rearrange(attention_output_with_bias[0], "(B L) 1 C -> L B C", L=L),
            attention_output_with_bias[1],
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states
        # residual = pre_cross_attn_layernorm_output

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states, shift_mlp, scale_mlp)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, gate_mlp, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # CUDA graph requires returned values to be Tensors
        if self.config.external_cuda_graph and self.training:
            return output
        return output, context
