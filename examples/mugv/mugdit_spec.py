# SPDX-License-Identifier: Apache-2.0

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

from mcore_patch.attention import (
    CrossAttentionQKNorm,
    CrossAttentionQKNormSubmodules,
    SelfAttention,
)
from mcore_patch.fusions.fused_bias_dropout_gate import get_bias_dropout_gate_add
from mugdit_layer import MUGDiTLayer, MUGDiTLayerSubmodules
from mugdit_modulate import ModulateLayerNorm, ScaleShiftTable, WrappedTorchLayerNorm

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules  # SelfAttention,
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

LNImpl = WrappedTorchLayerNorm
# try:
#     import apex

#     from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

#     HAVE_APEX = True
#     LNImpl = FusedLayerNorm
# except ImportError:
#     import warnings

#     # from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm
#     from mugdit_modulate import WrappedTorchLayerNorm

#     warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
#     LNImpl = WrappedTorchLayerNorm


def get_mugdit_layer_spec(qk_layernorm: bool = True, normalization: str = "LayerNorm") -> ModuleSpec:
    """MUGDiT Encoder TE spec (uses Transformer Engine components)."""
    attn_mask_type = AttnMaskType.no_mask
    mlp = ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        )
    if normalization == "LayerNorm":
        norm = LNImpl
    elif normalization == "RMSNorm":
        norm = TENorm
    else:
        raise RuntimeError("unknown normalization", normalization)

    return ModuleSpec(
        module=MUGDiTLayer,
        submodules=MUGDiTLayerSubmodules(
            input_layernorm=ModulateLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=norm if qk_layernorm else IdentityOp,
                    k_layernorm=norm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_gate_add,
            pre_cross_attn_layernorm=WrappedTorchLayerNorm, # Add pre_cross_attn_layernorm
            cross_attention=ModuleSpec(
                module=CrossAttentionQKNorm,
                params={"attn_mask_type": attn_mask_type},
                submodules=CrossAttentionQKNormSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=norm if qk_layernorm else IdentityOp,
                    k_layernorm=norm if qk_layernorm else IdentityOp,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add, # no gate here
            pre_mlp_layernorm=ModulateLayerNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_gate_add,
            scale_shift_table=ScaleShiftTable,
        )
    )


def get_mugdit_layer_spec_local(qk_layernorm: bool = True, normalization: str = "LayerNorm") -> ModuleSpec:
    """MUGDiT Encoder local spec (uses only megatron core components)."""
    attn_mask_type = AttnMaskType.no_mask
    mlp = ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ),
        )
    if normalization == "LayerNorm":
        norm = LNImpl
    elif normalization == "RMSNorm":
        norm = TENorm
    else:
        raise RuntimeError("unknown normalization", normalization)

    return ModuleSpec(
        module=MUGDiTLayer,
        submodules=MUGDiTLayerSubmodules(
            input_layernorm=ModulateLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_gate_add,
            pre_cross_attn_layernorm=WrappedTorchLayerNorm, # Add pre_cross_attn_layernorm
            cross_attention=ModuleSpec(
                module=CrossAttentionQKNorm,
                params={"attn_mask_type": attn_mask_type},
                submodules=CrossAttentionQKNormSubmodules(
                    linear_q=ColumnParallelLinear,
                    linear_kv=ColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add, # no gate here
            pre_mlp_layernorm=ModulateLayerNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_gate_add,
            scale_shift_table=ScaleShiftTable,
        )
    )
