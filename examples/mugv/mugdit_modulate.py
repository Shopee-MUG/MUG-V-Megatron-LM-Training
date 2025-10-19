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

import math
import warnings
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.normalization import FusedLayerNorm
from einops import rearrange
from torch import Tensor
from torch.cuda import default_stream
from torch.nn.parameter import Parameter

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.jit import jit_fuser
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
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
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

# try:
#     import apex

#     from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

#     HAVE_APEX = True
#     LNImpl = FusedLayerNorm
# except ImportError:
#     import warnings

#     from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

#     warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
#     LNImpl = WrappedTorchLayerNorm


@jit_fuser
def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class WrappedTorchLayerNorm(FusedLayerNorm):
    """ Adapt from megatron.core.transformer.torch_layer_norm.WrappedTorchLayerNorm,
        support affine and bias configurations as there is no weight and bias in mugdit layernorm.
        Update super class to FusedLayerNorm as HuggingFace version use apex.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = False,
        persist_layer_norm: bool = False,  ## TODO: unused arguments. See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/223
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",  # included to match TE interface
    ):
        self.config = config

        super().__init__(
            normalized_shape=hidden_size,  ## applied to last len(normalized_shape.size) dimensions
            eps=eps,
            elementwise_affine=elementwise_affine,
        )


class ModulateLayerNorm(WrappedTorchLayerNorm):
    """LayerNorm with post modulate

        We initialize the layernorm without parameters.
        PERF: In practice, we can fuse ScaleShiftTable into transformer engine layernorm to gain performance.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(ModulateLayerNorm, self).__init__(*args, elementwise_affine=False, bias=False, **kwargs)

    def forward(self, input: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        x = super().forward(input)
        return t2i_modulate(x, shift, scale)


class ScaleShiftTable(torch.nn.Module):
    """ScaleShiftTable without parallelism.

    Args:
        output_size: hidden_size of t_input.
        config: ModelParallelConfig object

    """

    def __init__(
        self,
        output_size: int,
        table_size: int,
        *,
        config: ModelParallelConfig,
    ):
        super(ScaleShiftTable, self).__init__()

        self.output_size = output_size
        self.table_size = table_size
        self.config = config

        # Parameters bias
        if config.use_cpu_initialization:
            self.bias = Parameter(torch.empty((self.table_size, self.output_size), dtype=config.params_dtype))
        else:
            self.bias = Parameter(
                torch.empty(
                    (self.table_size, self.output_size),
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )

        if config.perform_initialization:
            # random init scale shift table
            with torch.no_grad():
                self.bias.data.copy_(torch.randn(self.table_size, self.output_size) / self.output_size**0.5)
        # setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
        setattr(self.bias, 'sequence_parallel', False)

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

    def forward(self, t_input) -> Tuple:
        """Forward of ScaleShiftTable

        Args:
            t_input: 3D tensor whose order of dimension is [length, table_size, batch, hidden], i.e. [L, table_size, B, H]

        Returns:
            - output: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) each holds shape: [length, batch, hidden]
        """
        return (t_input.double() + self.bias[None, :, None, :].double()).to(t_input.dtype)

    def set_extra_state(self, state: Any):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

