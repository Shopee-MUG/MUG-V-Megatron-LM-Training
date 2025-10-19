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
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mugdit_modulate import ScaleShiftTable, WrappedTorchLayerNorm, t2i_modulate
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


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        drop_probs = drop
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        # TODO: move padding function to the dataloader to accelerate forward pass
        # padding
        _, _, D, H, W = x.size()
        pad_d = self.patch_size[0] - D % self.patch_size[0] if D % self.patch_size[0] != 0 else 0
        pad_h = self.patch_size[1] - H % self.patch_size[1] if H % self.patch_size[1] != 0 else 0
        pad_w = self.patch_size[2] - W % self.patch_size[2] if W % self.patch_size[2] != 0 else 0
        # Pad format: (width_left, width_right, height_left, height_right, depth_left, depth_right)
        assert pad_d == 0 and pad_h == 0 and pad_w == 0, f"{pad_d, pad_h, pad_w} should be 0"
        if not (pad_d == 0 and pad_h == 0 and pad_w == 0):
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        x = self.proj(x)  # (B C T H W)
        x = rearrange(x, "b c t h w -> b (t h w) c") # B C T H W -> B N C
        if self.norm is not None:
            x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, norm_layer=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        if norm_layer is not None:
            self.norm = norm_layer(hidden_size)
        else:
            self.norm = None

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float64) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        if self.norm is not None:
            t_emb = self.norm(t_emb)
        return t_emb


class SizeEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, norm_layer=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        if norm_layer is not None:
            self.norm = norm_layer(hidden_size)
        else:
            self.norm = None

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float64) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, s):
        """
        Args:
            s (Tensor): shape [batch, 1].

        """
        assert s.ndim == 2
        b, dims = s.shape
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        if self.norm is not None:
            s_emb = self.norm(s_emb)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.hidden_size)
        return s_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels / captions into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        config: TransformerConfig,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=lambda : nn.GELU(approximate="tanh"),
        token_num=120,
        norm_layer=None,
    ):
        super().__init__()
        self.config = config
        self.uncond_prob = uncond_prob
        self.uncond_prob_i2v = 0.5
        self.uncond_prob_t2v = 0.1
        self.use_dropout = self.uncond_prob > 0
        self.token_num = token_num
        self.rand_span = 50
        if self.use_dropout:
            self.register_buffer(
                "drop_emb",
                torch.randn(token_num, in_channels) / in_channels**0.5,
            )
        if norm_layer:
            self.pre_norm = WrappedTorchLayerNorm(config, in_channels, elementwise_affine=False, bias=False)
        else:
            self.pre_norm = None
        self.proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        if norm_layer:
            self.post_norm = WrappedTorchLayerNorm(config, hidden_size, elementwise_affine=False, bias=False)
        else:
            self.post_norm = None

    def token_drop(self, caption, caption_seqlens, force_drop_ids=None, task=None):
        """
        Drops labels / captions to enable classifier-free guidance, at token level.
        """
        uncond_prob = self.uncond_prob
        if task is not None:
            if task == 'i2v':
                uncond_prob = self.uncond_prob_i2v
            elif task == 't2v':
                uncond_prob = self.uncond_prob_t2v
            else:
                print(f'task {task} not supported')

        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device='cuda') < uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        pos_ids = torch.cat([torch.arange(i) for i in caption_seqlens])
        caption = torch.where(drop_ids[:, None], self.drop_emb[pos_ids], caption)
        return caption, caption_seqlens

    def sample_drop(self, caption, caption_seqlens, force_drop_ids=None):
        """
        Drops labels / captions to enable classifier-free guidance, at sample level.
        """
        drop_samples = torch.rand(caption_seqlens.shape, device='cuda') < self.uncond_prob

        new_caption_seqlens = caption_seqlens.clone()
        new_caption = []
        for i, seqlen in enumerate(caption_seqlens):
            if drop_samples[i].item() == 0:
                new_caption.append(caption[:seqlen])
            else:
                rand_token_num = torch.randint(seqlen - self.rand_span, seqlen + self.rand_span + 1, (1,), device=caption.device)
                rand_token_num = torch.clamp(rand_token_num, min=1, max=self.token_num)
                new_caption_seqlens[i] = rand_token_num
                new_caption.append(self.drop_emb[:rand_token_num])

            caption = caption[seqlen:]

        new_caption = torch.cat(new_caption, dim=0)
        return new_caption, new_caption_seqlens

    def forward(self, caption, caption_seqlens, train, force_drop_ids=None, task=None):
        if (train and self.use_dropout):
            caption, caption_seqlens = self.token_drop(caption, caption_seqlens, force_drop_ids)
        if self.pre_norm is not None:
            caption = self.pre_norm(caption)
        caption = self.proj(caption)
        if self.post_norm is not None:
            caption = self.post_norm(caption)
        if train and not next(self.parameters()).requires_grad:
            caption.requires_grad = True
        return caption, caption_seqlens


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, config):
        super().__init__()
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = ScaleShiftTable(hidden_size, table_size=2, config=config)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)

    def forward(self, x, mask_t_final):
        # mask_t_final: (L, 2, B, H)
        shift, scale = self.scale_shift_table(mask_t_final).unbind(1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
