# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
# This file is modified from Megatron-Core's rotary_pos_embedding.py to support:
# - 3D Rotary Position Embeddings for spatiotemporal video modeling

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from einops import rearrange

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_block import TransformerBlock

import logging

import torch
from torch import Tensor, nn

from megatron.core import parallel_state
from megatron.core.utils import is_te_min_version

logger = logging.getLogger(__name__)

# Prefer fused RoPE from Apex as we need the `transpose_output_memory` argument for the bshd trick.
# See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/merge_requests/2469.
try:
    from apex.transformer.functional import fused_apply_rotary_pos_emb
except ImportError:
    try:
        from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb
    except:
        fused_apply_rotary_pos_emb = None


try:
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb_thd
except ImportError:
    try:
        from apex.transformer.functional import fused_apply_rotary_pos_emb_thd
    except ImportError:
        fused_apply_rotary_pos_emb_thd = None


try:
    from flash_attn.layers.rotary import apply_rotary_emb as apply_rotary_emb_flash
except ImportError:
    apply_rotary_emb_flash = None


__all__ = ['RotaryEmbedding3D', 'apply_rotary_pos_emb']


def get_pos_emb_on_this_cp_rank(pos_emb: Tensor, seq_dim: int) -> Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2):])
    return pos_emb


class RotaryEmbedding3D(nn.Module):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from transformer config
        space_end (int):
        time_end (int):
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to 10000.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly on the GPU. Defaults to False
        scale_watershed (float):
        timestep (float):
        enable_start_end_tokens (bool): enable to concat two learnable parameter tokens before and after the input clip latents to inform DiT the location information of the current video clip in the whole original video.
    """

    def __init__(
        self,
        kv_channels: int,
        space_end: int,
        time_end: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = 1.0,
        rotary_base: int = 10000,
        use_cpu_initialization: bool = False,
        scale_watershed: float = 1.0,
        timestep: float = 1.0,
        enable_start_end_tokens: bool = True,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.time_end = time_end
        self.space_end = space_end
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor or 1.0

        assert not dim % 6, f'dim {dim} not support 3-dimentional'
        if timestep < scale_watershed:
            linear_factor = self.seq_len_interpolation_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = self.seq_len_interpolation_factor

        rotary_base = rotary_base * ntk_factor
        device = 'cpu' if use_cpu_initialization else torch.cuda.current_device(
        )
        self.inv_freq = 1.0 / (rotary_base**(
            torch.arange(0, dim, 6, dtype=torch.float64, device=device) /
            dim)) / linear_factor

        # NOTE: Precompute space_freqs and time_freqs with respect to the implementation of Open-Sora
        # Megatron-LM choose calculate freqs in forward, I don't know why yet.
        space_t = torch.arange(space_end, device=device,
                               dtype=torch.float64)  # type: ignore
        time_t = torch.arange(time_end, device=device,
                              dtype=torch.float64)  # type: ignore
        self.space_freqs = torch.outer(space_t, self.inv_freq)  # type: ignore
        self.time_freqs = torch.outer(time_t, self.inv_freq)  # type: ignore

        self.enable_start_end_tokens = enable_start_end_tokens

    def retrieve_freqs(self, H, W, seq_t_indices, oh=0, ow=0, ot=0):
        B, T = seq_t_indices.shape
        C = self.space_freqs.shape[-1]
        h_freq_cis = self.space_freqs[oh:H + oh].reshape(1, H, 1, 1, C).expand(
            T, H, W, B, C)
        w_freq_cis = self.space_freqs[ow:W + ow].reshape(1, 1, W, 1, C).expand(
            T, H, W, B, C)
        # XXX: seq_t_indices may be larger than max RoPE length of time dimension, I don't know why yet.
        # Here we try to clamp seq_t_indices accordingly, but it's not a good idea to do it in the forward pass.
        seq_t_indices = seq_t_indices.clamp(0, self.time_end - 1 - ot)
        t_freq_cis = self.time_freqs[seq_t_indices + ot].transpose(
            1, 0).reshape(T, 1, 1, B, C).expand(T, H, W, B, C)
        freq_cis = torch.cat([h_freq_cis, w_freq_cis, t_freq_cis], dim=-1)
        if not self.enable_start_end_tokens:
            return freq_cis.reshape(T * H * W, B, -1)
        start_token_cis = freq_cis[0, 0, 0, :].unsqueeze(0)
        end_token_cis = freq_cis[-1, -1, -1, :].unsqueeze(0)
        freq_cis = freq_cis[1:-1].reshape((T - 2) * H * W, B, -1)
        freq_cis = torch.cat([start_token_cis, freq_cis, end_token_cis], dim=0)
        return freq_cis

    def forward(
        self,
        max_seq_len_h: int,
        max_seq_len_w: int,
        seq_t_indices: Tensor,
        offset_h: int = 0,
        offset_w: int = 0,
        offset_t: int = 0,
    ) -> Tensor:
        """Forward pass of RoPE embedding 3D version.
        Args:
            max_seq_len_h (int): Maximum size of sequence at height dimension
            max_seq_len_w (int): Maximum size of sequence at width dimension
            seq_t_indices (Tensor): indices for time dimension to recognize each frame location
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        if self.inv_freq.device.type == 'cpu':
            # move `inv_freq` to GPU once at the first micro-batch forward pass
            self.inv_freq = self.inv_freq.to(
                device=torch.cuda.current_device())
            self.space_freqs = self.space_freqs.to(
                device=torch.cuda.current_device())
            self.time_freqs = self.time_freqs.to(
                device=torch.cuda.current_device())

        # retrieve freqs with respect to the implementation of Open-Sora
        freqs = self.retrieve_freqs(
            max_seq_len_h,
            max_seq_len_w,
            seq_t_indices,
            offset_h,
            offset_w,
            offset_t,
        )

        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            S, B, _ = freqs.shape
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)),
                              dim=-1).view(S, B, -1)
        # emb [seq_length, .., dim]
        emb = emb[:, :, None, :]
        if parallel_state.get_context_parallel_world_size() > 1:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0)
        return emb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f'{prefix}inv_freq', None)
        return super()._load_from_state_dict(state_dict, prefix, *args,
                                             **kwargs)

    def get_rotary_seq_len(
        self,
        inference_params,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
    ) -> float:
        """Function to get the rotary sequence length.

        Args:
            inference_params : Used during Inference time
            transformer (TransformerBlock): The transformer block (decoder/encoder) used by the model
            transformer_input (Tensor): _description_
            transformer_config (TransformerConfig): Transformer config used by the model

        Returns:
            float: The rotary sequence length
        """
        if inference_params is not None:
            rotary_seq_len = inference_params.max_sequence_length
        else:
            if transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.size(0)
            else:
                rotary_seq_len = transformer_input.size(0)

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        rotary_seq_len *= transformer_config.context_parallel_size

        return rotary_seq_len


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]
    in_type = t.dtype

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale)
    sin_ = (torch.sin(freqs) * mscale)

    t = (t.to(cos_) * cos_) + (_rotate_half(t.to(sin_), rotary_interleaved) * sin_)
    return torch.cat((t.to(in_type), t_pass), dim=-1)


def _get_thd_freqs_on_this_cp_rank(cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor) -> Tensor:
    if cp_size > 1:
        cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
        return torch.cat(
            [
                freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg],
            ]
        )
    else:
        return freqs[: x.size(0)]


def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cu_seqlens = cu_seqlens // cp_size
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

    return torch.cat(
        [
            _apply_rotary_pos_emb_bshd(
                x.unsqueeze(1),
                _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs),
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
            )
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)


def apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """

    if config.apply_rope_fusion:
        if cu_seqlens is None:
            assert fused_apply_rotary_pos_emb is not None, "apply_rope_fusion is not available."
            return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
        else:
            assert fused_apply_rotary_pos_emb_thd is not None, "apply_rope_fusion is not available."
            cp_size = parallel_state.get_context_parallel_world_size()
            if cp_size > 1:
                if not is_te_min_version("1.11.0", check_equality=False):
                    raise ValueError("Only TE >= 1.12 supports RoPE fusion for THD format with CP.")
                return fused_apply_rotary_pos_emb_thd(
                    t,
                    cu_seqlens,
                    freqs,
                    cp_size=cp_size,
                    cp_rank=parallel_state.get_context_parallel_rank(),
                )
            else:
                return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
    else:
        if cu_seqlens is None:
            return _apply_rotary_pos_emb_bshd(
                t,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale,
            )
        else:
            return _apply_rotary_pos_emb_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale,
            )


def apply_rotary_pos_emb_with_cos_sin(
    t: Tensor, cos: Tensor, sin: Tensor, rotary_interleaved: bool = False
) -> Tensor:
    """
    This function applies rotary positional embedding to the target tensor t
    using precomputed cos and sin of size (seq_len, d_rot / 2)
    """
    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)

    if apply_rotary_emb_flash is None:
        # Combine cos and sin into freqs
        freqs = torch.stack([cos, sin], dim=-1).flatten(start_dim=-2)

        # Expand freqs to match t's shape
        while freqs.dim() < t.dim():
            freqs = freqs.unsqueeze(1)
        freqs = freqs.expand(t.shape[:-1] + (-1,))

        y = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=rotary_interleaved,
            multi_latent_attention=False,
            mscale=1.0,
        )
    else:
        # Use Flash Attention's optimized kernel for rotary embedding
        t = t.permute(1, 0, 2, 3)
        y = apply_rotary_emb_flash(t, cos, sin, rotary_interleaved)
        y = y.permute(1, 0, 2, 3)

    return y

# HACK: monkey patch float32 precison implementation for megatron default low precision rotary_pos_embedding
import megatron.core.models.common.embeddings.rope_utils

megatron.core.models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd = _apply_rotary_pos_emb_bshd
