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

import itertools
import logging
import math
from collections import namedtuple
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mcore_patch.rotary_pos_embedding_3d import RotaryEmbedding3D
from mugdit_block import MUGDiTBlock, MUGDiTBlockSubmodules
from mugdit_embed import (
    CaptionEmbedder,
    PatchEmbed3D,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
)
from mugdit_patchify import XEmbedder, XProjector
from torch import Tensor
from torch.cuda import default_stream
from torch.nn.parameter import Parameter

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.jit import jit_fuser
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.packed_seq_params import PackedSeqParams
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
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_viewless_tensor

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


class MUGDiT(VisionModule):
    """MUGDiT vision model.
    We bind rectified flow scheduler with model class to make Megatron-LM happy.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        patch_dim (Tuple[int, int, int]): Image patch size.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
        allow_missing_norm_checkpoint (bool): Allow some weights to be missing when loading a checkpoint. Default False.
    """

    def __init__(
        self,
        scheduler,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_dim: Tuple[int, int, int] = (1, 2, 2),
        position_embedding_type: Literal['learned_absolute', 'rope',
                                         'none'] = 'rope',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        allow_missing_norm_checkpoint: bool = False,
        enable_start_end_tokens: bool = True,
        share_embeddings_and_output_weights: bool = False,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> None:
        super().__init__(config=transformer_config)

        self.scheduler = scheduler
        self.model_type = ModelType.encoder_or_decoder

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(transformer_config,
                               locals(),
                               prefix=type(self).__name__)
        logging.getLogger(__name__).warning(
            'MUGDiT is under active development. It may be missing features and its methods may change.'
        )

        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim

        self.in_channels = transformer_config.in_channels
        self.out_channels = self.in_channels * 2

        self.caption_channels = transformer_config.caption_channels
        self.caption_dropout_prob = transformer_config.caption_dropout_prob
        self.model_max_length = transformer_config.model_max_length
        self.position_embedding_type = position_embedding_type

        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.pre_process = pre_process
        self.post_process = post_process
        self.enable_start_end_tokens = enable_start_end_tokens

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding3D(
                kv_channels=self.config.kv_channels,
                space_end=384,
                time_end=480,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
        else:
            assert self.position_embedding_type == 'rope', 'Currently only support rope'

        if self.pre_process:
            # self.x_embedder = XEmbedder(self.in_channels,
            #                             self.visual_hidden_size)
            self.x_embedder = PatchEmbed3D(patch_dim, self.in_channels,
                                           self.visual_hidden_size)
            # NOTE: Shall we init this two token with bias stype? In case of weight decay?
            if self.enable_start_end_tokens:
                self.start_token = nn.Parameter(
                    torch.empty(self.visual_hidden_size))
                self.end_token = nn.Parameter(
                    torch.empty(self.visual_hidden_size))
                nn.init.normal_(self.start_token, std=0.02)
                nn.init.normal_(self.end_token, std=0.02)
            else:
                logging.getLogger(__name__).warning(
                    f'MUGDiT without start end token')
                self.start_token = self.end_token = None

        # replicate in every pp rank - vpp rank
        self.fps_embedder = SizeEmbedder(self.visual_hidden_size)
        self.timestep_embedder = TimestepEmbedder(self.visual_hidden_size)
        self.timestep_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.visual_hidden_size,
                      6 * self.visual_hidden_size,
                      bias=True),
        )
        self.context_embedder = CaptionEmbedder(
            config=self.config,
            in_channels=self.caption_channels,
            hidden_size=self.visual_hidden_size,
            uncond_prob=self.caption_dropout_prob,
            token_num=self.model_max_length,
            norm_layer=self.config.caption_norm,
        )
        for name, param in itertools.chain(
                self.fps_embedder.named_parameters(),
                self.timestep_embedder.named_parameters(),
                self.timestep_block.named_parameters(),
                self.context_embedder.named_parameters(),
                self.context_embedder.named_buffers(),
        ):
            setattr(param, 'pipeline_parallel', True)

        self.decoder = MUGDiTBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            post_layer_norm=False,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        if self.post_process:
            self.final_layer = T2IFinalLayer(
                self.visual_hidden_size,
                np.prod(self.patch_dim),
                out_channels=self.out_channels,
                config=transformer_config,
            )
            # self.projector = XProjector(self.visual_hidden_size, self.out_channels)

        if allow_missing_norm_checkpoint:
            norm_param_names = self.collect_missing_init_params()
            logging.getLogger(__name__).warning(
                f'MUGDiT register load state_dict hook to allow missing param names: {norm_param_names}'
            )
            self.register_load_state_dict_post_hook(
                partial(_load_state_dict_hook_ignore_param_names,
                        norm_param_names))

    def collect_missing_init_params(self):
        caption_norm_emb = [
            f'context_embedder.{name}'
            for name in self.context_embedder.state_dict().keys()
            if 'norm' in name or 'drop_emb' in name
        ]
        pre_cross_attn_norm = [
            f'decoder.{name}' for name in self.decoder.state_dict().keys()
            if 'pre_cross_attn_layernorm' in name
        ]
        cross_attn_qk_norm = [
            f'decoder.{name}' for name in self.decoder.state_dict().keys()
            if 'cross_attention.q_layernorm' in name
            or 'cross_attention.k_layernorm' in name
        ]
        missing_keys = caption_norm_emb + pre_cross_attn_norm + cross_attn_qk_norm
        unexpected_keys = ['context_embedder.drop_emb']
        return missing_keys + unexpected_keys

    def freeze(self,
               freeze_context_embedder: bool = False,
               freeze_cross_attn: bool = False):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_context_embedder (bool): Freeze the Context Embedder module.
        """
        modules = []
        if freeze_context_embedder:
            modules.append(self.context_embedder)

        if freeze_cross_attn:
            for layer in self.decoder.layers:
                modules.append(layer.pre_cross_attn_layernorm)
                modules.append(layer.cross_attention)

        logging.getLogger(__name__).warning(f'freeze {modules=}')

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(
            input_tensor
        ) == 1, 'input_tensor should only be length 1 for MUGDiT until triton refactor is complete'

        # if input_tensor[0] is not None:
        #     print('[mugdit] pp received input tensor: ', input_tensor[0].shape,
        #           input_tensor[0].dtype)
        # else:
        #     print('[mugdit] pp received input tensor: ', input_tensor[0])

        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor],
        temporal_mask,
        context,
        context_seqlens,
        timestep,
        fps,
        frame_indices,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ):
        """Forward function of the MUGDiT Model. This function passes the input tensors
        through the several embedding layer and then the transformer with a final layer norm.

        Args:
            x (Tensor): input data (from vae encoder) of shape [batch, hidden=4, temporal, img_h, img_w]
            attention_mask (Tensor with dtype=bool): Attention mask to use. Not check yet.
            temporal_mask (Tensor with dtype=bool): Temporal mask to use.
            context (Tensor): packed text_embeding. [total_seqlen, hidden]
            context_seqlens (Tensor with dtype=torch.int64): Seqlens of context used for varlen flashattn. [batch_size, ].
            timestep (Tensor): [batch, 1].
            fps (Tensor): [batch, 1].

        Returns:
             (Tensor): output after final transformer block of shape [batch, sequence, hidden].
        """

        # NOTE: Compared to ViT and traditional Transformer, MUGDiT has heavier pre-processing, i.e. several embedding layers.
        #       So we can't guarantee that if it will affect the efficiency of pipeline parallelism.
        #       For example, the load imbalance in the pipeline communication group.
        #       According to the above, the pipeline parallelism in MUGDiT maybe not a good idea, but we try it now.

        # print('[mugdit] input: ', x.shape, x.dtype)
        # === get pos embed ===
        # assert x.shape[2] == 19, "currently only support t == 19 "
        # x = torch.cat([x[:, :, :1, :, :], x], dim=2)
        B, C, Tx, Hx, Wx = x.shape

        T, H, W = get_dynamic_size(Tx, Hx, Wx, self.patch_dim)
        S = H * W

        # === get fps embed ===
        fps = self.fps_embedder(fps.unsqueeze(1))

        # === get timestep embed ===
        t_0_timestep = torch.cat((timestep, torch.zeros_like(timestep)), dim=0)
        t_0_emb = self.timestep_embedder(t_0_timestep, dtype=x.dtype).reshape(
            2, B, -1)  # [2, B, C]
        t_0_emb += fps
        t_0_mlp = self.timestep_block(t_0_emb)

        # === get context embed ===
        task_counter = kwargs.get('task_counter', None)
        task = None
        if task_counter is not None:
            for k, v in task_counter.items():
                if v.item() > 0:
                    task = k
                    break
        # In case random drop changed the context_seqlens, we need to update it.
        context, context_seqlens = self.context_embedder(context,
                                                         context_seqlens,
                                                         train=self.training,
                                                         task=task)

        # === get x embed ===
        if self.pre_process:
            x = self.x_embedder(x)
            x = rearrange(x, 'B L C -> L B C')
            if self.config.sequence_parallel:
                x = tensor_parallel.scatter_to_sequence_parallel_region(x)
                if self.config.clone_scatter_output_in_embedding:
                    x = x.clone()

            if self.enable_start_end_tokens:
                # pad start and end token
                x = torch.cat(
                    [
                        self.start_token.view(1, 1, -1).expand(-1, B, -1),
                        x,
                        self.end_token.view(1, 1, -1).expand(-1, B, -1),
                    ],
                    dim=0,
                )

            # contiguous() call required as `rearrange` may sparsify the tensor and this breaks pipelining
            x = x.contiguous()
        else:
            x = None

        temporal_mask = temporal_mask.unsqueeze(2).expand(B, T, S)
        temporal_mask = rearrange(temporal_mask, 'B T S -> (T S) B')
        if self.config.sequence_parallel:
            temporal_mask = tensor_parallel.scatter_to_sequence_parallel_region(temporal_mask)
            if self.config.clone_scatter_output_in_embedding:
                temporal_mask = temporal_mask.clone()
            S = S // parallel_state.get_tensor_model_parallel_world_size()

        if self.enable_start_end_tokens:
            temporal_mask = torch.cat(
                [
                    torch.zeros_like(temporal_mask[:1, :]).view(1, B),
                    temporal_mask,
                    torch.zeros_like(temporal_mask[:1, :]).view(1, B),
                ],
                dim=0,
            )  # [1 + T * S // tp_size + 1, B]
        else:
            # remove the indices of start and end tokens in frame_indices
            frame_indices = frame_indices[:, 1:-1] - 1

        t_0_mlp = rearrange(t_0_mlp, 'D B (N C) -> D 1 N B C', N=6) # [2, 1(broadcast L), 6, B, C]
        mask_timesteps = torch.where(
            temporal_mask[:, None, :, None],
            t_0_mlp[0],
            t_0_mlp[1],
        )  # [T * S + 2, 6, B, C]
        # mask_timesteps = rearrange(mask_timesteps, 'B N L C -> L N B C', N=6)

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            # TODO: Adapt to 3-dim rotary_seq_len, i.e. H, W, T. Now assume no parallelism
            # rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            #     inference_params, self.decoder, x, self.config
            # )
            rotary_pos_emb = self.rotary_pos_emb(H, W, frame_indices)
            # print(f'[mugdit] rotary_pos_emb shape: {rotary_pos_emb.shape}')

        packed_seq_params_for_cross_attn = (
            self.make_packed_seq_params_for_context(x if self.pre_process else self.decoder.input_tensor, context_seqlens))
        x = self.decoder(
            hidden_states=x,
            attention_mask=attention_mask,
            context=context,
            context_mask=None,
            mask_timesteps=mask_timesteps,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params_for_cross_attn,
        )
        x = x.contiguous()
        if not self.post_process:
            return x

        mask_t_final = torch.where(
            temporal_mask[:, None, :, None],  # (L B) -> L, 1, B, C
            t_0_emb[0, None, None, :, :], # [1, 1(L), 1, B, C]
            t_0_emb[1, None, None, :, :],
        ) # [L, N, B, C]
        x = self.final_layer(x, mask_t_final)
        x = x[1:-1] if self.enable_start_end_tokens else x

        if self.config.sequence_parallel:
            x = tensor_parallel.gather_from_sequence_parallel_region(x)

        x = rearrange(x, 'L B C -> B L C')
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)
        x = x.contiguous()
        print('[mugdit] output: ', x.shape, x.dtype)
        return x
        # return x[:, :, 1:, :, :]

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        T_p, H_p, W_p = self.patch_dim
        x = rearrange(
            x,
            'B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)',
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    def make_packed_seq_params_for_context(self, query, context_seqlens):
        """
        Build cross-attention PackedSeqParams based on query and context_mask。

        Args:
            query: torch.Tensor, shape [L, B, C].
            context_mask: torch.Tensor, shape [B, max_context_length].

        Returns:
            PackedSeqParams
        """
        L, B, _ = query.shape
        cu_seqlens_q = torch.arange(0, (B + 1) * L,
                                    step=L,
                                    dtype=torch.int32,
                                    device=torch.cuda.current_device())
        max_seqlen_q = torch.tensor([L], dtype=torch.int32).cuda()

        seqlens_kv = context_seqlens.to(torch.int32)
        cu_seqlens_kv = torch.cat([
            torch.tensor([0], dtype=torch.int32).cuda(),
            seqlens_kv.cumsum(dim=0)
        ]).to(torch.int32)
        max_seqlen_kv = seqlens_kv.max(dim=0, keepdim=True)[0]
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            qkv_format='thd',
        )
        return packed_seq_params

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: tuple = (),
            metadata: Optional[Dict] = None) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(
            prefix, sharded_offsets, metadata)
        for module in [
                'fps_embedder', 'timestep_embedder', 'timestep_block',
                'context_embedder'
        ]:
            for param_name, param in getattr(self, module).named_parameters():
                weight_key = f'{prefix}{module}.{param_name}'
                self._set_embedder_weights_replica_id(param,
                                                      sharded_state_dict,
                                                      weight_key)
            for param_name, param in getattr(self, module).named_buffers():
                weight_key = f'{prefix}{module}.{param_name}'
                self._set_embedder_weights_replica_id(param,
                                                      sharded_state_dict,
                                                      weight_key)
        return sharded_state_dict

    def _set_embedder_weights_replica_id(self, tensor: Tensor,
                                         sharded_state_dict: ShardedStateDict,
                                         embedder_weight_key: str) -> None:
        """set replica ids of the weights in t_embedder for sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            weight_key (str): key of the weight in the state dict.
                This entry will be replaced with a tied version

        Returns: None, acts in-place
        """
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank() or 0
        vpp_world = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if embedder_weight_key in sharded_state_dict:
            del sharded_state_dict[embedder_weight_key]
        replica_id = (
            tp_rank,
            (vpp_rank + pp_rank * vpp_world),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )
        sharded_state_dict[
            embedder_weight_key] = make_sharded_tensor_for_checkpoint(
                tensor=tensor,
                key=embedder_weight_key,
                replica_id=replica_id,
                allow_shape_mismatch=False,
            )
        # print(f'replica_id: {replica_id} tp_rank: {tp_rank}, vpp_rank: {vpp_rank}, pp_rank: {pp_rank}')
        # print(f'shard_tensor: {sharded_state_dict[embedder_weight_key]}')


@lru_cache(maxsize=None)
def get_dynamic_size(
    Tx: int, Hx: int, Wx: int,
    patch_size: Tuple[int, int, int] = (1, 2, 2)) -> Tuple[int, int, int]:
    # assert Tx % patch_size[0] == 0
    # assert Hx % patch_size[1] == 0
    # assert Wx % patch_size[2] == 0
    T = (Tx + patch_size[0] - 1) // patch_size[0]
    H = (Hx + patch_size[1] - 1) // patch_size[1]
    W = (Wx + patch_size[2] - 1) // patch_size[2]
    return (T, H, W)


def _load_state_dict_hook_ignore_param_names(param_names: List[str],
                                             module: torch.nn.Module,
                                             incompatible_keys: namedtuple):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this if you want to load a checkpoint that contains partial model weights
    but not the other weights, i.e. train some weight from scratch and finetune the rest.

    Args:
        param_names (list str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f'{param_name} being removed from incompatible_keys.missing_keys in MUGDiT'
            )
            incompatible_keys.missing_keys.remove(param_name)

        if param_name in incompatible_keys.unexpected_keys:
            logging.getLogger(__name__).warning(
                f'{param_name} being removed from incompatible_keys.unexpected_keys in MUGDiT'
            )
            incompatible_keys.unexpected_keys.remove(param_name)
