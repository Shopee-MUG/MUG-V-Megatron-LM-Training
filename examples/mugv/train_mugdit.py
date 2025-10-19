# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Pretrain or SFT MUGDiT video generation model."""
import contextlib
import os
import sys
import types
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from functools import partial

import torch

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir,
                     os.path.pardir)))

from dataloader_dummy_provider import train_valid_test_dataloaders_provider
from model_flops_utilization import register_flops_hook_for_logging
from mugdit import MUGDiT
from mugdit_spec import get_mugdit_layer_spec, get_mugdit_layer_spec_local
from mugdit_tracker import calc_params_l2_norm_by_filter, task_loss_tracker
from rectified_flow import RFlowScheduler

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.training import (
    get_args,
    get_tensorboard_writer,
    get_timers,
    is_last_rank,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import unwrap_model

# fix for our 6-dim latent broadcasting
tensor_parallel.data._MAX_DATA_DIM = 6


# NOTE: Some mugdit related monkey patching for Megatron-Core is applied here post-initialization to ensure all components are ready,
# avoiding intrusive changes and reducing rebase conflicts with upstream updates.
# It will be called in model_provider as a temporary solution until we can find a more elegant solution.
# This approach, while practical, may not be ideal or intuitive. Suggestions for improving logic or clarity are always welcome.
def global_mugdit_register_post_initialization_hooks():
    args = get_args()
    register_flops_hook_for_logging()

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        # FIXME: Currently, the torch.distributed.batch_isend_irecv interface used by pipeline parallelism
        # requires a global communication initialization step. Without this initialization, subsequent point-to-point communication may hang after ngc 24.07.
        # The root cause of this issue has not yet been identified, so for now, it is temporarily resolved by performing a manual communication step.
        pp_group = mpu.get_pipeline_model_parallel_group()
        torch.distributed.barrier(pp_group)


def model_provider(pre_process=True,
                   post_process=True,
                   add_encoder=True,
                   add_decoder=True,
                   parallel_output=True) -> MUGDiT:
    """Builds the MUGDiT model.

    Args:
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        parallel_output (bool): Enable parallel model output.

    Returns:
        model: A MUGDiT model.
    """
    args = get_args()
    global_mugdit_register_post_initialization_hooks()

    use_te = args.transformer_impl == 'transformer_engine'

    print_rank_0('building a rectified flow scheduler ...')

    scheduler = RFlowScheduler(
        use_timestep_transform=True,
        sample_method='logit-normal',
    )

    print_rank_0('building a video generation model ...')

    transformer_config = core_transformer_config_from_args(args)
    model_type = getattr(args, 'model_type', 'mugdit_10b')
    transformer_config.model_type = model_type
    transformer_config.calculate_per_token_loss = False

    # MUGDiT-specific knobs
    transformer_config.in_channels = getattr(args, 'in_channels', 24)
    transformer_config.caption_channels = getattr(args, 'caption_channels', 4096)
    transformer_config.caption_dropout_prob = getattr(args, 'caption_dropout_prob', 0.0)
    transformer_config.caption_norm = not getattr(args, 'no_caption_norm', False)
    transformer_config.model_max_length = getattr(args, 'model_max_length', 300)
    transformer_config.enable_start_end_tokens = not getattr(args, 'disable_start_end_tokens', False)
    # For pipeline parallelism over variable sequence length inputs
    transformer_config.variable_seq_lengths = True
    # Derive defaults if not set via CLI
    if getattr(transformer_config, 'ffn_hidden_size', None) in (None, 0):
        transformer_config.ffn_hidden_size = transformer_config.hidden_size * 4
    if getattr(transformer_config, 'kv_channels', None) in (None, 0):
        transformer_config.kv_channels = transformer_config.hidden_size // transformer_config.num_attention_heads
    if getattr(transformer_config, 'num_query_groups', None) in (None, 0):
        transformer_config.num_query_groups = transformer_config.num_attention_heads

    if use_te:  # We default to using TE spec for MUGDiT.
        transformer_layer_spec = get_mugdit_layer_spec(
            qk_layernorm=transformer_config.qk_layernorm,
            normalization=transformer_config.normalization,
        )
    else:
        transformer_layer_spec = get_mugdit_layer_spec_local(
            qk_layernorm=transformer_config.qk_layernorm,
            normalization=transformer_config.normalization,
        )

    model = MUGDiT(
        scheduler=scheduler,
        transformer_config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        allow_missing_norm_checkpoint=args.allow_missing_norm_checkpoint,
        enable_start_end_tokens=transformer_config.enable_start_end_tokens,
        pre_process=pre_process,
        post_process=post_process,
    )
    model.freeze(freeze_context_embedder=args.freeze_context_embedder,
                 freeze_cross_attn=False)


    if args.lora:

        from lora_tp_layer_patch import LoraParallelLinear
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora import tp_layer

        tp_layer.LoraParallelLinear = LoraParallelLinear
        transformer_config.to_dict = types.MethodType(lambda self: asdict(self), transformer_config)

        print_rank_0('preparing Peft lora module ...')
        model.requires_grad_(False)
        # lora_target_modules = []
        # for n, m in model.named_modules():
        #     if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d,
        #                     torch.nn.Conv1d)) and 'final_layer' not in n:
        #         lora_target_modules.append(n)
        lora_target_modules=["linear_qkv", "linear_proj", "linear_kv", "linear_q", "linear_fc1", "linear_fc2"]
        # lora_target_modules=["linear_kv", "linear_q"]
        print_rank_0(f'training lora_target_modules: {lora_target_modules}')
        lora_config = LoraConfig(
            r=64,
            lora_alpha=32,
            # lora_dropout=0.1,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
            megatron_config=transformer_config,
            megatron_core="megatron.core",
        )
        model = get_peft_model(model, lora_config, adapter_name='adapter')
        model.print_trainable_parameters()

    return model


def get_batch(data_iterator, scheduler: RFlowScheduler):
    """Generate a batch"""

    dp_rank = mpu.get_data_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    # print(
    #     f'[in get_batch fn] --> dp_rank: {dp_rank}, tp_rank: {tp_rank}, pp_rank: {pp_rank}, {data_iterator} <--'
    # )

    args = get_args()

    # We only load data on the tensor parallel rank 0, and broadcast the data to all other ranks.
    # This is to avoid the data being loaded multiple times on different ranks.
    # This also avoid the broadcasting varible shapes on different ranks.

    latent = None
    num_frames = None
    frame_indices = None
    height = None
    width = None
    fps = None
    temporal_mask = None
    context = None
    context_seqlens = None

    noisy_latent = None
    noise = None
    timestep = None

    # Broadcast data.
    torch.cuda.nvtx.range_push('get_data')
    if data_iterator is not None:
        data = next(data_iterator)
        # print(f'{data.keys()}')
        data = scheduler.prepare_loss_args(
            x_start=data['video'],
            noise=None,
            t=None,
            mask=data['x_mask'],
            model_kwargs=data,
            seed=args.seed + data['index'].sum().item(),
        )
        # print(f'{data.keys()}')
        # for k, v in data.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f'{k}: {v.shape} {v.dtype} {v.device}')
        #     else:
        #         print(f'{k}: {v}')
    else:
        data = None
    latent = tensor_parallel.broadcast_data(['video'], data,
                                            torch.bfloat16)['video']
    num_frames = tensor_parallel.broadcast_data(['num_frames'], data,
                                                torch.int64)['num_frames']
    frame_indices = tensor_parallel.broadcast_data(
        ['frame_indices'], data, torch.int64)['frame_indices']
    height = tensor_parallel.broadcast_data(['height'], data,
                                            torch.int64)['height']
    width = tensor_parallel.broadcast_data(['width'], data,
                                           torch.int64)['width']
    fps = tensor_parallel.broadcast_data(['fps'], data, torch.int64)['fps']
    temporal_mask = tensor_parallel.broadcast_data(['x_mask'], data,
                                                   torch.bool)['x_mask']
    context = tensor_parallel.broadcast_data(['text'], data,
                                             torch.bfloat16)['text']
    context_seqlens = tensor_parallel.broadcast_data(
        ['context_seqlens'], data, torch.int64)['context_seqlens']

    noisy_latent = tensor_parallel.broadcast_data(
        ['noisy_latent'], data, torch.float32)['noisy_latent']
    noise = tensor_parallel.broadcast_data(['noise'], data,
                                           torch.bfloat16)['noise']
    timestep = tensor_parallel.broadcast_data(['timestep'], data,
                                              torch.float32)['timestep']

    if args.fp16:
        train_dtype = torch.float16
    elif args.bf16:
        train_dtype = torch.bfloat16
    else:
        train_dtype = torch.float32
    noisy_latent = noisy_latent.to(train_dtype)
    noise = noise.to(train_dtype)
    timestep = timestep.to(train_dtype)
    context = context.to(train_dtype)

    # tracker for logging
    PREDEFINED_TASKS = [
        't2i', 't2v', 'i2v', 't2v_randone', 'i2v_randone', 'autoreg'
    ]
    if mpu.get_tensor_model_parallel_rank() == 0:
        task_counter = Counter(data['task_type'])
        for k in PREDEFINED_TASKS:
            task_counter[k] = torch.tensor([task_counter[k]],
                                           dtype=torch.int64)
    else:
        task_counter = None

    task_counter_ret = Counter()
    for k in PREDEFINED_TASKS:
        task_counter_ret[k] = tensor_parallel.broadcast_data([k], task_counter,
                                                             torch.int64)[k]

    torch.cuda.nvtx.range_pop()

    # NOTE: We currently inherit dynamic data loading logic (dynamic batch size, token length, and layout),
    # so the actual realtime total batch size is approximately equal to:
    # (batch size per DP rank) * (DP world size) * (nano batch size)
    # We need to keep track of the latent shape, as we want to report the average FLOPs correctly.
    args.cur_batch_shape = latent.shape

    return latent, num_frames, frame_indices, height, width, fps, temporal_mask, context, context_seqlens, noisy_latent, noise, timestep, task_counter_ret


def forward_step(data_iterator, model: MUGDiT):
    """Forward training step.

    Args:
        data_iterator (torch.utils.data.dataloader): Input data iterator
        model: MUGDiT model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    timers = get_timers()
    # print(f'forward_step')

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    (
        latent,
        num_frames,
        frame_indices,
        height,
        width,
        fps,
        temporal_mask,
        context,
        context_seqlens,
        noisy_latent,
        noise,
        timestep,
        task_counter,
    ) = get_batch(data_iterator,
                  unwrap_model(model).scheduler)
    timers('batch-generator').stop()
    model_inputs = {
        'x': noisy_latent,
        'attention_mask': None,
        'temporal_mask': temporal_mask,
        'context': context,
        'context_seqlens': context_seqlens,
        'timestep': timestep,
        'fps': fps,
        'num_frames': num_frames,
        'frame_indices': frame_indices,
        'height': height,
        'width': width,
        'task_counter': task_counter,
    }
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            # print(f'{k}: {v.shape} {v.dtype} {v.device}')
            if torch.isnan(v).any() or torch.isinf(v).any():
                warnings.warn(f'{k} contains nan: {v}')
                exit(1)

    args = get_args()
    if args.fp16:
        train_dtype = torch.float16
    elif args.bf16:
        train_dtype = torch.bfloat16
    else:
        train_dtype = torch.float32

    if train_dtype == torch.float32:
        output_tensor = model(**model_inputs)
    else:
        with torch.amp.autocast('cuda', dtype=train_dtype):
            output_tensor = model(**model_inputs)
    if output_tensor.isnan().any():
        warnings.warn(f'output_tensor contains nan: {output_tensor}')
        exit(1)

    key_list = ['self_attention', 'cross_attention', 'mlp']
    layer_list = [0, 1, 27, 28, 54, 55]

    def filter_by_key_and_layer(name, key, layer):
        return key in name and f'layers.{layer}.' in name

    # FIXME: chunk_params_norm will hang when using pipeline parallelism.
    # chunk_params_norm = {
    #     f'layer{layer}/{key}':
    #     calc_params_l2_norm_by_filter(
    #         model, lambda name: filter_by_key_and_layer(name, key, layer))
    #     for layer in layer_list
    #     for key in key_list
    # }

    logging_stats = {
        'task_counter': task_counter,
        # 'chunk_params_norm': chunk_params_norm,
    }

    return output_tensor, partial(
        unwrap_model(model).scheduler.loss_func,
        x_start=latent,
        noise=noise,
        t=timestep,
        mask=temporal_mask,
        weights=None,
        data_group=mpu.get_data_parallel_group(),
        logging_stats=logging_stats,
    )


def add_mugdit_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='mugdit arguments')

    group.add_argument('--lora', action='store_true', default=False)

    group.add_argument('--use-ema', action='store_true', default=False)
    group.add_argument('--ema-decay', type=float, default=0.9999)
    group.add_argument('--ema-interval', type=int, default=1)

    # MUGDiT hyperparameters (shell-controlled)
    group.add_argument('--model-type', type=str, default='mugdit_10b', dest='vision_model_type',
                       help='Model type tag (e.g., mugdit_1b/4b/10b/18b/debug).')
    group.add_argument('--in-channels', type=int, default=24,
                       help='Input latent channels from VAE (default: 24).')
    group.add_argument('--caption-channels', type=int, default=4096,
                       help='Caption/text embedding size (default: 4096).')
    group.add_argument('--caption-dropout-prob', type=float, default=0.0,
                       help='Dropout prob for caption embedder (default: 0.0).')
    group.add_argument('--no-caption-norm', action='store_true', default=False,
                       help='Disable caption norm layer (enabled by default).')
    group.add_argument('--model-max-length', type=int, default=300,
                       help='Max tokens for caption encoder (default: 300).')
    group.add_argument('--disable-start-end-tokens', action='store_true', default=False,
                       help='Disable adding start/end tokens around latent sequence.')

    group.add_argument('--allow-missing-norm-checkpoint',
                       action='store_true',
                       default=False)
    group.add_argument('--freeze-context-embedder',
                       action='store_true',
                       default=False)
    group.add_argument('--dataloader-save',
                       type=str,
                       default=None,
                       help='Energon dataloader state save path')
    return parser


if __name__ == '__main__':

    # currently we directly use custom dataloader, which is not distributed yet.
    train_valid_test_dataloaders_provider.is_distributed = False

    with contextlib.nullcontext():
        # with torch.autograd.detect_anomaly(check_nan=False):
        pretrain(
            train_valid_test_dataloaders_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
            extra_args_provider=add_mugdit_extra_args,
        )
