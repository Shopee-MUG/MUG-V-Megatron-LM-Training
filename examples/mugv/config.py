# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import torch.nn.functional as F


def get_mugdit_model_config(config):
    """hook for MUGDiT config.

    It ensures a few MUGDiT-specific defaults if they are not already set by CLI.
    """
    # Activation for fused kernels (kept consistent with Megatron defaults)
    config.activation_func = F.gelu

    # MUGDiT-specific defaults (do not override if set by CLI)
    if getattr(config, 'in_channels', None) is None:
        config.in_channels = 24
    config.out_channels = config.in_channels * 2

    if getattr(config, 'caption_channels', None) is None:
        config.caption_channels = 4096
    if getattr(config, 'caption_dropout_prob', None) is None:
        config.caption_dropout_prob = 0.0
    if getattr(config, 'caption_norm', None) is None:
        config.caption_norm = True
    if getattr(config, 'model_max_length', None) is None:
        config.model_max_length = 300
    if getattr(config, 'enable_start_end_tokens', None) is None:
        config.enable_start_end_tokens = True

    # Allow variable sequence lengths for PP by default
    if getattr(config, 'variable_seq_lengths', None) is None:
        config.variable_seq_lengths = True

    return config
