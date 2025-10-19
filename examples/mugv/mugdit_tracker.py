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

"""General utilities."""
import os
import sys
from collections import defaultdict
from datetime import datetime

import torch

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        multi_tensor_applier = None

    try:
        from amp_C import multi_tensor_l2norm
    except ImportError:
        import warnings
        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_l2norm'
        )

        from megatron.core.utils import (
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
            local_multi_tensor_applier as multi_tensor_applier,
        )

from megatron.core import DistributedDataParallel as DDP
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.transformer.module import Float16Module
from megatron.legacy.model.module import param_is_not_shared
from megatron.training import get_adlr_autoresume, get_args


def calc_params_l2_norm_by_filter(model, filter_fn=lambda : True):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for name, param in model_.named_parameters():
            if not filter_fn(name):
                continue
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if mpu.get_expert_model_parallel_rank() > 0:
                if not getattr(param, 'allreduce', True) and is_not_tp_duplicate:
                    assert param_is_not_shared(param)
                    params_data.append(param.data.float() if args.bf16 else param.data)
            else:
                is_not_shared = param_is_not_shared(param)
                if is_not_shared and is_not_tp_duplicate:
                    params_data.append(param.data.float() if args.bf16 else param.data)

    if len(params_data) == 0:
        return -1

    # Calculate norm
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    norm, _ = multi_tensor_applier(
        multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    if mpu.get_expert_model_parallel_world_size() == 1:
        # Sum across all model-parallel GPUs(tensor + pipeline).
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_model_parallel_group())
    else:
        # Sum across tensor, pipeline and expert model-parallel GPUs.
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_tensor_and_expert_parallel_group())
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_pipeline_model_parallel_group())
    return norm_2.item() ** 0.5


class TaskLossTracker:

    def __init__(self):
        self.track_dict = defaultdict(lambda: (0, 0.0))

    @torch.no_grad()
    def __call__(self, loss, task, num=1, smoothing=False):
        cnt, avg_loss = self.track_dict[task]
        cnt += num
        avg_loss += (loss.item() - avg_loss) / cnt
        self.track_dict[task] = (cnt, avg_loss)
        return (cnt, avg_loss) if smoothing else (num, loss.item())

    def get_avg_loss(self, task):
        if task in self.track_dict:
            return self.track_dict[task][1]
        else:
            raise ValueError(f"Task '{task}' not found.")

    def reset(self, task=None):
        if task is None:
            self.track_dict = defaultdict(lambda: (0, 0.0))
        else:
            if task in self.track_dict:
                del self.track_dict[task]
            else:
                raise ValueError(f"Task '{task}' not found.")


task_loss_tracker = TaskLossTracker()
