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
import os
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import get_checkpoint_name

vae_mean = (
    -0.0194091796875,
    0.09619140625,
    -0.796875,
    0.1591796875,
    -0.2578125,
    0.359375,
    -0.3203125,
    -0.287109375,
    -0.0069580078125,
    0.3046875,
    0.310546875,
    -0.451171875,
    -0.1728515625,
    0.369140625,
    0.2177734375,
    -3.075599670410156e-05,
    0.1630859375,
    -0.267578125,
    -0.1962890625,
    -0.1298828125,
    -0.28515625,
    -0.515625,
    0.5859375,
    -0.34375,
)
vae_std = (
    0.8817,
    0.6523,
    0.9152,
    1.2117,
    3.3516,
    0.7528,
    0.8177,
    0.8637,
    0.9075,
    2.8875,
    0.8980,
    1.1202,
    1.0003,
    2.5163,
    0.6652,
    1.2573,
    0.7279,
    1.0777,
    1.5159,
    0.8680,
    1.1859,
    1.0484,
    2.4750,
    1.5881,
)


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    if mpu.get_tensor_model_parallel_rank() != 0:
        logger.warning("> No dataloader initialized for non-tp0 rank")
        return None, None, None

    from data_module.dataloader import prepare_dataloader
    from data_module.datasets import LatentDataset

    args = get_args()

    data_path = getattr(args, "data_path")[0]
    dataset = LatentDataset(data_path=data_path, vae_shift=vae_mean, vae_scale=vae_std)
    logger.info(f"Dataset contains {len(dataset)} samples.")

    dataloader = prepare_dataloader(
        dataset=dataset,
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        sampler_configs={},
        process_group=mpu.get_data_parallel_group(),
    )

    dataloader = SavableDataloader(dataloader)
    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = mpu.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                pipeline_rank=0,
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(data_save_name, map_location="cpu")
                    dataloader.restore_state(
                        dataset_state_dict["dataloader_state_dict"]
                    )
                    print_rank_0(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print_rank_0(
                        "loading dataloader checkpoint failed. Skipping. " + str(e)
                    )

    dp_rank = mpu.get_data_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    return dataloader, dataloader, dataloader


class SavableDataloader:
    """A Savable wrapper for dataloader in the Megatron-LM training loop."""

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._sampler = dataloader.sampler
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        args = get_args()
        return self._sampler.state_dict(args.curr_iteration + 1)

    def restore_state(self, state):
        self._sampler.load_state_dict(state)


def cyclic_iter(iter):
    while True:
        for x in iter:
            if x is not None:
                yield x


if __name__ == "__main__":
    train_dataloader, valid_dataloader, test_dataloader = (
        train_valid_test_dataloaders_provider(train_val_test_num_samples=1)
    )

    loader_batch = None
    for i, batch in enumerate(train_dataloader):
        print(i, batch)
        loader_batch = batch
        break
    __import__("ipdb").set_trace()
