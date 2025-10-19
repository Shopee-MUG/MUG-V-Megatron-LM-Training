import time
from collections import OrderedDict, defaultdict
from math import ceil
from pprint import pformat
from typing import Iterator, List, Optional

import numpy as np
import orjson
import torch
from loguru import logger
from torch.utils.data import DistributedSampler

from megatron.core import mpu


class StatefulDistributedSampler(DistributedSampler):

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index:]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def reset(self) -> None:
        self.start_index = 0

    def state_dict(self, step) -> dict:
        return {'start_index': step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

