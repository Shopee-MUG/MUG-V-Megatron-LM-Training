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

from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def controlled_random_state_torch(seed=None):
    # TODO: can be repleaced by torch.fork_rng
    old_cpu_state = torch.random.get_rng_state()
    old_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available(
    ) else None
    try:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(old_cpu_state)
        if old_cuda_state is not None:
            torch.cuda.set_rng_state_all(old_cuda_state)
