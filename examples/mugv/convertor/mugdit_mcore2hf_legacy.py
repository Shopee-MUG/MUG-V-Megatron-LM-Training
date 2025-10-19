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

import argparse
import io
import json
import os

import torch
from einops import rearrange
from loguru import logger
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

model_config = {
    "DEBUG": {
        "hidden_dim": 288,
        "num_heads": 4,
        "num_layers": 2,
    },
    "1B": {
        "hidden_dim": 1152,
        "num_heads": 16,
        "num_layers": 56,
    },
    "4B": {
        "hidden_dim": 2304,
        "num_heads": 32,
        "num_layers": 56,
    },
    "10B": {
        "hidden_dim": 3456,
        "num_heads": 48,
        "num_layers": 56,
    },
    "18B": {
        "hidden_dim": 4608,
        "num_heads": 64,
        "num_layers": 56,
    },
}


def load_from_sharded_or_unsharded(path):
    if os.path.isfile(path) and path.endswith(".pt"):
        return torch.load(path, weights_only=True)

    index_file = os.path.join(path, "pytorch_model.bin.index.json")
    model_shards_prefix = "pytorch_model-"

    if os.path.isdir(path) and os.path.exists(index_file):
        with open(index_file, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        state_dict = {}
        for param_name, shard_file in weight_map.items():
            shard_path = f"{model_shards_prefix}{shard_file.split('-')[-1]}"
            if param_name not in state_dict:
                logger.info(f"Loading shard: {shard_path}")
                shard_state = torch.load(
                    os.path.join(path, shard_path),
                    map_location="cpu",
                    weights_only=True,
                )
                state_dict.update(shard_state)

        return state_dict


def load_state_dict(dcp_dir, torch_path, ref_ckpt=None):
    if torch_path is not None:
        state_dict = load_from_sharded_or_unsharded(torch_path)
        logger.info(f"Directly loaded torch checkpoint {torch_path=}")

    else:
        mem_file = io.BytesIO()
        dcp_to_torch_save(dcp_dir, mem_file)
        mem_file.seek(0)
        state_dict = torch.load(mem_file)
        logger.info(f"Converted dcp to torch checkpoint and loaded {dcp_dir=}")

    ref_state_dict = (
        load_from_sharded_or_unsharded(ref_ckpt) if ref_ckpt is not None else None
    )

    return state_dict, ref_state_dict


def convert_mcore2hf(
    dcp_checkpoint_dir,
    torch_checkpoint_file,
    torch_save_file,
    model_size,
    ref_hf_ckpt,
):
    logger.info(f"Try to convert dcp {model_size} model to hf format")

    model_cfg = model_config.get(model_size.upper(), {})
    assert model_cfg != {}, f"invalid {model_size=}, choose in {model_config.keys()}"
    hidden_dim = model_cfg["hidden_dim"]
    num_heads = model_cfg["num_heads"]
    num_layers = model_cfg["num_layers"]
    head_dim = hidden_dim // num_heads

    state_dict, ref_state_dict = load_state_dict(
        dcp_checkpoint_dir, torch_checkpoint_file, ref_hf_ckpt
    )
    new_state_dicts = dict()

    logger.info(
        "Converted dcp checkpoint to torch format. Start converting to hf format..."
    )

    # Reversed Indices from mapping pytorch multihead attention to megatron.
    def mapping(value, n):
        return rearrange(
            value, "(nh n c) ... -> (n nh c) ...", n=n, nh=num_heads, c=head_dim
        )

    if "y_embedder.y_embedding" in state_dict:
        state_dict.pop("y_embedder.y_embedding")
    if "context_embedder.drop_emb" in state_dict:
        state_dict.pop("context_embedder.drop_emb")

    for name, tensor in state_dict.items():
        if name.startswith("optimizer."):
            continue

        if "_extra_state" in name:
            continue

        if "rng_state" in name:
            continue  # TODO: do we need this?

        if tensor.dtype in [torch.float16, torch.bfloat16]:
            tensor = tensor.to(torch.float32)

        new_name = ""
        new_tensor = tensor
        if "start_token" in name:
            new_name = name
        elif "end_token" in name:
            new_name = name
        elif "x_embedder" in name:
            new_name = name
        elif name.startswith("t_embedder"):  # fix bugs with context_embedder
            new_name = name
        elif "fps_embedder" in name:
            new_name = name
        elif "t_block" in name:
            new_name = name

        elif "context_embedder" in name:
            if "proj" in name:
                new_name = name.replace("context_embedder.proj", "y_embedder.y_proj")
            elif "drop_emb" in name:
                new_name = name.replace(
                    "context_embedder.drop_emb", "y_embedder.y_embedding"
                )

        elif "final_layer" in name:
            if "scale_shift_table" in name:
                new_name = name.replace(
                    "final_layer.scale_shift_table.bias",
                    "final_layer.scale_shift_table",
                )
            else:
                new_name = name

        if new_name != "":
            new_state_dicts[new_name] = new_tensor

        def get_hf_prefix(i):
            bname = "spatial_blocks" if i & 1 == 0 else "temporal_blocks"
            return f"{bname}.{i // 2}"

        if "decoder.layers" in name:
            for layer_idx in range(num_layers):
                new_tensor = tensor[layer_idx]
                prefix = get_hf_prefix(layer_idx)
                if ".self_attention.linear_qkv.weight" in name:
                    new_name = f"{prefix}.attn.qkv.weight"
                    new_tensor = mapping(new_tensor, 3)
                elif ".self_attention.linear_qkv.bias" in name:
                    new_name = f"{prefix}.attn.qkv.bias"
                    new_tensor = mapping(new_tensor, 3)

                elif ".self_attention.linear_proj.weight" in name:
                    new_name = f"{prefix}.attn.proj.weight"
                elif ".self_attention.linear_proj.bias" in name:
                    new_name = f"{prefix}.attn.proj.bias"

                elif ".self_attention.q_layernorm.weight" in name:
                    new_name = f"{prefix}.attn.q_norm.weight"
                elif ".self_attention.q_layernorm.bias" in name:
                    new_name = f"{prefix}.attn.q_norm.bias"
                elif ".self_attention.k_layernorm.weight" in name:
                    new_name = f"{prefix}.attn.k_norm.weight"
                elif ".self_attention.k_layernorm.bias" in name:
                    new_name = f"{prefix}.attn.k_norm.bias"

                # deal with cross attention
                elif ".cross_attention.linear_q.weight" in name:
                    new_name = f"{prefix}.cross_attn.q_linear.weight"
                    new_tensor = mapping(new_tensor, 1)
                elif ".cross_attention.linear_q.bias" in name:
                    new_name = f"{prefix}.cross_attn.q_linear.bias"
                    new_tensor = mapping(new_tensor, 1)
                elif ".cross_attention.linear_kv.weight" in name:
                    new_name = f"{prefix}.cross_attn.kv_linear.weight"
                    new_tensor = mapping(new_tensor, 2)
                elif ".cross_attention.linear_kv.bias" in name:
                    new_name = f"{prefix}.cross_attn.kv_linear.bias"
                    new_tensor = mapping(new_tensor, 2)

                elif ".cross_attention.linear_proj.weight" in name:
                    new_name = f"{prefix}.cross_attn.proj.weight"
                elif ".cross_attention.linear_proj.bias" in name:
                    new_name = f"{prefix}.cross_attn.proj.bias"

                # deal with mlp
                elif ".mlp.linear_fc1.weight" in name:
                    new_name = f"{prefix}.mlp.fc1.weight"
                elif ".mlp.linear_fc1.bias" in name:
                    new_name = f"{prefix}.mlp.fc1.bias"
                elif ".mlp.linear_fc2.weight" in name:
                    new_name = f"{prefix}.mlp.fc2.weight"
                elif ".mlp.linear_fc2.bias" in name:
                    new_name = f"{prefix}.mlp.fc2.bias"

                # deal with block scale shift table
                elif ".scale_shift_table.bias" in name:
                    new_name = f"{prefix}.scale_shift_table"

                new_state_dicts[new_name] = new_tensor

    if ref_state_dict is not None:
        assert set(new_state_dicts.keys()) == set(ref_state_dict.keys())
        for k1, k2 in zip(
            sorted(new_state_dicts.keys()), sorted(ref_state_dict.keys())
        ):
            assert k1 == k2
            v1 = new_state_dicts[k1].cuda()
            v2 = ref_state_dict[k2].cuda()
            if not torch.allclose(v1, v2, atol=1e-4):
                logger.debug(k1, (v1 - v2).abs().sum())
        logger.info("check ref state dict finished.")

    torch.save(new_state_dicts, torch_save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert mcore dcp checkpoint / mcore state dict to HuggingFace MUGDiT state dict.


Example usage:
python -m examples.mugv.convertor.mugdit_mcore2hf_legacy --dcp-dir /dcp/ckpt/folder --save-file /torch/save/file --model-size 10B
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dcp-dir",
        type=str,
        default=None,
        required=False,
        help="Mcore dcp format checkpoint directory",
    )
    parser.add_argument(
        "--mcore-state",
        type=str,
        default=None,
        required=False,
        help="Torch checkpoint path(sharded) or file(unsharded) of mcore state dict",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default="/tmp/hf_ckpt.pt",
        required=False,
        help="Output file for HuggingFace MUGDiT state dict",
    )
    parser.add_argument(
        "--ref-hf-ckpt",
        type=str,
        default=None,
        help="Reference HuggingFace checkpoint for checking percision",
    )
    parser.add_argument(
        "--model-size", type=str, default="10b", help="model size: debug / 10b"
    )
    args = parser.parse_args()

    assert (args.dcp_dir is not None) ^ (args.mcore_state is not None), (
        "Please provide either dcp checkpoint dir or mcore state dict"
    )
    convert_mcore2hf(
        args.dcp_dir,
        args.mcore_state,
        args.save_file,
        args.model_size,
        args.ref_hf_ckpt,
    )
    logger.info(f"done. Please refer to {args.save_file}")
