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
import json
import os

import torch
from einops import rearrange
from loguru import logger

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
        return torch.load(path, map_location="cpu", weights_only=True)

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

    else:
        raise NotImplementedError(f"Unsupported file format or file not found: {path}")


def convert_hf2mcore(hf_ckpt, output_path, tensor_parallel_size, use_te, model_size):
    logger.info(f"Try to convert {model_size} hf model")

    model_cfg = model_config.get(model_size.upper(), {})
    assert model_cfg != {}, f"invalid {model_size=}, choose in {model_config.keys()}"
    hidden_dim = model_cfg["hidden_dim"]
    num_heads = model_cfg["num_heads"]
    num_layers = model_cfg["num_layers"]
    head_dim = hidden_dim // num_heads

    hf_ckpt = hf_ckpt or "./megatron/model_ckpt.pt"
    state_dict = load_from_sharded_or_unsharded(hf_ckpt)

    new_state_dicts = [{"model": {}} for _ in range(tensor_parallel_size)]

    # Indices from mapping pytorch multihead attention to megatron.
    def mapping(value, n):
        return rearrange(
            value, "(n nh c) ... -> (nh n c) ...", n=n, nh=num_heads, c=head_dim
        )

    if "y_embedder.y_embedding" not in state_dict:
        state_dict["y_embedder.y_embedding"] = torch.load(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "fixtures", "y_embedding.pt"
            ),
            map_location="cpu",
            weights_only=True,
        )
    if "context_embedder.drop_emb" in state_dict:
        dummy_drop_emb = state_dict.pop("context_embedder.drop_emb")
        logger.debug(
            (dummy_drop_emb - state_dict["y_embedder.y_embedding"]).abs().sum()
        )
    # state_dict.pop("start_token")
    # state_dict.pop("end_token")
    # Remove the following two lines to be compatible with new setting
    # if "y_embedder.y_embedding" in state_dict:
    #     state_dict.pop("y_embedder.y_embedding")
    # if "context_embedder.drop_emb" in state_dict:
    #     state_dict.pop("context_embedder.drop_emb")

    keys = list(state_dict.keys())
    for name in keys:
        tensor = state_dict.pop(name)
        if tensor is None:
            continue

        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor
        if new_tensor.dtype in [torch.float16, torch.bfloat16]:
            new_tensor = new_tensor.to(torch.float32)

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None

        if "start_token" in name:
            new_name = name
        elif "end_token" in name:
            new_name = name
        elif "x_embedder" in name:
            new_name = name
        elif name.startswith("t_embedder"):
            new_name = name.replace("t_embedder", "timestep_embedder")
        elif "fps_embedder" in name:
            new_name = name
        elif "t_block" in name:
            new_name = name.replace("t_block", "timestep_block")
        elif "y_embedder" in name:
            if "y_proj" in name:
                new_name = name.replace("y_embedder.y_proj", "context_embedder.proj")
            elif "y_embedding" in name:
                new_name = name.replace(
                    "y_embedder.y_embedding", "context_embedder.drop_emb"
                )

        elif "final_layer" in name:
            if "scale_shift_table" in name:
                new_name = name.replace(
                    "final_layer.scale_shift_table",
                    "final_layer.scale_shift_table.bias",
                )
            else:
                new_name = name

        elif "blocks" in name:
            layer_idx = int(name.split(".")[1])
            base = f"decoder.layers.{layer_idx}"

            # deal with self attention
            if ".attn.qkv.weight" in name:
                new_name = f"{base}.self_attention.linear_qkv.weight"
                new_tensor = mapping(new_tensor, 3)
                chunk_dim = 0
            elif ".attn.qkv.bias" in name:
                new_name = f"{base}.self_attention.linear_qkv.bias"
                new_tensor = mapping(new_tensor, 3)
                chunk_dim = 0
            elif ".attn.proj.weight" in name:
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1
            elif ".attn.proj.bias" in name:
                new_name = f"{base}.self_attention.linear_proj.bias"

            # BUG: qk layernorm in hf has no bias
            elif ".attn.q_norm.weight" in name:
                new_name = f"{base}.self_attention.q_layernorm.weight"
            elif ".attn.q_norm.bias" in name:
                new_name = f"{base}.self_attention.q_layernorm.bias"
            elif ".attn.k_norm.weight" in name:
                new_name = f"{base}.self_attention.k_layernorm.weight"
            elif ".attn.k_norm.bias" in name:
                new_name = f"{base}.self_attention.k_layernorm.bias"

            # deal with pre cross attn layernorm
            elif ".cross_attn_norm.weight" in name:
                new_name = f"{base}.pre_cross_attn_layernorm.weight"
            elif ".cross_attn_norm.bias" in name:
                new_name = f"{base}.pre_cross_attn_layernorm.bias"

            # deal with newly added qk_norm params in cross attention
            elif ".cross_attn.q_norm.weight" in name:
                new_name = f"{base}.cross_attention.q_layernorm.weight"
            elif ".cross_attn.q_norm.bias" in name:
                new_name = f"{base}.cross_attention.q_layernorm.bias"
            elif ".cross_attn.k_norm.weight" in name:
                new_name = f"{base}.cross_attention.k_layernorm.weight"
            elif ".cross_attn.k_norm.bias" in name:
                new_name = f"{base}.cross_attention.k_layernorm.bias"

            # deal with cross attention
            elif ".cross_attn.q_linear.weight" in name:
                new_name = f"{base}.cross_attention.linear_q.weight"
                new_tensor = mapping(new_tensor, 1)
                chunk_dim = 0
            elif ".cross_attn.q_linear.bias" in name:
                new_name = f"{base}.cross_attention.linear_q.bias"
                new_tensor = mapping(new_tensor, 1)
                chunk_dim = 0
            elif ".cross_attn.kv_linear.weight" in name:
                new_name = f"{base}.cross_attention.linear_kv.weight"
                new_tensor = mapping(new_tensor, 2)
                chunk_dim = 0
            elif ".cross_attn.kv_linear.bias" in name:
                new_name = f"{base}.cross_attention.linear_kv.bias"
                new_tensor = mapping(new_tensor, 2)
                chunk_dim = 0
            elif ".cross_attn.proj.weight" in name:
                new_name = f"{base}.cross_attention.linear_proj.weight"
                chunk_dim = 1
            elif ".cross_attn.proj.bias" in name:
                new_name = f"{base}.cross_attention.linear_proj.bias"

            # deal with mlp
            elif ".mlp.fc1.weight" in name:
                new_name = f"{base}.mlp.linear_fc1.weight"
                chunk_dim = 0
            elif ".mlp.fc1.bias" in name:
                new_name = f"{base}.mlp.linear_fc1.bias"
                chunk_dim = 0
            elif ".mlp.fc2.weight" in name:
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif ".mlp.fc2.bias" in name:
                new_name = f"{base}.mlp.linear_fc2.bias"
            # BUG: mlp linear_fc1 layernorm has no weight and bias

            # deal with block scale shift table
            elif ".scale_shift_table" in name:
                new_name = f"{base}.scale_shift_table.bias"

        assert new_name != "", f"unexpected layer name {name}"

        if chunk_dim is None:
            new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
        else:
            new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        for i in range(tensor_parallel_size):
            # chunk() creates a view of a bigger tensor. clone() is used here to avoid excessive storage.
            new_state_dicts[i]["model"][new_name] = new_tensors[i].clone()

            # TE sets _extra_state (for FP8 purposes), so set an empty one here for compatibility.
            extra_state_layers = (
                "linear_qkv",
                "linear_q",
                "linear_kv",
                "linear_proj",
                "linear_fc1",
                "linear_fc2",
                "scale_shift_table",
            )
            is_extra_state_layer = any([l in new_name for l in extra_state_layers])
            if use_te and is_extra_state_layer:
                layer = new_name.split(".")[-2]
                if layer in extra_state_layers:
                    extra_state_name = (
                        new_name[: new_name.rfind(".") + 1] + "_extra_state"
                    )  # Replace the weight name.
                    new_state_dicts[i]["model"][extra_state_name] = None

    for i in range(tensor_parallel_size):
        output_dir_tp = os.path.join(output_path, "iter_0000001", f"mp_rank_0{i}")
        os.makedirs(output_dir_tp, exist_ok=True)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)

    with open(os.path.join(output_path, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert MUGDiT hf weights to megatron-core weights.


Example usage:
python -m examples.mugv.convertor.mugdit_hf2mcore --hf-ckpt /hf/ckpt/folder --output /mcore/output/folder --tensor-parallel-size 4 --use-te --model-size 10B
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-ckpt",
        type=str,
        default=None,
        required=False,
        help="HuggingFace checkpoints weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/converted",
        required=False,
        help="output directory for megatron state dict file(s)",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=4, help="model tensor parallel size"
    )
    parser.add_argument("--use-te", action="store_true", help="Use Transformer Engine")
    parser.add_argument(
        "--model-size", type=str, default="10b", help="model size: debug / 10b"
    )
    args = parser.parse_args()

    args.output = os.path.join(args.output, "checkpoints")
    convert_hf2mcore(
        args.hf_ckpt,
        args.output,
        args.tensor_parallel_size,
        args.use_te,
        args.model_size,
    )
    logger.info(f"done. Please refer to {args.output}")
