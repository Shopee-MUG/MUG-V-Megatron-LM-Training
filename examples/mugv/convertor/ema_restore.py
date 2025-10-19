# Copyright (c) 2024, Babyfan. All rights reserved.

# This script is used to restore the EMA state into parameter states while preserving the distributed checkpoint format.

import argparse
import os
import shutil

import torch
from einops import rearrange
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict

ema_prefix = 'optimizer.state.exp_moving_avg'
opt_prefix = 'optimizer.state'


def load_dcp(dcp_dir):
    state: STATE_DICT_TYPE = {}
    _load_state_dict(
        state,
        storage_reader=FileSystemReader(dcp_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state


def save_dcp(state, out_dcp_dir):
    os.makedirs(out_dcp_dir, exist_ok=True)
    _save_state_dict(state,
                     storage_writer=FileSystemWriter(out_dcp_dir,
                                                     overwrite=True),
                     no_dist=True)


def tp_mapping(v, chunk_dim=None, ov=None):
    if chunk_dim == None:
        return v.view_as(ov)

    oh1, oh2 = ov.shape[-2:]
    tp1, tp2 = v.shape[1:3]
    h1 = oh1 // tp1
    h2 = oh2 // tp2
    return rearrange(v,
                     'l tp1 tp2 (h1 h2) -> l (tp1 h1) (tp2 h2)',
                     h1=h1,
                     h2=h2)


def restore_ema_into_params(dcp_dir, ema_dcp_dir):

    state = load_dcp(dcp_dir)
    print(f"INFO: Start restoring EMA state into params...")

    for k, v in state.items():
        if '_extra_state' in k:
            continue

        elif 'rng_state/' in k:
            continue

        elif k.startswith(ema_prefix):
            ok = k[len(ema_prefix) + 1:]
            if ok not in state:
                print(f'WARN: {ok} not in state')
                __import__('ipdb').set_trace()
            else:
                ov = state[ok]
                v = v.to(torch.bfloat16)
                # print(ok, v.shape, ov.shape)
                chunk_dim = None
                if 'self_attention.linear_qkv.weight' in ok:
                    chunk_dim = 0
                elif 'self_attention.linear_proj.weight' in ok:
                    chunk_dim = 1
                elif 'cross_attention.linear_q.weight' in ok:
                    chunk_dim = 0
                elif 'cross_attention.linear_kv.weight' in ok:
                    chunk_dim = 0
                elif 'cross_attention.linear_proj.weight' in ok:
                    chunk_dim = 1
                elif 'mlp.linear_fc1.weight' in ok:
                    chunk_dim = 0
                elif 'mlp.linear_fc2.weight' in ok:
                    chunk_dim = 1
                reshaped_v = tp_mapping(v, chunk_dim=chunk_dim, ov=ov)
                if (reshaped_v.shape != ov.shape):
                    __import__('ipdb').set_trace()
                    reshaped_v = tp_mapping(v, chunk_dim=chunk_dim, ov=ov)
                    print(reshaped_v.shape, ov.shape)

                # For debug correctness, ensure EMA decay=0 before proceeding.
                correct_check = False
                if correct_check:
                    diff = (reshaped_v - ov).abs().sum()
                    if diff > 0:
                        __import__('ipdb').set_trace()
                        print(f"WARNING: {ok} - Difference detected: {diff}")
                    else:
                        print(f"INFO: {ok} - Correctness verified.")

        elif k.startswith(opt_prefix):
            continue

        else:
            continue
            # print(f"WARN: no process logic for key: {k}, keep it.")

    # remove unnecessary keys
    for k in list(state.keys()):
        if 'rng_state/' in k:
            del state[k]
        elif k.startswith(opt_prefix):
            del state[k]

    save_dcp(state, ema_dcp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Restore the EMA state into parameter states while preserving the distributed checkpoint format.

NOTE: This script loads the entire checkpoint into memory at once, which can be very large.
We recommend running it in a compute resource pool, or on a machine with sufficient memory.

IMPORTANT: This logic does not modify optimizer metadata and thus cannot be used to resume optimizer states;
attempting to do so may lead to errors.

Example usage:
python ema_restore.py --dcp-dir /path/to/dcp/checkpoints --output /path/to/output/checkpoints --iter 2000
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dcp-dir",
                        type=str,
                        default='/distributed/checkpoint/folder',
                        required=False,
                        help="megatron dcp directory")
    parser.add_argument("--output",
                        type=str,
                        default='/tmp/ema_dcp',
                        required=False,
                        help="ema megatron dcp output directory")
    parser.add_argument("--iter",
                        type=int,
                        default=1,
                        required=False,
                        help="convert iter")

    args = parser.parse_args()
    args.dcp_dir = os.path.join(args.dcp_dir, f"iter_{args.iter:07d}")
    args.output = os.path.join(args.output, f"iter_{args.iter:07d}")

    restore_ema_into_params(args.dcp_dir, args.output)

    # copy misc files from dcp_dir to output
    for misc in ['common.pt', 'metadata.json']:
        shutil.copyfile(os.path.join(args.dcp_dir, misc),
                        os.path.join(args.output, misc))
    with open(
            os.path.join(os.path.dirname(args.output),
                         "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(args.iter))
    print(f"Ema state restore done. Please refer to {args.output}")
