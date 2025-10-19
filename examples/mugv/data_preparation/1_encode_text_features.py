#!/usr/bin/env python3
"""
Extract text features using MUG-V T5Encoder from MUG-V.

This script extracts T5-XXL text embeddings for video generation training.
It uses the exact same text encoder from MUG-V to ensure consistency.

Prerequisites:
    uv pip install -r examples/mugv/data_preparation/requirements.txt

Output format (per sample_id):
    text_features/<sample_id>_text.pt = {
        'y': FloatTensor[1, 1, seq_len, 4096],  # Text embeddings
        'mask': BoolTensor[1, seq_len]          # Attention mask
    }

Supported inputs:
    - CSV with columns for sample_id and text
    - JSONL with fields {"sample_id": ..., "text": ...}

Example usage:
    python data_preparation/1_encode_text_features.py \
        --captions /path/to/captions.csv \
        --output-dir /path/to/text_features \
        --batch-size 32
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

try:
    from mug_v.encoder.text_encoder import T5
except ImportError as e:
    raise RuntimeError(
        f"Failed to import T5Encoder from MUG-V: {e}\n"
        "Install with: uv pip install -r examples/mugv/data_preparation/requirements.txt"
    )


def read_csv(path: str, id_col: str, text_col: str) -> List[Tuple[str, str]]:
    """Read captions from CSV file."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = str(r[id_col]).strip()
            txt = str(r[text_col]).strip()
            if sid and txt:
                rows.append((sid, txt))
    return rows


def read_jsonl(path: str, id_field: str, text_field: str) -> List[Tuple[str, str]]:
    """Read captions from JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = str(obj[id_field]).strip()
            txt = str(obj[text_field]).strip()
            if sid and txt:
                rows.append((sid, txt))
    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Extract text features using MUG-V T5")
    parser.add_argument(
        "--captions",
        type=str,
        required=True,
        help="CSV or JSONL file with captions",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column/field name for text (default: text)",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="sample_id",
        help="Column/field name for sample ID (default: sample_id)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for text features",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="DeepFloyd/t5-v1_1-xxl",
        help="T5 model name or path (default: DeepFloyd/t5-v1_1-xxl)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=300,
        help="Maximum sequence length (default: 300)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read captions (We support CSV and JSONL formats)
    if args.captions.lower().endswith(".csv"):
        pairs = read_csv(args.captions, args.id_column, args.text_column)
    elif args.captions.lower().endswith(".jsonl"):
        pairs = read_jsonl(args.captions, args.id_column, args.text_column)
    else:
        raise ValueError("Unsupported format. Use .csv or .jsonl please")

    print(f"Loaded {len(pairs)} captions from {args.captions}")

    # Initialize T5 from MUG-V
    print(f"Loading T5: {args.model_name}")
    text_encoder = T5(
        from_pretrained=args.model_name,
        max_model_len=args.max_length,
        device=args.device,
        dtype=torch.bfloat16,
    )
    print(f"T5 output dimension: {text_encoder.output_dim}")

    # Batch encode
    total_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
    for i in tqdm(
        range(0, len(pairs), args.batch_size),
        desc="Encode text",
        total=total_batches,
    ):
        batch = pairs[i : i + args.batch_size]
        sample_ids, texts = zip(*batch)

        # Encode texts using T5
        text_info = text_encoder.encode(list(texts))

        # text_info['y']: [B, 1, seq_len, hidden]
        # text_info['mask']: [B, seq_len]
        embeddings = text_info["y"].cpu()  # [B, 1, L, H]
        masks = text_info["mask"].cpu()  # [B, L]

        # Save samples
        for sid, emb, mask in zip(sample_ids, embeddings, masks):
            # emb shape: [1, L, 4096]
            # mask shape: [L]
            # NOTE: LatentDataset expects: text_feat["y"][-1] and text_feat["mask"][-1]
            obj = {
                "y": emb.unsqueeze(0).to(torch.float32),  # [1, 1, L, 4096]
                "mask": mask.unsqueeze(0).bool(),  # [1, L]
            }
            save_path = os.path.join(args.output_dir, f"{sid}_text.pt")
            torch.save(obj, save_path)

    print(f"Saved {len(pairs)} text features to {args.output_dir}")


if __name__ == "__main__":
    main()
