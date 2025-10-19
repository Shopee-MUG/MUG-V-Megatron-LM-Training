#!/usr/bin/env python3
"""
Generate train.csv for MUG-V Megatron training.

This script creates a training CSV by matching latents and text features by sample_id.
It automatically discovers files and validates data integrity.

CSV format:
    sample_id,source,latent_path,text_feat_path

Modes:
    1. Auto-discovery: --latents + --text-features
    2. From captions CSV: --captions + --latents + --text-features

Example usage:
    # Auto-discovery mode
    python data_preparation/3_generate_training_csv.py \
        --latents /path/to/latents \
        --text-features /path/to/text_features \
        --output /path/to/train.csv

    # With captions CSV for additional validation
    python data_preparation/3_generate_training_csv.py \
        --captions /path/to/captions.csv \
        --latents /path/to/latents \
        --text-features /path/to/text_features \
        --output /path/to/train.csv
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple


def read_captions_csv(
    path: str, id_col: str = "sample_id", text_col: str = "text"
) -> Dict[str, str]:
    """Read captions from CSV file."""
    captions = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row[id_col]).strip()
            text = str(row[text_col]).strip()
            if sample_id and text:
                captions[sample_id] = text
    return captions


def index_latents(latents_dir: str) -> Set[str]:
    """Index latent files by sample_id."""
    sample_ids = set()
    for f in os.listdir(latents_dir):
        if f.endswith(".pt"):
            sample_id = Path(f).stem
            sample_ids.add(sample_id)
    return sample_ids


def index_text_features(text_dir: str) -> Set[str]:
    """Index text feature files by sample_id."""
    sample_ids = set()
    for f in os.listdir(text_dir):
        if f.endswith("_text.pt"):
            sample_id = f[: -len("_text.pt")]
            sample_ids.add(sample_id)
    return sample_ids


def validate_files(
    sample_ids: List[str],
    latents_dir: str,
    text_dir: str,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """Validate that files exist and are readable."""
    valid = []
    invalid = []

    for sample_id in sample_ids:
        latent_path = os.path.join(latents_dir, f"{sample_id}.pt")
        text_path = os.path.join(text_dir, f"{sample_id}_text.pt")

        if not os.path.exists(latent_path):
            invalid.append(f"Missing latent: {sample_id}")
            continue

        if not os.path.exists(text_path):
            invalid.append(f"Missing text: {sample_id}")
            continue

        valid.append(sample_id)

    if verbose and invalid:
        print(f"⚠ Found {len(invalid)} invalid samples:")
        for err in invalid[:10]:  # Show first 10
            print(f"  - {err}")
        if len(invalid) > 10:
            print(f"  ... and {len(invalid) - 10} more")

    return valid, invalid


def main():
    parser = argparse.ArgumentParser(
        description="Generate training CSV for MUG-V Megatron"
    )
    parser.add_argument(
        "--latents",
        type=str,
        required=True,
        help="Directory containing latent .pt files",
    )
    parser.add_argument(
        "--text-features",
        type=str,
        required=True,
        help="Directory containing text feature .pt files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--captions",
        type=str,
        default=None,
        help="Optional: captions CSV for validation (with sample_id column)",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="sample_id",
        help="Column name for sample ID in captions CSV (default: sample_id)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name for text in captions CSV (default: text)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="real",
        choices=["real", "generated"],
        help="Source type: real (applies VAE normalization), generated (no normalization). Default: real",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Use relative paths in CSV (relative to CSV location)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Scanning latents: {args.latents}")
    latent_ids = index_latents(args.latents)
    print(f"  Found {len(latent_ids)} latent files")

    print(f"Scanning text features: {args.text_features}")
    text_ids = index_text_features(args.text_features)
    print(f"  Found {len(text_ids)} text feature files")

    # Find intersection
    common_ids = latent_ids.intersection(text_ids)
    print(f"Matched {len(common_ids)} samples with both latents and text features")

    if len(common_ids) == 0:
        raise RuntimeError("No matching sample_ids between latents and text features")

    # Optional: validate against captions CSV
    if args.captions:
        print(f"Loading captions from {args.captions}")
        captions = read_captions_csv(args.captions, args.id_column, args.text_column)
        print(f"  Found {len(captions)} captions")

        # Filter to only samples with captions
        caption_ids = set(captions.keys())
        common_ids = common_ids.intersection(caption_ids)
        print(f"After caption filtering: {len(common_ids)} samples")

        if len(common_ids) == 0:
            raise RuntimeError("No samples have both data files and captions")

    # Validate files
    sample_ids = sorted(common_ids)
    valid_ids, invalid_msgs = validate_files(
        sample_ids, args.latents, args.text_features, verbose=True
    )

    if len(valid_ids) == 0:
        raise RuntimeError("No valid samples found")

    print(f"Validated {len(valid_ids)} samples")

    # Determine paths
    csv_dir = os.path.dirname(os.path.abspath(args.output))

    if args.relative_paths:
        # Make paths relative to CSV location
        latents_rel = os.path.relpath(args.latents, csv_dir)
        text_rel = os.path.relpath(args.text_features, csv_dir)
    else:
        latents_rel = "latents"
        text_rel = "text_features"

    # Write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "source", "latent_path", "text_feat_path"])

        for sample_id in valid_ids:
            writer.writerow([
                sample_id,
                args.source,
                os.path.join(latents_rel, f"{sample_id}.pt"),
                os.path.join(text_rel, f"{sample_id}_text.pt"),
            ])

    print(f"All valid {len(valid_ids)} rows to {args.output}")
    print(f"Total samples: {len(valid_ids)}")
    print(f"Source type: {args.source}")
    print(f"Latents: {args.latents}")
    print(f"Text features: {args.text_features}")
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
