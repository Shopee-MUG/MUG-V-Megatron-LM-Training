#!/usr/bin/env python3
"""
Encode videos to latents using MUG-V VideoVAE from MUG-V.

This script encodes videos into latent representations for video generation training.
It uses the exact same VideoVAE from MUG-V to ensure consistency.

Prerequisites:
    uv pip install -r examples/mugv/data_preparation/requirements.txt

Output format:
    latents/<sample_id>.pt = FloatTensor[24, T, H, W]
    where 24 = VAE latent channels, T = temporal frames, H×W = spatial resolution

Example usage:
    python data_preparation/2_encode_video_latents.py \
        --video-dir /path/to/videos \
        --output-dir /path/to/latents \
        --vae-checkpoint /path/to/vae.pt \
        --batch-size 1 \
        --fps 24
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

# Import from MUG-V (installed via pip)
try:
    from mug_v.encoder import MUGVAE
except ImportError as e:
    raise RuntimeError(
        f"Failed to import MUGVAE from MUG-V: {e}\n"
        "Install with: uv pip install -r examples/mugv/data_preparation/requirements.txt"
    )

try:
    import av

    HAVE_AV = True
except ImportError:
    HAVE_AV = False

try:
    import decord

    HAVE_DECORD = True
except ImportError:
    HAVE_DECORD = False


def list_videos(video_dir: str) -> List[str]:
    """List all video files in directory."""
    exts = (".mp4", ".webm", ".mkv", ".mov", ".avi")
    videos = []
    for f in os.listdir(video_dir):
        if f.lower().endswith(exts):
            videos.append(os.path.join(video_dir, f))
    return sorted(videos)


def read_video_av(path: str, fps: int = 24) -> torch.Tensor:
    """Read video using PyAV."""
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    frames = []
    for frame in container.decode(stream):
        img = frame.to_rgb().to_ndarray()  # HWC uint8
        frames.append(img)
    container.close()

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from {path}")

    # [T, H, W, C] -> [T, C, H, W]
    import numpy as np

    frames = np.stack(frames)  # [T, H, W, C]
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return frames


def read_video_decord(path: str, fps: int = 24) -> torch.Tensor:
    """Read video using Decord."""
    vr = decord.VideoReader(path, num_threads=1)
    frames = vr[:].asnumpy()  # [T, H, W, C] uint8
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return frames


def read_video(path: str, fps: int = 24) -> torch.Tensor:
    """Read video using available backend."""
    if HAVE_AV:
        return read_video_av(path, fps)
    elif HAVE_DECORD:
        return read_video_decord(path, fps)
    else:
        raise RuntimeError(
            "No video reading backend available. "
            "Install PyAV or Decord:\n"
            "  uv pip install av\n"
            "  uv pip install decord"
        )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Encode videos using VideoVAE")
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for latents",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        required=True,
        help="Path to VideoVAE checkpoint",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target FPS for video reading (default: 24)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1, increase if you have enough VRAM)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile for faster encoding",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # List videos
    videos = list_videos(args.video_dir)
    if len(videos) == 0:
        raise RuntimeError(f"No videos found in {args.video_dir}")

    print(f"Found {len(videos)} videos in {args.video_dir}")

    # Load VideoVAE
    print(f"Loading VideoVAE from {args.vae_checkpoint}")
    vae = MUGVAE(from_pretrained=args.vae_checkpoint)
    vae = vae.to(args.device, dtype=torch.bfloat16).eval()

    if args.compile:
        print("Compiling VAE with torch.compile...")
        vae = torch.compile(vae)

    print(f"VAE loaded on {args.device}")

    # Encode videos
    for video_path in tqdm(videos, desc="Encoding videos"):
        sample_id = Path(video_path).stem

        try:
            # Read video
            frames = read_video(video_path, args.fps)  # [T, 3, H, W]
            frames = frames.unsqueeze(0)  # [1, T, 3, H, W]
            frames = frames.permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]
            frames = frames.to(args.device, dtype=torch.bfloat16)

            # Encode
            latents = vae.encode(frames)  # [1, 24, T', H', W']
            latents = latents.squeeze(0).cpu().float()  # [24, T', H', W']

            # Save
            save_path = os.path.join(args.output_dir, f"{sample_id}.pt")
            torch.save(latents, save_path)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

    print(f"Encoded {len(videos)} videos to {args.output_dir}")


if __name__ == "__main__":
    main()
