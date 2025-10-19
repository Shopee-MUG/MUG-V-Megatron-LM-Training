import copy
import io
import json
import os
import random

import numpy as np
import orjson
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image, ImageFile
from torch.utils.data import Dataset, default_collate
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path=None,
        vae_shift=0.0,
        vae_scale=1.0,
    ):
        super().__init__()
        logger.info(f"ignore loading: {data_path=}")
        self.data = pl.read_csv(data_path)
        self.data_root = os.path.dirname(data_path)
        self.vae_shift = torch.tensor(vae_shift)[:, None, None, None]
        self.vae_scale = torch.tensor(vae_scale)[:, None, None, None]
        self.device = torch.device("cpu")

    @property
    def length(self):
        return len(self.data)

    def __len__(self):
        return self.length

    def get_latent(self, sample, key="latent_path", source=None):
        latent = torch.load(
            os.path.join(self.data_root, sample[key].item()),
            weights_only=True,
            map_location=self.device,
        ).squeeze(0)
        assert source in ["real", "generated"]
        if source == "real":
            latent = (
                latent - self.vae_shift.to(latent.dtype).to(latent.device)
            ) / self.vae_scale.to(latent.dtype).to(latent.device)
        return latent

    def __getitem__(self, index):
        sample = self.data[index]
        sample_id = sample["sample_id"].item()
        source = sample["source"].item()

        text_feat = torch.load(
            os.path.join(self.data_root, sample["text_feat_path"].item()),
            weights_only=True,
            map_location=self.device,
        )
        y = text_feat["y"][-1]
        mask = text_feat["mask"][-1]

        latent = self.get_latent(sample, key="latent_path", source=source)

        height, width = latent.shape[-2:]
        height *= 8
        width *= 8
        frame_indice = torch.arange(latent.shape[1] + 2).to(torch.int64)
        x_mask = torch.ones(latent.shape[1], dtype=torch.bool)
        ret = {
            "sample_id": sample_id,
            "index": index,
            "video": latent.bfloat16(),
            "num_frames": 10,
            "height": height,
            "width": width,
            "fps": 24,
            "frame_indices": frame_indice,
            "x_mask": x_mask,
            "text": y.bfloat16(),
            "mask": mask,
            "task_type": "t2v",
            "source": source,
        }
        return ret

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        batch = torch.utils.data.default_collate(batch)
        batch = self.pack_text(batch)
        return batch

    def pack_text(self, batch):
        context = batch.pop("text")
        context_mask = batch.pop("mask")

        context = (
            context.squeeze(1)
            .masked_select(context_mask.unsqueeze(-1) != 0)
            .view(-1, context.shape[-1])
        )
        context_seqlens = context_mask.sum(dim=1)

        batch["text"] = context
        batch["context_seqlens"] = context_seqlens
        return batch
