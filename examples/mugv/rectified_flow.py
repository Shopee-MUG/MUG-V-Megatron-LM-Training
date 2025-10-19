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

from typing import List

import numpy as np
import torch
from einops import rearrange
from mugdit_tracker import task_loss_tracker
from random_utils import controlled_random_state_torch, nullcontext
from torch.distributions import LogisticNormal

from megatron.core import mpu, tensor_parallel
from megatron.training import get_args, get_tensorboard_writer

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def normalize_latent(x, max_val, quantile_val, get_max=True, std_threshold=1.0):
    x = x.detach().clone()

    x_abs = x.abs()
    x_reshaped = rearrange(x_abs, "b c ... -> b c (...)")

    s = torch.quantile(
        x_reshaped,
        quantile_val,
        dim=-1,
        keepdim=True,
    )
    s = s.squeeze(-1)
    for i in range(len(max_val)):
        if max_val[i] > 0:
            if get_max:
                s[:, i] = torch.maximum(
                    s[:, i], torch.tensor(max_val[i], dtype=s.dtype)
                )
            else:
                s[:, i] = torch.minimum(
                    s[:, i], torch.tensor(max_val[i], dtype=s.dtype)
                )
        else:
            s[:, i] = torch.tensor(1.0, dtype=s.dtype)
    s = right_pad_dims_to(x, s)
    no_clamp_channel_ids = [
        0,
        1,
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    ]
    s[:, no_clamp_channel_ids] = 10000.0

    x = x.clamp(-s, s)
    print(f"clip with threshold {s.view(-1)}")
    return x


def _extract_into_tensor(
    arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: List[int]
):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def mean_flat(tensor: torch.Tensor, mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        denom = mask.sum(dim=1) * tensor.shape[-1]
        loss = (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom
        return loss


def sum_flat_mean_logic(tensor: torch.Tensor, mask=None):
    """
    Compute total loss and total token number in a way that is equivalent
    to the logic in mean_flat, but returns the sum over all tokens and
    total token count.
    """
    if mask is None:
        return (
            tensor.mean(dim=list(range(1, len(tensor.shape)))) * tensor.shape[0],
            torch.prod(torch.tensor(tensor.shape, device=tensor.device)),
        )
    else:
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        denom = mask.sum(dim=1) * tensor.shape[-1]
        token_num = denom.sum()
        mean_loss_per_sample = (tensor * mask.unsqueeze(2)).sum(dim=[1, 2]) / denom
        # take batchsize as dummy token num
        dummy_token_num = torch.tensor(
            mean_loss_per_sample.shape, device=mean_loss_per_sample.device
        )
        return (mean_loss_per_sample.sum(), dummy_token_num)


def sum_flat(tensor: torch.Tensor, mask=None):
    """
    Take the sum over all dimensions, including the batch dimension.
    """
    if mask is None:
        return tensor.sum(), torch.prod(
            torch.tensor(tensor.shape, device=tensor.device)
        )
    else:
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        token_num = mask.sum() * tensor.shape[-1]
        loss = (tensor * mask.unsqueeze(2)).sum()
        return loss, token_num


def timestep_transform(
    t,
    model_kwargs,
    vae_type,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()

    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        assert vae_type in {"3d", "2d"}, vae_type
        num_frames = model_kwargs["num_frames"]
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


def controlled_random_state(seed=None):
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        # NOTE: For pipeline parallelism, the entire add-noise process must use a fixed random seed.
        # Because this step runs at both the first and last pipeline stages, using a consistent seed
        # ensures identical noise is generated across these stages.
        assert seed is not None, (
            "logistic_normal_sample requires a seed for pipeline parallelism"
        )
        return controlled_random_state_torch(seed)

    else:
        return nullcontext()


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
        vae_type="3d",
        cfg_scale=4.0,
        dynamic_clamp=False,
        guidance_rescale=None,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps
        self.vae_type = vae_type
        self.dynamic_clamp = dynamic_clamp
        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert sample_method == "uniform" or not use_discrete_timesteps, (
            "Only uniform sampling is supported for discrete timesteps"
        )
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(
                torch.tensor([loc]), torch.tensor([scale])
            )
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(
                x.device
            )

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

        # inference config
        self.cfg_scale = cfg_scale
        self.guidance_rescale = guidance_rescale

        latent_shift = (
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
        latent_scale = (
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

        self.latent_scale = torch.tensor(latent_scale)
        self.latent_shift = torch.tensor(latent_shift)

        if len(self.latent_scale.shape) > 0:
            self.latent_scale = self.latent_scale[None, :, None, None, None]
        if len(self.latent_shift.shape) > 0:
            self.latent_shift = self.latent_shift[None, :, None, None, None]

    def prepare_loss_args(
        self, x_start, noise=None, t=None, mask=None, model_kwargs={}, seed=None
    ):
        with controlled_random_state(seed):
            if noise is None:
                noise = torch.randn_like(x_start)
            assert noise.shape == x_start.shape

            if t is None:
                if self.use_discrete_timesteps:
                    t = torch.randint(
                        0,
                        self.num_timesteps,
                        (x_start.shape[0],),
                        device=x_start.device,
                    )
                elif self.sample_method == "uniform":
                    t = (
                        torch.rand((x_start.shape[0],), device=x_start.device)
                        * self.num_timesteps
                    )
                elif self.sample_method == "logit-normal":
                    t = self.sample_t(x_start) * self.num_timesteps

                if self.use_timestep_transform:
                    t = timestep_transform(
                        t,
                        model_kwargs,
                        vae_type=self.vae_type,
                        scale=self.transform_scale,
                        num_timesteps=self.num_timesteps,
                    )

            x_t = self.add_noise(x_start, noise, t)
            if mask is not None:
                t0 = torch.zeros_like(t)
                x_t0 = self.add_noise(x_start, noise, t0)
                x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

            model_kwargs |= {"noisy_latent": x_t, "timestep": t, "noise": noise}
            return model_kwargs

    def smooth_l1_percentile_loss(self, pred, target, beta=0.6, percentile=95):
        diff = torch.abs(pred - target)
        l2_part = 0.5 * (diff**2) / beta  # L2 loss (small error)
        l1_part = diff - 0.5 * beta  # L1 loss (large error)
        loss = torch.where(diff < beta, l2_part, l1_part)
        B = diff.shape[0]
        high_loss_mask = torch.zeros_like(diff, dtype=torch.bool)
        for i in range(B):
            flat_diff = diff[i].flatten()
            k = max(1, int(flat_diff.numel() * (percentile / 100)))
            threshold_value = torch.kthvalue(flat_diff, k)[0]
            high_loss_mask[i] = diff[i] >= threshold_value
        loss = torch.where(high_loss_mask, l1_part, loss)
        return loss

    def loss_func(
        self,
        output_tensor,
        x_start,
        noise,
        t,
        mask,
        weights=None,
        data_group=None,
        logging_stats={},
    ):
        # TODO: check mean logic and tokens count, currently we take the pixel-level token.
        #       this logic is not consistent with the hf implementation.

        output_tensor = output_tensor.to(torch.float32)
        velocity_pred = output_tensor.chunk(2, dim=1)[0]
        # l1_pred_diff = self.smooth_l1_percentile_loss(velocity_pred, x_start - noise, beta=0.6, percentile=95)
        l1_pred_diff = (velocity_pred - (x_start - noise)).pow(2)
        if weights is None:
            total_loss, total_tokens = sum_flat_mean_logic(l1_pred_diff, mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            total_loss, total_tokens = sum_flat_mean_logic(
                weight * l1_pred_diff, mask=mask
            )

        # NOTE: Add t2i weight for rescaling down the loss.
        if "task_counter" in logging_stats:
            task_counter = logging_stats["task_counter"]
            if task_counter["t2i"].item() > 0:
                print(
                    f"t2i task count: {task_counter['t2i'].item()}, loss: {total_loss.item()}"
                )
                total_loss = total_loss * 0.3

        loss = torch.cat([total_loss.view(1), total_tokens.view(1)])

        reporting_loss = loss.clone().detach()
        torch.distributed.all_reduce(reporting_loss, group=data_group)

        local_num_tokens = loss[1].clone().detach().to(torch.int)

        # Write training stats to tensorboard
        # NOTE: We delay the logic for writing to TensorBoard until after the loss function
        # because we want to record the loss split by task. This logic is somewhat unusual,
        # but we couldn't find a more appropriate place for it.

        writer = get_tensorboard_writer()
        args = get_args()
        if writer and (args.curr_iteration % args.tensorboard_log_interval == 0):
            if "task_counter" in logging_stats:
                task_counter = logging_stats.pop("task_counter")
                for task, num in task_counter.items():
                    if num.item() > 0:
                        cnt, avg_loss = task_loss_tracker(
                            total_loss / local_num_tokens, task
                        )
                        writer.add_scalar(
                            f"task-tracker/{task}/loss", avg_loss, args.curr_iteration
                        )
                        writer.add_scalar(
                            f"task-tracker/{task}/cnt", cnt, args.curr_iteration
                        )
                        break
            for tab, info in logging_stats.items():
                for k, v in info.items():
                    writer.add_scalar(f"{tab}/{k}", v, args.curr_iteration)

        return (
            total_loss,
            local_num_tokens,
            {
                "lm loss": reporting_loss,
            },
        )

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(
            1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4]
        )

        return timepoints * original_samples + (1 - timepoints) * noise

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args={},
        mask=None,
        guidance_scale=None,
        guidance_rescale=None,
        dtype=None,
        reverse_start_step=None,
    ):
        max_val_list = [
            0.20,
            0.50,
            0.20,
            0.20,
            0.15,
            0.50,
            0.20,
            0.20,
            0.15,
            0.15,
            0.20,
            0.50,
            0.50,
            0.15,
            0.20,
            0.20,
            0.50,
            0.20,
            0.15,
            0.20,
            0.20,
            0.20,
            0.15,
            0.15,
        ]
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        negative = "ugly, blurry, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        print(f"use negative prompts {negative}")

        n = len(prompts)
        prompts = prompts + [negative] * n
        model_args = text_encoder.encode(prompts)
        model_args.update(additional_args)

        # prepare timesteps
        timesteps = [
            (1.0 - i / self.num_sampling_steps) * self.num_timesteps
            for i in range(self.num_sampling_steps)
        ]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(
                    t,
                    additional_args,
                    vae_type=self.vae_type,
                    num_timesteps=self.num_timesteps,
                )
                for t in timesteps
            ]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)
        for i, t in enumerate(timesteps):
            if reverse_start_step is not None:
                if reverse_start_step >= len(timesteps):
                    print("directly return z")
                    return z
                elif reverse_start_step == i:
                    print(
                        "adding noise to the generated input to get the reverse inital at step",
                        reverse_start_step,
                    )
                    x0 = z.clone()
                    x_noise = self.add_noise(x0, torch.randn_like(x0), t)
                    z = torch.where((mask[:, None, :, None, None] == 1), x_noise, x0)
                elif reverse_start_step > i:
                    print(
                        f"jump step {i} to reach the start reverse step {reverse_start_step}"
                    )
                    continue

            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)

            noisy_latent = z_in
            temporal_mask = model_args["x_mask"]
            context = model_args["y"]
            context_mask = model_args["mask"]
            timestep = t
            fps = model_args["fps"]
            num_frames = model_args["num_frames"]
            frame_indices = model_args["frame_indices"]
            height = model_args["height"]
            width = model_args["width"]

            context = (
                context.squeeze(1)
                .masked_select(context_mask.unsqueeze(-1) != 0)
                .view(-1, context.shape[-1])
            )
            context_seqlens = context_mask.sum(dim=1)

            # __import__('ipdb').set_trace()
            model_inputs = {
                "x": noisy_latent.to(torch.bfloat16),
                "attention_mask": None,
                "temporal_mask": temporal_mask,
                "context": context.to(torch.bfloat16),
                "context_seqlens": context_seqlens,
                "timestep": timestep.to(torch.bfloat16),
                "fps": fps,
                "num_frames": num_frames,
                "frame_indices": frame_indices,
                "height": height,
                "width": width,
            }
            # for k, v in model_inputs.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f'{k}: {v.shape} {v.dtype} {v.device}')
            pred = model(**model_inputs).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)

            delta_ = pred_cond - pred_uncond
            print(f"{self.dynamic_clamp=}")
            if self.dynamic_clamp:
                if t[0].item() > 950.0:
                    delta_clamped = normalize_latent(
                        delta_, max_val=max_val_list, quantile_val=0.98, get_max=True
                    )
                    # delta_clamped = delta_
                    v_pred = pred_uncond + guidance_scale * delta_clamped
                elif t[0].item() > 400.0:
                    delta_clamped = normalize_latent(
                        delta_, max_val=max_val_list, quantile_val=0.98, get_max=False
                    )
                    # delta_clamped = delta_
                    v_pred = pred_uncond + guidance_scale * delta_clamped
                else:
                    delta_clamped = normalize_latent(
                        delta_, max_val=max_val_list, quantile_val=0.95, get_max=False
                    )
                    # delta_clamped = delta_
                    v_pred = pred_uncond + guidance_scale * delta_clamped
                delta_final = delta_clamped
            else:
                delta_final = delta_
                v_pred = pred_uncond + guidance_scale * delta_

            # rescale 'noise_cfg' according to 'guidance_rescale'
            if self.guidance_rescale is not None:
                std_cond = pred_cond.std(
                    dim=list(range(1, pred_cond.ndim)), keepdim=True
                )
                std_v = v_pred.std(dim=list(range(1, v_pred.ndim)), keepdim=True)
                rescale_factor = std_cond / std_v
                print(f"rescale_factor: {rescale_factor.item()}")
                rescaled_v_pred = v_pred * rescale_factor
                v_pred = (
                    self.guidance_rescale * rescaled_v_pred
                    + (1 - self.guidance_rescale) * v_pred
                )

            # update z
            dt = (
                timesteps[i] - timesteps[i + 1]
                if i < len(timesteps) - 1
                else timesteps[i]
            )
            dt = dt / self.num_timesteps

            z = z + v_pred * dt[:, None, None, None, None]
            printdata = delta_final.clone()
            mean_ = torch.mean(printdata, dim=[2, 3, 4], keepdim=True)
            std_ = torch.std(printdata, dim=[2, 3, 4], keepdim=True)
            max_, _ = torch.max(printdata, dim=2, keepdim=True)
            max_, _ = torch.max(max_, dim=3, keepdim=True)
            max_, _ = torch.max(max_, dim=4, keepdim=True)
            min_, _ = torch.min(printdata, dim=2, keepdim=True)
            min_, _ = torch.min(min_, dim=3, keepdim=True)
            min_, _ = torch.min(min_, dim=4, keepdim=True)
            print(t[0].item())
            print(f"mean={torch.squeeze_copy(mean_)}")
            print(f"std={torch.squeeze_copy(std_)}")
            print(f"max={torch.squeeze_copy(max_)}")
            print(f"min={torch.squeeze_copy(min_)}")

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
