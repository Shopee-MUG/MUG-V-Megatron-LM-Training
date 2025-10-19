from functools import partial
from inspect import signature
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

REGISTERED_ACT_DICT: dict[str, type] = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'hswish': nn.Hardswish,
    'silu': nn.SiLU,
    'gelu': partial(nn.GELU, approximate='tanh'),
}


def build_kwargs_from_config(config: dict,
                             target_func: Callable) -> dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=8,
                                  num_channels=in_channels,
                                  eps=1e-6,
                                  affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class PixelUnshuffleChannelAveragingDownSampleLayer2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        assert self.in_channels * self.factor**2 % self.out_channels == 0
        group_size = self.in_channels * self.factor**2 // self.out_channels
        assert (W % self.factor == 0 and H % self.factor
                == 0), f'{W=} or {H=} cannot be divided by {self.factor=}'
        x = x.view(
            B,
            C,
            T,
            H // self.factor,
            self.factor,
            W // self.factor,
            self.factor,
        )
        x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
        x = x.view(
            B,
            C * self.factor**2,
            T,
            H // self.factor,
            W // self.factor,
        )
        x = x.view(
            B,
            self.out_channels,
            group_size,
            T,
            H // self.factor,
            W // self.factor,
        )
        x = x.mean(dim=2)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer1D(
        PixelUnshuffleChannelAveragingDownSampleLayer2D):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        assert self.in_channels * self.factor % self.out_channels == 0
        group_size = self.in_channels * self.factor // self.out_channels
        time_pad = T % self.factor
        pad = (
            0,
            0,
            0,
            0,
            time_pad,
            0,
        )  # (left, right, top, bottom, front, back)
        x = F.pad(x, pad)
        T += time_pad
        x = x.view(
            B,
            C,
            T // self.factor,
            self.factor,
            H,
            W,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor,
            H,
            W,
        )
        x = x.view(
            B,
            self.out_channels,
            group_size,
            T // self.factor,
            H,
            W,
        )
        x = x.mean(dim=2)
        return x


class ConvPixelUnshuffleDownSampleLayer2D(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: (int, int, int),
            factor: int,
            out_ratio: int,
    ):
        super().__init__()
        self.factor = factor
        assert out_channels % out_ratio == 0
        self.conv = nn.Conv3d(in_channels,
                              out_channels // out_ratio,
                              kernel_size=kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        time_pad = self.kernel_size[0] // 2
        padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            time_pad,
        )
        x = F.pad(x, padding)
        x = self.conv(x)
        x = self.pixel_unshuffle(x, self.factor)
        return x

    @staticmethod
    def pixel_unshuffle(x: torch.Tensor, factor: int) -> torch.Tensor:
        B, C, T, H, W = x.shape
        assert (W % factor == 0 and H % factor
                == 0), f'{W=} or {H=} cannot be divided by {factor=}'
        x = x.view(B, C, T, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
        x = x.view(B, C * factor**2, T, H // factor, W // factor)
        return x


class ConvPixelUnshuffleDownSampleLayer1D(ConvPixelUnshuffleDownSampleLayer2D):

    @staticmethod
    def pixel_unshuffle(x: torch.Tensor, factor: int) -> torch.Tensor:
        B, C, T, H, W = x.shape
        time_pad = T % factor
        T += time_pad
        pad = (
            0,
            0,
            0,
            0,
            time_pad,
            0,
        )

        x = F.pad(x, pad)
        x = x.view(B, C, T // factor, factor, H, W)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, C * factor, T // factor, H, W)
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        out_ratio: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * out_ratio % in_channels == 0
        self.repeats = out_channels * out_ratio // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor,
            self.factor,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2),
            x.size(3) * self.factor,
            x.size(5) * self.factor,
        )
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer1D(
        ChannelDuplicatingPixelUnshuffleUpSampleLayer2D):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        time_pad = 1 if T == 1 else 0
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            B,
            self.out_channels,
            self.factor,
            T,
            H,
            W,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(
            B,
            self.out_channels,
            T * self.factor,
            H,
            W,
        )
        return x[:, :, time_pad:, :, :]


class ConvPixelShuffleUpSampleLayer2D(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: (int, int, int),
            factor: int,
            out_ratio: int,
    ):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv3d(in_channels,
                              out_channels * out_ratio,
                              kernel_size=kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        time_pad = self.kernel_size[0] // 2
        padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            time_pad,
        )
        x = F.pad(x, padding)
        x = self.conv(x)
        x = self.pixel_shuffle(x, self.factor)
        return x

    @staticmethod
    def pixel_shuffle(x: torch.Tensor, factor: int) -> torch.Tensor:
        batch_size, channels, depth, height, width = x.size()
        new_channels = channels // (factor**2)
        new_height = height * factor
        new_width = width * factor
        x = x.view(
            batch_size,
            new_channels,
            factor,
            factor,
            depth,
            height,
            width,
        )
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        x = x.view(batch_size, new_channels, depth, new_height, new_width)
        return x


class ConvPixelShuffleUpSampleLayer1D(ConvPixelShuffleUpSampleLayer2D):

    @staticmethod
    def pixel_shuffle(x: torch.Tensor, factor: int) -> torch.Tensor:
        batch_size, channels, depth, height, width = x.size()
        time_pad = 1 if depth == 1 else 0
        x = x.view(
            batch_size,
            channels // factor,
            factor,
            depth,
            height,
            width,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch_size, channels // factor, depth * factor, height,
                   width)
        return x[:, :, time_pad:, :, :]


class ResidualBlock(nn.Module):

    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        pre_act: Optional[nn.Module] = None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.pre_act = pre_act

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            assert (self.pre_act
                    is None), 'pre_norm and pre_act must be the same state'
            return self.main(x)
        else:
            assert (self.pre_act
                    is not None), 'pre_norm and pre_act must be the same state'
            return self.main(self.pre_act(self.pre_norm(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
        return res


class XEmbedder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        channels = 4 * in_channels * 2**3
        self.conv_in = nn.Conv3d(in_channels,
                                 channels,
                                 kernel_size=(3, 3, 3),
                                 padding=1,
                                 stride=1)
        self.downsample_2d_2x_1 = ResidualBlock(
            ConvPixelUnshuffleDownSampleLayer2D(
                channels,
                channels,
                kernel_size=(3, 3, 3),
                factor=2,
                out_ratio=2**2,
            ),
            PixelUnshuffleChannelAveragingDownSampleLayer2D(
                channels, channels, 2),
            pre_act=build_act('silu'),
            pre_norm=Normalize(channels),
        )

        # self.downsample_2d_2x_2 = ResidualBlock(
        #     ConvPixelUnshuffleDownSampleLayer2D(
        #         channels,
        #         channels,
        #         kernel_size=(3, 3, 3),
        #         factor=2,
        #         out_ratio=2**2,
        #     ),
        #     PixelUnshuffleChannelAveragingDownSampleLayer2D(
        #         channels, channels, 2
        #     ),
        #     pre_act=build_act("silu"),
        #     pre_norm=Normalize(channels),
        # )

        self.downsample_1d_2x = ResidualBlock(
            ConvPixelUnshuffleDownSampleLayer1D(
                channels,
                channels,
                kernel_size=(3, 3, 3),
                factor=2,
                out_ratio=2,
            ),
            PixelUnshuffleChannelAveragingDownSampleLayer1D(
                channels, channels, 2),
            pre_act=build_act('silu'),
            pre_norm=Normalize(channels),
        )

        self.conv_out = nn.Conv3d(channels,
                                  out_channels,
                                  kernel_size=(3, 3, 3),
                                  padding=1,
                                  stride=1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert (T == 1
                or T % 2 == 0), f'only support T==1 or T==2**n, but got {T=}'
        assert (H % 4 == 0 and W % 4 == 0
                ), f'only support H==4**n and H==4**n, but got {H=} and {W=}'
        x = self.conv_in(x)
        x = self.downsample_2d_2x_1(x)
        # x = self.downsample_2d_2x_2(x)
        x = self.downsample_1d_2x(x)
        x = self.conv_out(x)
        return x


class XProjector(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # channels = 4 * in_channels * 2**3
        channels = 4 * out_channels * 2**3
        self.conv_in = nn.Conv3d(in_channels,
                                 channels,
                                 kernel_size=(3, 3, 3),
                                 padding=1,
                                 stride=1)
        self.upsample_1d_2x = ResidualBlock(
            ConvPixelShuffleUpSampleLayer1D(
                channels,
                channels,
                kernel_size=(3, 3, 3),
                factor=2,
                out_ratio=2,
            ),
            ChannelDuplicatingPixelUnshuffleUpSampleLayer1D(
                channels,
                channels,
                2,
                out_ratio=2,
            ),
            pre_act=build_act('silu'),
            pre_norm=Normalize(channels),
        )
        self.upsample_2d_2x_1 = ResidualBlock(
            ConvPixelShuffleUpSampleLayer2D(
                channels,
                channels,
                kernel_size=(3, 3, 3),
                factor=2,
                out_ratio=2**2,
            ),
            ChannelDuplicatingPixelUnshuffleUpSampleLayer2D(
                channels,
                channels,
                2,
                out_ratio=2**2,
            ),
            pre_act=build_act('silu'),
            pre_norm=Normalize(channels),
        )
        # self.upsample_2d_2x_2 = ResidualBlock(
        #     ConvPixelShuffleUpSampleLayer2D(
        #         channels,
        #         channels,
        #         kernel_size=(3, 3, 3),
        #         factor=2,
        #         out_ratio=2**2,
        #     ),
        #     ChannelDuplicatingPixelUnshuffleUpSampleLayer2D(
        #         channels,
        #         channels,
        #         2,
        #         out_ratio=2**2,
        #     ),
        #     pre_act=build_act("silu"),
        #     pre_norm=Normalize(channels),
        # )

        self.conv_out = nn.Conv3d(channels,
                                  out_channels,
                                  kernel_size=(3, 3, 3),
                                  padding=1,
                                  stride=1)

    def forward(self, x):

        B, C, T, H, W = x.shape
        x = self.conv_in(x)
        x = self.upsample_1d_2x(x)
        x = self.upsample_2d_2x_1(x)
        # x = self.upsample_2d_2x_2(x)
        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    import torch

    batches = [
        torch.randn(1, 16, 1, 60, 60),
        torch.randn(1, 16, 1, 72, 52),
        torch.randn(1, 16, 1, 52, 72),
        torch.randn(1, 16, 1, 48, 80),
        torch.randn(1, 16, 1, 80, 48),
        torch.randn(1, 16, 30, 60, 60),
        torch.randn(1, 16, 30, 72, 52),
        torch.randn(1, 16, 30, 52, 72),
        torch.randn(1, 16, 30, 48, 80),
        torch.randn(1, 16, 30, 80, 48),
    ]

    embedder = XEmbedder(16, 3456).cuda()
    print(embedder)
    projector = XProjector(3456, 16).cuda()
    print(projector)
    __import__('ipdb').set_trace()
    for b in batches:
        b_emb = embedder(b.cuda())
        b_proj = projector(b_emb)
        print(b.shape, b_emb.shape, b_proj.shape)
