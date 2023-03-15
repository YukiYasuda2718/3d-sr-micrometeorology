import typing
from logging import getLogger

import torch
import torch.nn as nn

from model.custom_conv import MyConvWithAct2
from model.voxel_shuffle import VoxelUnshuffle

logger = getLogger()


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        conv_mode: str,
        n_layers_in_block: int,
    ):
        super(DownBlock, self).__init__()

        assert n_layers_in_block >= 1

        layers = [
            MyConvWithAct2(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias,
                conv_mode=conv_mode,
                act=nn.ReLU(),
            )
        ]

        for _ in range(n_layers_in_block - 1):
            layers.append(
                MyConvWithAct2(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                    conv_mode=conv_mode,
                    act=nn.ReLU(),
                )
            )

        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in1_channels: int,
        in2_channels: int,
        out_channels: int,
        bias: bool,
        conv_mode: str,
        n_layers_in_block: int,
    ):
        super(UpBlock, self).__init__()

        assert n_layers_in_block >= 1

        layers = [
            MyConvWithAct2(
                in1_channels + in2_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_mode=conv_mode,
                act=nn.LeakyReLU(),
            )
        ]

        for _ in range(n_layers_in_block - 1):
            layers.append(
                MyConvWithAct2(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                    conv_mode=conv_mode,
                    act=nn.LeakyReLU(),
                )
            )

        self.convs = nn.Sequential(*layers)

        self.up = nn.Sequential(
            nn.Conv3d(
                in1_channels,
                in1_channels * 8,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(),
            VoxelUnshuffle(factor=2),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x3 = self.up(x1)

        y = torch.cat([x2, x3], dim=1)  # concat along channel dim

        return self.convs(y)


class UNetSR(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        num_feat0: int = 64,
        num_feat1: int = 128,
        num_feat2: int = 256,
        num_feat3: int = 512,
        num_feat4: int = None,
        num_x2upsample: int = 2,
        num_latent_layers: int = 3,
        bias_feat_extraction: bool = False,
        conv_mode_feat_extraction: str = "g_conv_with_separated_bias",
        conv_mode_down_block: str = "g_conv_with_separated_bias",
        conv_mode_up_block: str = None,
        n_layers_in_block: int = 2,
    ):
        super(UNetSR, self).__init__()

        logger.info(f"conv_mode_feat_extraction = {conv_mode_feat_extraction}")
        logger.info(f"conv_mode_down_block = {conv_mode_down_block}")
        logger.info(f"conv_mode_up_block = {conv_mode_up_block}")

        self.up0 = nn.Upsample(scale_factor=2 ** num_x2upsample, mode="nearest")

        # `+ 1` in channel is necessary to concatenate with building data
        self.conv0 = MyConvWithAct2(
            in_channels=in_channels + 1,
            out_channels=num_feat0,
            kernel_size=3,
            padding=1,
            bias=bias_feat_extraction,
            conv_mode=conv_mode_feat_extraction,
            act=None,
        )

        self.down = nn.AvgPool3d(kernel_size=2, stride=2)

        # `+ 1` in channel is necessary to concatenate with building data
        self.down1 = DownBlock(
            in_channels=num_feat0 + 1,
            out_channels=num_feat1,
            bias=False,
            conv_mode=conv_mode_down_block,
            n_layers_in_block=n_layers_in_block,
        )
        self.down2 = DownBlock(
            in_channels=num_feat1 + 1,
            out_channels=num_feat2,
            bias=False,
            conv_mode=conv_mode_down_block,
            n_layers_in_block=n_layers_in_block,
        )
        self.down3 = DownBlock(
            in_channels=num_feat2 + 1,
            out_channels=num_feat3,
            bias=False,
            conv_mode=conv_mode_down_block,
            n_layers_in_block=n_layers_in_block,
        )
        self.down4 = None

        if num_feat4 is not None and num_feat4 > 0:
            self.down4 = DownBlock(
                in_channels=num_feat3 + 1,
                out_channels=num_feat4,
                bias=False,
                conv_mode=conv_mode_down_block,
                n_layers_in_block=n_layers_in_block,
            )

        # `+ 1` in channel is necessary to concatenate with building data
        latent_layers = []
        for i in range(num_latent_layers):
            _in = num_feat3 if i > 0 else num_feat3 + 1
            latent_layers.append(
                nn.Conv3d(_in, num_feat3, kernel_size=3, padding=1, bias=False)
            )
            latent_layers.append(nn.LeakyReLU())
        self.latent_layers = nn.Sequential(*latent_layers)

        self.up4 = None

        if num_feat4 is not None and num_feat4 > 0:
            # `+ 1` in channel is necessary to concatenate with building data
            self.up4 = UpBlock(
                in1_channels=num_feat4 + 1,
                in2_channels=num_feat3 + 1,
                out_channels=num_feat3,
                bias=False,
                conv_mode=conv_mode_up_block,
                n_layers_in_block=n_layers_in_block,
            )

        # `+ 1` in channel is necessary to concatenate with building data
        self.up3 = UpBlock(
            in1_channels=num_feat3 + 1,
            in2_channels=num_feat2 + 1,
            out_channels=num_feat2,
            bias=False,
            conv_mode=conv_mode_up_block,
            n_layers_in_block=n_layers_in_block,
        )
        self.up2 = UpBlock(
            in1_channels=num_feat2 + 1,
            in2_channels=num_feat1 + 1,
            out_channels=num_feat1,
            bias=False,
            conv_mode=conv_mode_up_block,
            n_layers_in_block=n_layers_in_block,
        )
        self.up1 = UpBlock(
            in1_channels=num_feat1 + 1,
            in2_channels=num_feat0 + 1,
            out_channels=num_feat0,
            bias=False,
            conv_mode=conv_mode_up_block,
            n_layers_in_block=n_layers_in_block,
        )

        self.last = nn.Conv3d(
            num_feat0 + in_channels + 1,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def get_last_params(self) -> typing.List[torch.nn.Parameter]:
        return list(self.last.parameters())
        # Ref:
        # https://github.com/AvivNavon/AuxiLearn/blob/8ff7dd28ab045a817757a5970cd02f14af983e3d/experiments/nyuv2/trainer_baseline.py#L157-L160

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x0 = self.up0(x)
        x0 = torch.cat([x0, b], dim=1)

        y0 = self.conv0(x0)
        y0 = torch.cat([y0, b], dim=1)

        y1 = self.down1(y0)
        b1 = self.down(b)
        y1 = torch.cat([y1, b1], dim=1)

        y2 = self.down2(y1)
        b2 = self.down(b1)
        y2 = torch.cat([y2, b2], dim=1)

        y3 = self.down3(y2)
        b3 = self.down(b2)
        y3 = torch.cat([y3, b3], dim=1)

        if self.down4 is None and self.up4 is None:
            y = self.latent_layers(y3)
        else:
            y4 = self.down4(y3)
            b4 = self.down(b3)
            y4 = torch.cat([y4, b4], dim=1)

            y = self.latent_layers(y4)

            y = torch.cat([y, b4], dim=1)
            y = self.up4(y, y3)

        y = torch.cat([y, b3], dim=1)

        y = self.up3(y, y2)
        y = torch.cat([y, b2], dim=1)

        y = self.up2(y, y1)
        y = torch.cat([y, b1], dim=1)

        y = self.up1(y, y0)
        y = torch.cat([y, x0], dim=1)  # concat along channel dim

        y = self.last(y)

        return y