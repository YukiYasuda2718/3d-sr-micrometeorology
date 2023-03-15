import typing

import torch
import torch.nn.functional as F
from torch import nn


class MyConvWithAct1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_mode: str = None,
        act: nn.Module = None,
    ):
        super().__init__()

        self.act = act
        self.conv_mode = conv_mode

        if conv_mode is None:
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
            )
        elif conv_mode == "g_conv":
            self.conv = GatedConv3d(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
            )
        elif conv_mode == "p_conv":
            self.conv = PartialConv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                multi_channel=True,
                return_mask=True,
            )
        else:
            raise NotImplementedError(f"{conv_mode} is not supported.")

    def forward(
        self, input: torch.Tensor, mask_in: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        if self.conv_mode is None:
            output = self.conv(input)
            if self.act is not None:
                output = self.act(output)
            return output, None

        elif self.conv_mode == "p_conv":
            output, mask_out = self.conv(input, mask_in)
            if self.act is not None:
                output = self.act(output)
            return output, mask_out

        elif self.conv_mode == "g_conv":
            output, mask_out = self.conv(input)
            if self.act is not None:
                output = self.act(output)
            return mask_out * output, None

        else:
            raise NotImplementedError(f"{self.conv_mode} is not supported.")


class MyConvWithAct2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_mode: str = None,
        act: nn.Module = None,
    ):
        super().__init__()

        self.act = act
        self.conv_mode = conv_mode

        if conv_mode is None:
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
            )
        elif conv_mode == "g_conv":
            self.conv = GatedConv3d(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
            )
        elif conv_mode == "g_conv_with_separated_bias":
            self.conv = GatedConv3dWithSeparatedBias(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
            )
        else:
            raise NotImplementedError(f"{conv_mode} is not supported.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        if self.conv_mode is None:
            output = self.conv(input)
            if self.act is not None:
                output = self.act(output)
            return output

        elif self.conv_mode == "g_conv" or self.conv_mode == "g_conv_with_separated_bias":
            output, mask_out = self.conv(input)
            if self.act is not None:
                output = self.act(output)
            return mask_out * output

        else:
            raise NotImplementedError(f"{self.conv_mode} is not supported.")


class PartialConv3d(nn.Conv3d):
    """
    https://github.com/NVIDIA/partialconv/blob/610d373f35257887d45adae84c86d0ce7ad808ec/models/partialconv3d.py
    """

    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super(PartialConv3d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
            * self.weight_maskUpdater.shape[4]
        )

        self.last_size = (None, None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 5
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                            input.data.shape[4],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3], input.data.shape[4]
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv3d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv3d, self).forward(
            torch.mul(input, mask_in) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class GatedConv3d(nn.Module):
    """
    https://github.com/avalonstrel/GatedConvolution_pytorch/blob/0a49013a70e77cc484ab45a5da535c2ac003b252/models/networks.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(GatedConv3d, self).__init__()

        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.mask_conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)  # initialization of He normal

    def forward(self, input: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        output = self.conv3d(input)
        mask = self.sigmoid(self.mask_conv3d(input))

        return output, mask


class GatedConv3dWithSeparatedBias(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(GatedConv3dWithSeparatedBias, self).__init__()

        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.mask_conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True
        )
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)  # initialization of He normal

    def forward(self, input: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        output = self.conv3d(input)
        mask = self.sigmoid(self.mask_conv3d(input))

        return output, mask