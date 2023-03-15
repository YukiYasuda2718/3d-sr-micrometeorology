import torch
import torch.nn as nn


def shuffle_voxels(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    The inverse transformation is `unshuffle_voxels`
    https://github.com/dariofuoli/RLSP/blob/master/pytorch/functions.py
    https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html
    https://github.com/gap370/pixelshuffle3d/blob/master/pixelshuffle3d.py
    """
    # format: (B, C, D, H, W)
    b, c, d, h, w = x.shape

    # assert (
    #     d % factor == 0 and h % factor == 0 and w % factor == 0
    # ), f"D, H, and W must be multiples of {factor}"

    y = x.reshape(b, c, d // factor, factor, h // factor, factor, w // factor, factor)
    y = y.permute(0, 3, 5, 7, 1, 2, 4, 6)
    y = y.reshape(b, c * factor ** 3, d // factor, h // factor, w // factor)

    return y


def unshuffle_voxels(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    The inverse transformation is `shuffle_voxels`
    https://github.com/dariofuoli/RLSP/blob/master/pytorch/functions.py
    https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html
    https://github.com/gap370/pixelshuffle3d/blob/master/pixelshuffle3d.py
    """
    # format: (B, C, D, H, W)
    b, c, d, h, w = x.shape

    # assert c % factor ** 3 == 0, f"C must be a multiple of {factor ** 3}"

    y = x.reshape(b, factor, factor, factor, int(c / (factor ** 3)), d, h, w)
    y = y.permute(0, 4, 5, 1, 6, 2, 7, 3)
    y = y.reshape(b, int(c / (factor ** 3)), factor * d, factor * h, factor * w)

    return y


class VoxelShuffle(nn.Module):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = shuffle_voxels(x, self.factor)
        return y


class VoxelUnshuffle(nn.Module):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = unshuffle_voxels(x, self.factor)
        return y
