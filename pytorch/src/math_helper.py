import torch
import torch.nn.functional as F
from torch import nn


def differentiate_along_x(
    xs: torch.Tensor, delta: float = 1.0, padding: int = 1
) -> torch.Tensor:
    assert len(xs.shape) == 5  # batch, channel, z, y, x

    n_channels = xs.shape[1]
    weight = torch.zeros(
        size=(n_channels, 1, 3, 3, 3), dtype=xs.dtype, device=xs.device
    )

    weight[:, :, 1, 1, 0] = -1 / (2 * delta)
    weight[:, :, 1, 1, 2] = 1 / (2 * delta)

    # depth-wise derivative
    ys = F.conv3d(xs, weight, padding=padding, groups=n_channels)

    return ys


def differentiate_along_y(
    xs: torch.Tensor, delta: float = 1.0, padding: int = 1
) -> torch.Tensor:
    assert len(xs.shape) == 5  # batch, channel, z, y, x

    n_channels = xs.shape[1]
    weight = torch.zeros(
        size=(n_channels, 1, 3, 3, 3), dtype=xs.dtype, device=xs.device
    )

    weight[:, :, 1, 0, 1] = -1 / (2 * delta)
    weight[:, :, 1, 2, 1] = 1 / (2 * delta)

    # depth-wise derivative
    ys = F.conv3d(xs, weight, padding=padding, groups=n_channels)

    return ys


def differentiate_along_z(
    xs: torch.Tensor, delta: float = 1.0, padding: int = 1
) -> torch.Tensor:
    assert len(xs.shape) == 5  # batch, channel, z, y, x

    n_channels = xs.shape[1]
    weight = torch.zeros(
        size=(n_channels, 1, 3, 3, 3), dtype=xs.dtype, device=xs.device
    )

    weight[:, :, 0, 1, 1] = -1 / (2 * delta)
    weight[:, :, 2, 1, 1] = 1 / (2 * delta)

    # depth-wise derivative
    ys = F.conv3d(xs, weight, padding=padding, groups=n_channels)

    return ys


def _differentiate_along_x(xs: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    assert len(xs.shape) == 5  # batch, channel, z, y, x

    _ys = torch.zeros_like(xs)

    for k in range(1, xs.shape[2] - 1):
        for j in range(1, xs.shape[3] - 1):
            for i in range(1, xs.shape[4] - 1):
                _ys[:, :, k, j, i] = (xs[:, :, k, j, i + 1] - xs[:, :, k, j, i - 1]) / (
                    2 * delta
                )

    return _ys


def _differentiate_along_y(xs: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    assert len(xs.shape) == 5  # batch, channel, z, y, x

    _ys = torch.zeros_like(xs)

    for k in range(1, xs.shape[2] - 1):
        for j in range(1, xs.shape[3] - 1):
            for i in range(1, xs.shape[4] - 1):
                _ys[:, :, k, j, i] = (xs[:, :, k, j + 1, i] - xs[:, :, k, j - 1, i]) / (
                    2 * delta
                )

    return _ys


def _differentiate_along_z(xs: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    assert len(xs.shape) == 5  # batch, channel, z, y, x

    _ys = torch.zeros_like(xs)

    for k in range(1, xs.shape[2] - 1):
        for j in range(1, xs.shape[3] - 1):
            for i in range(1, xs.shape[4] - 1):
                _ys[:, :, k, j, i] = (xs[:, :, k + 1, j, i] - xs[:, :, k - 1, j, i]) / (
                    2 * delta
                )

    return _ys