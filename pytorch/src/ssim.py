# Reference: GitHub, jinh0park/pytorch-ssim-3D
# https://github.com/jinh0park/pytorch-ssim-3D/blob/ada88564a754cd857730d649c511384dd41f9b4e/pytorch_ssim/__init__.py
# https://pytorch.org/ignite/_modules/ignite/metrics/ssim.html#SSIM

from logging import getLogger
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable

logger = getLogger()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def uniform(window_size):
    uniform = torch.ones(window_size)
    return uniform / uniform.sum()


def create_window_3D(window_size, channel, sigma, use_gaussian=True):
    _win = uniform(window_size)
    if use_gaussian:
        _win = gaussian(window_size, sigma)

    _1D_window = _win.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = (
        _1D_window.mm(_2D_window.reshape(1, -1))
        .reshape(window_size, window_size, window_size)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    window = Variable(
        _3D_window.expand(
            channel, 1, window_size, window_size, window_size
        ).contiguous()
    )
    return window


def _ssim_3D(
    img1,
    img2,
    mask,
    window,
    window_size,
    channel,
    size_average=True,
    max_val=1.0,
    eps=1e-7,
):
    assert img1.shape == img2.shape == mask.shape

    _img1 = img1 * mask
    _img2 = img2 * mask

    mu1 = F.conv3d(_img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(_img2, window, padding=window_size // 2, groups=channel)

    weights = F.conv3d(mask, window, padding=window_size // 2, groups=channel) + eps
    mu1 = mu1 / weights
    mu2 = mu2 / weights

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv3d(_img1 * _img1, window, padding=window_size // 2, groups=channel)
        / weights
        - mu1_sq
    )
    sigma2_sq = (
        F.conv3d(_img2 * _img2, window, padding=window_size // 2, groups=channel)
        / weights
        - mu2_sq
    )
    sigma12 = (
        F.conv3d(_img1 * _img2, window, padding=window_size // 2, groups=channel)
        / weights
        - mu1_mu2
    )

    C1 = (max_val * 0.01) ** 2
    C2 = (max_val * 0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def ssim3D(
    img1,
    img2,
    mask,
    window_size=11,
    sigma=1.5,
    size_average=True,
    max_val=1.0,
    use_gaussian=True,
):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(
        window_size, channel, sigma=sigma, use_gaussian=use_gaussian
    )

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(
        img1=img1,
        img2=img2,
        mask=mask,
        window=window,
        window_size=window_size,
        channel=channel,
        size_average=size_average,
        max_val=max_val,
    )


class SSIM3D(torch.nn.Module):
    def __init__(
        self,
        window_size=11,
        sigma=1.5,
        size_average=True,
        max_val=1.0,
        eps=1e-7,
        use_gaussian=True,
    ):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.channel = 4
        self.max_val = max_val
        self.eps = eps
        self.use_gaussian = use_gaussian
        self.window = create_window_3D(
            self.window_size, self.channel, self.sigma, self.use_gaussian
        )
        logger.info(f"Use Gaussian = {self.use_gaussian}")

    def forward(self, img1, img2, mask):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(
                self.window_size, channel, self.sigma, self.use_gaussian
            )

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_3D(
            img1=img1,
            img2=img2,
            mask=mask,
            window=window,
            window_size=self.window_size,
            channel=self.channel,
            size_average=self.size_average,
            max_val=self.max_val,
            eps=self.eps,
        )