import glob
import os
import pathlib
import typing
from logging import getLogger

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from src.utils import RandomCrop3D
from torch.utils.data import Dataset

logger = getLogger()


class DatasetWithoutAligningResolution(Dataset):
    def __init__(
        self,
        data_dirs: typing.List[pathlib.Path],
        hr_3d_build_path: pathlib.Path,
        means: typing.List[float] = [0.0, 0.0, 0.0, 0.0],
        stds: typing.List[float] = [1.0, 1.0, 1.0, 1.0],
        nan_value: float = 0.0,
        scale_factor: int = 4,
        hr_org_size: tuple = (32, 320, 320),
        hr_crop_size: tuple = (16, 64, 64),
        num_channels: int = 4,
        dtype: torch.dtype = torch.float32,
        use_cropping: bool = True,
        use_clipping: bool = True,
        datasize: int = None,
        seed: int = 42,
        lr_scaling: float = None,
        max_height_index: int = 32,
        max_discarded_lr_z_index: int = None,
        **kwargs,
    ):

        logger.info(f"hr_org_size = {hr_org_size}")
        logger.info(f"hr_crop_size = {hr_crop_size}")
        logger.info(f"means = {means}")
        logger.info(f"stds = {stds}")
        logger.info(f"NAN value = {nan_value}")
        logger.info(f"use_cropping = {use_cropping}")
        logger.info(f"use_clipping = {use_clipping}")
        logger.info(f"lr_scaling = {lr_scaling}")

        self.nan_value = nan_value
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.dtype = dtype
        self.lr_scaling = lr_scaling
        self.max_height_index = max_height_index
        self.max_discarded_lr_z_index = max_discarded_lr_z_index

        logger.info(f"Scale factor = {self.scale_factor}")

        self.inv_scale_factor = 1.0 / scale_factor
        assert self.inv_scale_factor in [0.25, 0.125], "Not implemented yet."

        assert hr_crop_size[0] % scale_factor == 0
        assert hr_crop_size[1] % scale_factor == 0
        assert hr_crop_size[2] % scale_factor == 0

        self.hr_org_size = hr_org_size
        self.use_cropping = use_cropping
        self.use_clipping = use_clipping

        if self.max_discarded_lr_z_index is not None:
            logger.info(f"max_discarded_lr_z_index = {self.max_discarded_lr_z_index}")
            assert self.max_height_index == 32
            assert hr_crop_size[0] == 32
            assert self.hr_org_size[0] == 32
            # no cropping along z direction

        self.random_3d_crop = RandomCrop3D(self.hr_org_size, hr_crop_size)

        hr_files, lr_files = [], []
        for dir_path in data_dirs:
            logger.info(f"Dir {dir_path} is used.")
            hr_files += [p for p in sorted(glob.glob(str(dir_path / "*_HR.npy")))]
            lr_files += [
                p
                for p in sorted(
                    glob.glob(str(dir_path / f"*_LR_x{scale_factor:02}.npy"))
                )
            ]

        assert len(hr_files) == len(lr_files)

        if datasize is not None and datasize < len(hr_files):
            logger.info(
                f"Org data size = {len(hr_files)}, but only {datasize} is used."
            )
            hr_files, lr_files = sklearn.utils.shuffle(
                hr_files, lr_files, random_state=seed, n_samples=datasize
            )
            assert len(hr_files) == len(lr_files) == datasize
        else:
            logger.info(f"Full size {len(hr_files)} is used.")

        for h, l in zip(hr_files, lr_files):
            assert (
                os.path.basename(h).split("_")[0] == os.path.basename(l).split("_")[0]
            )

        self.hr_files = hr_files
        self.lr_files = lr_files

        # Dims of `self.hr_3d_build_data` = (channel, z, y, x), where len(channel) == 1
        self.hr_3d_build_data = self._read_numpy_data(str(hr_3d_build_path))[0:1]
        assert np.sum(np.isnan(self.hr_3d_build_data.numpy())) == 0

        # To broadcast, unsqueeze dims, where axes = [channel, z, y, x]
        # add dims of z, y, and x
        if self.use_clipping:
            logger.info("torch.clamp will be applied to HR data: min = 0, max = None")

        self.means = torch.Tensor(means).to(dtype)
        self.means = self.means[..., None, None, None]

        self.stds = torch.Tensor(stds).to(dtype)
        self.stds = self.stds[..., None, None, None]

    def __len__(self):
        return len(self.hr_files)

    def _read_numpy_data(self, path: str) -> torch.Tensor:
        data = torch.from_numpy(np.load(path)).to(self.dtype)
        return data

    def _scale_and_clamp(self, data: torch.Tensor, use_clipping: bool) -> torch.Tensor:
        ret = (data - self.means) / self.stds
        if use_clipping:
            ret = torch.clamp(ret, min=0.0, max=1.0)
        return ret

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        hr_data = self._read_numpy_data(self.hr_files[idx])
        lr_data = self._read_numpy_data(self.lr_files[idx])

        if self.lr_scaling is not None:
            lr_data = self.lr_scaling * lr_data

        assert hr_data.shape[-3] == lr_data.shape[-3] * self.scale_factor
        assert hr_data.shape[-2] == lr_data.shape[-2] * self.scale_factor
        assert hr_data.shape[-1] == lr_data.shape[-1] * self.scale_factor

        # add batch dim, interpolate, and delete batch dim
        lr_data = F.interpolate(
            lr_data.unsqueeze(0), size=hr_data.shape[-3:], mode="nearest"
        ).squeeze()
        assert hr_data.shape == lr_data.shape

        hr_data = self._scale_and_clamp(hr_data, use_clipping=self.use_clipping)
        lr_data = self._scale_and_clamp(lr_data, use_clipping=True)

        # swap 0 and 1.
        bldg = self.hr_3d_build_data.clone()
        bldg = torch.where(bldg == 0, torch.ones_like(bldg), torch.zeros_like(bldg))

        stacked = torch.cat([bldg, hr_data, lr_data], dim=0)
        logger.debug(stacked.shape)

        stacked = stacked[:, : self.hr_org_size[0], ...]
        assert stacked.shape[1:] == self.hr_org_size  # verify shape of z, y, x dims
        logger.debug(stacked.shape)

        if self.use_cropping:
            stacked = self.random_3d_crop(stacked)
        else:
            stacked = stacked[:, : self.max_height_index, ...]

        stacked = torch.nan_to_num(stacked, nan=self.nan_value)

        hr_bldg = stacked[0]
        hr_data = stacked[1 : 1 + self.num_channels]
        lr_data = stacked[1 + self.num_channels :]
        assert hr_data.shape == lr_data.shape
        assert hr_bldg.shape == hr_data.shape[1:] == lr_data.shape[1:]

        # add batch dim, interpolate, and delete batch dim
        lr_data = F.interpolate(
            lr_data.unsqueeze(0), scale_factor=self.inv_scale_factor, mode="nearest"
        ).squeeze()

        # Fill the lower part of the lr input when max discarded index is specified.
        if (
            self.max_discarded_lr_z_index is not None
            and self.max_discarded_lr_z_index > 0
        ):
            logger.debug(
                f"{self.nan_value} is substituted below z idx = {self.max_discarded_lr_z_index}"
            )
            lr_data[:, : self.max_discarded_lr_z_index, ...] = self.nan_value

        return lr_data, hr_bldg, hr_data