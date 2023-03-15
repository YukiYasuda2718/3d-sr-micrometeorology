from copy import deepcopy
from logging import getLogger
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from src.math_helper import (
    differentiate_along_x,
    differentiate_along_y,
    differentiate_along_z,
)
from src.ssim import SSIM3D
from torch import nn

logger = getLogger()


def make_loss(config: dict) -> nn.Module:

    if config["train"]["loss"]["name"] == "L1":
        logger.info("L1 loss is created.")
        return MyL1Loss()
    elif config["train"]["loss"]["name"] == "L2":
        logger.info("L2 loss is created.")
        return MyL2Loss()
    elif config["train"]["loss"]["name"] == "WeightedL1":
        logger.info("Weighted L1 loss is created.")
        return WeightedL1Loss(config["train"]["loss"]["weight_outside_building"])
    elif config["train"]["loss"]["name"] == "WeightedL2":
        logger.info("Weighted L2 loss is created.")
        return WeightedL2Loss(config["train"]["loss"]["weight_outside_building"])
    elif config["train"]["loss"]["name"] == "MixedGradientL2Loss":
        logger.info("Mixed gradient L2 loss is created.")
        return MixedGradientL2Loss(
            weight_gradient_loss=config["train"]["loss"].get(
                "weight_gradient_loss", None
            ),
        )
    elif config["train"]["loss"]["name"] == "MixedDivergenceGradientL2Loss":
        logger.info("MixedDivergenceGradientL2Loss is created")
        return MixedDivergenceGradientL2Loss(
            weight_gradient_loss=config["train"]["loss"].get(
                "weight_gradient_loss", 0.0
            ),
            weight_divergence_loss=config["train"]["loss"].get(
                "weight_divergence_loss", 0.0
            ),
            scales=config["data"]["stds"][1:],
        )
    else:
        raise NotImplementedError(
            f'{config["train"]["loss"]["name"]} is not supported.'
        )


def calc_mask_near_build_wall(
    building: torch.Tensor, num_filter_applications: int = 1
) -> torch.Tensor:

    assert len(building.shape) == 5  # dims = [batch, channel, z, y, x]
    is_in_build = 1 - building
    n_channels = is_in_build.shape[1]

    weight = torch.ones(
        size=(n_channels, 1, 3, 3, 3),
        dtype=is_in_build.dtype,
        device=is_in_build.device,
    )
    filtered = is_in_build
    for _ in range(num_filter_applications):
        filtered = F.conv3d(filtered, weight, padding=1, groups=n_channels)

    filtered = torch.where(
        filtered > 0, torch.ones_like(filtered), torch.zeros_like(filtered)
    )

    is_near_wall = torch.where(
        filtered * building > 0, torch.ones_like(building), torch.zeros_like(building)
    )
    is_near_wall.requires_grad = False

    return is_near_wall


def calc_residual_continuity_eq(
    bs: torch.Tensor, preds: torch.Tensor, scales: List[float], delta_meter: float = 5.0
) -> Tuple[torch.Tensor, float]:

    assert len(preds.shape) == len(bs.shape) == 5
    assert preds.shape[2:] == bs.shape[2:]  # z,y,x dims
    assert preds.shape[0] == bs.shape[0]  # batch dim

    # The first channel is temperature, so preds[:, 1:, ...] is used.
    _scales = torch.tensor(scales, device=preds.device)
    velocity = _scales[None, :, None, None, None] * preds[:, 1:, ...]

    continuity_eq = _calc_residual_continuity_eq(velocity, delta_meter)
    is_near_wall = calc_mask_near_build_wall(bs)

    continuity_eq = continuity_eq[..., 1:-1, 1:-1, 1:-1]
    _bs = bs[..., 1:-1, 1:-1, 1:-1]
    _is_near_wall = is_near_wall[..., 1:-1, 1:-1, 1:-1]

    assert continuity_eq.shape == _bs.shape == _is_near_wall.shape

    continuity_eq = continuity_eq * _bs
    continuity_eq = continuity_eq * (1 - _is_near_wall)

    num_grids = torch.sum(_bs) - torch.sum(_is_near_wall)

    return continuity_eq, num_grids


def _calc_residual_continuity_eq(
    velocity: torch.Tensor,
    delta_meter: float = 5.0,
    padding: int = 1,
) -> torch.Tensor:

    assert len(velocity.shape) == 5
    assert velocity.shape[1] == 3  # 3 channels, u, v, w

    dudx = differentiate_along_x(velocity[:, 0:1], delta_meter, padding)
    dvdy = differentiate_along_y(velocity[:, 1:2], delta_meter, padding)
    dwdz = differentiate_along_z(velocity[:, 2:3], delta_meter, padding)

    residual = dudx + dvdy + dwdz

    return residual


def calc_vorticity_vector(
    bs: torch.Tensor, preds: torch.Tensor, scales: List[float], delta_meter: float = 5.0
) -> Tuple[torch.Tensor, float]:

    assert len(preds.shape) == len(bs.shape) == 5
    assert preds.shape[2:] == bs.shape[2:]  # z,y,x dims
    assert preds.shape[0] == bs.shape[0]  # batch dim

    # The first channel is temperature, so preds[:, 1:, ...] is used.
    _scales = torch.tensor(scales, device=preds.device)
    velocity = _scales[None, :, None, None, None] * preds[:, 1:, ...]

    omega = _calc_vorticity_vector(velocity, delta_meter)
    is_near_wall = calc_mask_near_build_wall(bs)

    omega = omega[..., 1:-1, 1:-1, 1:-1]
    _bs = bs[..., 1:-1, 1:-1, 1:-1]
    _is_near_wall = is_near_wall[..., 1:-1, 1:-1, 1:-1]

    assert omega.shape[0] == _bs.shape[0] == _is_near_wall.shape[0]
    assert omega.shape[2:] == _bs.shape[2:] == _is_near_wall.shape[2:]
    assert _bs.shape[1] == _is_near_wall.shape[1] == 1

    omega = omega * _bs
    omega = omega * (1 - _is_near_wall)

    num_grids = torch.sum(_bs) - torch.sum(_is_near_wall)

    return omega, num_grids


def _calc_vorticity_vector(
    velocity: torch.Tensor,
    delta_meter: float = 5.0,
    padding: int = 1,
) -> torch.Tensor:

    assert len(velocity.shape) == 5
    assert velocity.shape[1] == 3  # 3 channels, u, v, w

    grd_x = differentiate_along_x(velocity, delta_meter, padding)
    grd_y = differentiate_along_y(velocity, delta_meter, padding)
    grd_z = differentiate_along_z(velocity, delta_meter, padding)

    dwdy = grd_y[:, 2:3]
    dvdz = grd_z[:, 1:2]
    vor_x = dwdy - dvdz

    dudz = grd_z[:, 0:1]
    dwdx = grd_x[:, 2:3]
    vor_y = dudz - dwdx

    dvdx = grd_x[:, 1:2]
    dudy = grd_y[:, 0:1]
    vor_z = dvdx - dudy

    vor = torch.cat([vor_x, vor_y, vor_z], dim=1)

    return vor


class MyL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        return self.loss(predicts, targets)


class MyL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        return self.loss(predicts, targets)


class WeightedL1Loss(nn.Module):
    def __init__(self, weight_outside_building: float):
        super().__init__()
        self.weight = weight_outside_building

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        abs_diff = torch.abs(predicts - targets)

        _masks = torch.broadcast_to(masks, abs_diff.shape)
        one_region_diff = torch.sum(_masks * abs_diff) / (torch.sum(_masks) + 1)

        _masks = 1 - _masks
        zero_region_diff = torch.sum(_masks * abs_diff) / (torch.sum(_masks) + 1)

        return (self.weight * one_region_diff + zero_region_diff) / (self.weight + 1)


class WeightedL2Loss(nn.Module):
    def __init__(self, weight_outside_building: float):
        super().__init__()
        self.weight = weight_outside_building

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        sq_diff = (predicts - targets) ** 2

        _masks = torch.broadcast_to(masks, sq_diff.shape)
        _masks.requires_grad = False

        one_region_diff = torch.sum(_masks * sq_diff) / (torch.sum(_masks) + 1)

        __masks = 1 - _masks
        __masks.requires_grad = False

        zero_region_diff = torch.sum(__masks * sq_diff) / (torch.sum(__masks) + 1)

        return (self.weight * one_region_diff + zero_region_diff) / (self.weight + 1)


class MixedGradientL2Loss(nn.Module):
    def __init__(self, weight_gradient_loss: float):
        super().__init__()
        self.weight_gradient_loss = weight_gradient_loss
        logger.info(f"weight grad loss = {self.weight_gradient_loss}")

    def calc_loss_terms(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        diff = predicts - targets
        sq_diff = (diff) ** 2
        mse = torch.mean(sq_diff)

        if self.weight_gradient_loss is None or self.weight_gradient_loss == 0:
            return mse, None

        is_near_walls = calc_mask_near_build_wall(masks)
        grd_mask = masks[:, :, 1:-1, 1:-1, 1:-1] * (
            1 - is_near_walls[:, :, 1:-1, 1:-1, 1:-1]
        )
        grd_mask.requires_grad = False

        grd_x = differentiate_along_x(diff, padding=0)
        grd_y = differentiate_along_y(diff, padding=0)
        grd_z = differentiate_along_z(diff, padding=0)

        grd_sum = grd_x ** 2 + grd_y ** 2 + grd_z ** 2

        # multiplication of `4` is necessary becasue predicts and targets channels are 4, but mask channel is 1.
        grd_mse = torch.sum(grd_sum * grd_mask) / (4 * torch.sum(grd_mask) + 1)

        return mse, grd_mse

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        mse, grd_mse = self.calc_loss_terms(
            predicts=predicts, targets=targets, masks=masks
        )

        if self.weight_gradient_loss is None or self.weight_gradient_loss == 0:
            return mse
        else:
            return mse + self.weight_gradient_loss * grd_mse


class MixedGradientWeightedL2Loss(nn.Module):
    def __init__(self, weight_outside_building: float, weight_gradient_loss: float):
        super().__init__()
        self.weight_outside_building = weight_outside_building
        self.weight_gradient_loss = weight_gradient_loss

    def calc_loss_terms(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        is_near_walls = calc_mask_near_build_wall(masks)
        is_near_walls.requires_grad = False

        diff = predicts - targets
        sq_diff = (diff) ** 2

        l2_masks = torch.broadcast_to(masks, sq_diff.shape)
        l2_masks.requires_grad = False

        one_region_diff = torch.sum(l2_masks * sq_diff) / (torch.sum(l2_masks) + 1)

        l2_rev_masks = 1 - l2_masks
        l2_rev_masks.requires_grad = False

        zero_region_diff = torch.sum(l2_rev_masks * sq_diff) / (
            torch.sum(l2_rev_masks) + 1
        )

        grd_x = differentiate_along_x(diff, padding=0)
        grd_y = differentiate_along_y(diff, padding=0)
        grd_z = differentiate_along_z(diff, padding=0)

        grd_mask = l2_masks[:, :, 1:-1, 1:-1, 1:-1] * (
            1 - is_near_walls[:, :, 1:-1, 1:-1, 1:-1]
        )

        grd_sum = grd_x ** 2 + grd_y ** 2 + grd_z ** 2
        grd_mse = torch.sum(grd_sum * grd_mask) / (torch.sum(grd_mask) + 1)

        return one_region_diff, zero_region_diff, grd_mse

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        one_region_diff, zero_region_diff, grd_mse = self.calc_loss_terms(
            predicts=predicts, targets=targets, masks=masks
        )

        mse = (self.weight_outside_building * one_region_diff + zero_region_diff) / (
            self.weight_outside_building + 1
        )

        return mse + self.weight_gradient_loss * grd_mse


class MixedDivergenceGradientL2Loss(nn.Module):
    def __init__(
        self,
        weight_gradient_loss: float,
        weight_divergence_loss: float,
        scales: List[float],
        delta_meter: float = 5.0,
    ):
        super().__init__()
        assert (
            len(scales) == 3
        ), "velocity components have 3. So scales length must be 3."

        self.weight_gradient_loss = weight_gradient_loss
        self.weight_divergence_loss = weight_divergence_loss
        self.scales = deepcopy(scales)
        self.mean_scale = np.mean(scales)
        self.delta_meter = delta_meter

        logger.info(f"weight grad loss = {self.weight_gradient_loss}")
        logger.info(f"weight divergence loss = {self.weight_divergence_loss}")
        logger.info(f"velocity scales = {self.scales}, its mean = {self.mean_scale}")
        logger.info(f"delta meter = {self.delta_meter}")

        if self.weight_gradient_loss == 0.0:
            logger.info("No gradient loss term")
        if self.weight_divergence_loss == 0.0:
            logger.info("No divergence loss term")

    def calc_loss_terms(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        diff = predicts - targets
        sq_diff = (diff) ** 2
        mse = torch.mean(sq_diff)

        is_near_walls = calc_mask_near_build_wall(masks)
        grd_mask = masks[:, :, 1:-1, 1:-1, 1:-1] * (
            1 - is_near_walls[:, :, 1:-1, 1:-1, 1:-1]
        )
        grd_mask.requires_grad = False

        grd_mse = 0.0
        if self.weight_gradient_loss != 0.0:
            grd_x = differentiate_along_x(diff, padding=0)
            grd_y = differentiate_along_y(diff, padding=0)
            grd_z = differentiate_along_z(diff, padding=0)

            grd_sum = grd_x ** 2 + grd_y ** 2 + grd_z ** 2

            # multiplication of `4` is necessary becasue predicts and targets channels are 4, but mask channel is 1.
            grd_mse = torch.sum(grd_sum * grd_mask) / (4 * torch.sum(grd_mask) + 1)

        div_mse = 0.0
        if self.weight_divergence_loss == 0.0:
            return mse, grd_mse, div_mse

        _scales = torch.tensor(self.scales, device=predicts.device)
        _scales = _scales[None, :, None, None, None]  # batch, channel, z, y, x dims
        _scales.requires_grad = False

        # The first channel is temperature, so targets[:, 1:] and preds[:, 1:] are used.
        scaled_trgt_v = _scales * targets[:, 1:]
        scaled_pred_v = _scales * predicts[:, 1:]

        trgt_div = _calc_residual_continuity_eq(
            scaled_trgt_v, self.delta_meter, padding=0
        )
        pred_div = _calc_residual_continuity_eq(
            scaled_pred_v, self.delta_meter, padding=0
        )

        diff_div = (
            (trgt_div - pred_div) * self.delta_meter / self.mean_scale
        )  # non-dimensionalized

        # multiplication of `4` is NOT necessary becasue diff_div channel is 1 and mask channel is 1.
        div_mse = torch.sum((diff_div ** 2) * grd_mask) / (torch.sum(grd_mask) + 1)

        return mse, grd_mse, div_mse

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        mse, grd_mse, div_mse = self.calc_loss_terms(
            predicts=predicts, targets=targets, masks=masks
        )

        return (
            mse
            + self.weight_gradient_loss * grd_mse
            + self.weight_divergence_loss * div_mse
        )


class MixedDivergenceGradientL2LossMse(MixedDivergenceGradientL2Loss):
    def __init__(
        self,
        scales: List[float],
        delta_meter: float = 5.0,
    ):
        super().__init__(
            weight_gradient_loss=0.0,
            weight_divergence_loss=0.0,
            scales=scales,
            delta_meter=delta_meter,
        )

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        mse, grd_mse, div_mse = super().calc_loss_terms(
            predicts=predicts, targets=targets, masks=masks
        )

        return mse


class MixedDivergenceGradientL2LossGrdMse(MixedDivergenceGradientL2Loss):
    def __init__(
        self,
        scales: List[float],
        delta_meter: float = 5.0,
    ):
        super().__init__(
            weight_gradient_loss=1.0,
            weight_divergence_loss=0.0,
            scales=scales,
            delta_meter=delta_meter,
        )

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        mse, grd_mse, div_mse = super().calc_loss_terms(
            predicts=predicts, targets=targets, masks=masks
        )

        return grd_mse


class MixedDivergenceGradientL2LossDivMse(MixedDivergenceGradientL2Loss):
    def __init__(
        self,
        scales: List[float],
        delta_meter: float = 5.0,
    ):
        super().__init__(
            weight_gradient_loss=0.0,
            weight_divergence_loss=1.0,
            scales=scales,
            delta_meter=delta_meter,
        )

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        mse, grd_mse, div_mse = super().calc_loss_terms(
            predicts=predicts, targets=targets, masks=masks
        )

        return div_mse


class MaskedL1Loss(nn.Module):
    def __init__(self, eps: float = 1e-30):
        super().__init__()
        self.eps = eps

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        abs_diff = torch.abs(predicts - targets)

        _masks = torch.broadcast_to(masks, abs_diff.shape)

        return torch.sum(_masks * abs_diff) / (torch.sum(_masks) + self.eps)


class MaskedL2Loss(nn.Module):
    def __init__(self, eps: float = 1e-30):
        super().__init__()
        self.eps = eps

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        diff = (predicts - targets) ** 2

        _masks = torch.broadcast_to(masks, diff.shape)

        return torch.sum(_masks * diff) / (torch.sum(_masks) + self.eps)


class MaskedL1LossNearWall(nn.Module):
    def __init__(self, eps: float = 1e-30, num_filter_applications: int = 1):
        super().__init__()
        self.eps = eps
        self.num_filter_applications = num_filter_applications

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, org_masks: torch.Tensor
    ):
        abs_diff = torch.abs(predicts - targets)

        masks = calc_mask_near_build_wall(org_masks, self.num_filter_applications)
        assert masks.shape == org_masks.shape
        _masks = torch.broadcast_to(masks, abs_diff.shape)

        return torch.sum(_masks * abs_diff) / (torch.sum(_masks) + self.eps)


class MaskedL2LossNearWall(nn.Module):
    def __init__(self, eps: float = 1e-30, num_filter_applications: int = 1):
        super().__init__()
        self.eps = eps
        self.num_filter_applications = num_filter_applications

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, org_masks: torch.Tensor
    ):
        diff = (predicts - targets) ** 2

        masks = calc_mask_near_build_wall(org_masks, self.num_filter_applications)
        assert masks.shape == org_masks.shape
        _masks = torch.broadcast_to(masks, diff.shape)

        return torch.sum(_masks * diff) / (torch.sum(_masks) + self.eps)


class ResidualContinuity(nn.Module):
    def __init__(self, scales: List[float], delta_meter: float = 5.0):
        super().__init__()
        assert len(scales) == 3
        self.scales = deepcopy(scales)
        self.delta_meter = delta_meter

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):

        residuals, num_grids = calc_residual_continuity_eq(
            masks, predicts, self.scales, self.delta_meter
        )

        return torch.sum(torch.abs(residuals)) / num_grids

    def calc_both_pred_and_target(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):

        res, n = calc_residual_continuity_eq(
            masks, predicts, self.scales, self.delta_meter
        )
        pred_res = torch.sum(torch.abs(res)) / n

        res, n = calc_residual_continuity_eq(
            masks, targets, self.scales, self.delta_meter
        )
        trgt_res = torch.sum(torch.abs(res)) / n

        return pred_res, trgt_res


class DiffVelocityVectorNorm(nn.Module):
    def __init__(self, scales: List[float], eps: float = 1e-30, lev: int = None):
        super().__init__()
        assert len(scales) == 3
        self.scales = deepcopy(scales)
        self.scales = torch.tensor(self.scales)[
            None, :, None, None, None
        ]  # add batch, z, y, and x dim
        self.eps = eps
        self.lev = lev

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):

        assert predicts.shape[1] == targets.shape[1] == 4  # channels == T, u, v, w

        scales = self.scales.to(predicts.device)
        v_pred = predicts[:, 1:] * scales
        v_trgt = targets[:, 1:] * scales

        diff = torch.linalg.norm(v_pred - v_trgt, dim=1, keepdim=True)

        _masks = torch.broadcast_to(masks, diff.shape)

        if self.lev is not None:
            assert len(diff.shape) == 5  # batch, chennel, z, y, x
            diff = diff[:, :, self.lev]
            _masks = _masks[:, :, self.lev]

        return torch.sum(_masks * diff) / (torch.sum(_masks) + self.eps)


class AbsDiffTemperature(nn.Module):
    def __init__(self, scale: float, eps: float = 1e-30, lev: int = None):
        super().__init__()
        self.scale = scale
        self.eps = eps
        self.lev = lev

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):

        assert predicts.shape[1] == targets.shape[1] == 4  # channels == T, u, v, w

        pred = predicts[:, 0:1]
        trgt = targets[:, 0:1]
        assert pred.shape[1] == trgt.shape[1] == 1  # only T (temperature) channel

        diff = torch.abs(pred - trgt) * self.scale

        _masks = torch.broadcast_to(masks, diff.shape)

        if self.lev is not None:
            assert len(diff.shape) == 5  # batch, chennel, z, y, x
            diff = diff[:, :, self.lev]
            _masks = _masks[:, :, self.lev]

        return torch.sum(_masks * diff) / (torch.sum(_masks) + self.eps)


class AbsDiffDivergence(nn.Module):
    def __init__(self, scales: List[float], delta_meter: float = 5.0):
        super().__init__()
        assert len(scales) == 3
        self.scales = deepcopy(scales)
        self.delta_meter = delta_meter

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):

        pred, n1 = calc_residual_continuity_eq(
            masks, predicts, self.scales, self.delta_meter
        )

        trgt, n2 = calc_residual_continuity_eq(
            masks, targets, self.scales, self.delta_meter
        )

        return torch.sum(torch.abs(pred - trgt)) / n1


class DiffOmegaVectorNorm(nn.Module):
    def __init__(self, scales: List[float], delta_meter: float = 5.0):
        super().__init__()
        assert len(scales) == 3
        self.scales = deepcopy(scales)
        self.delta = delta_meter

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        pred, n1 = calc_vorticity_vector(masks, predicts, self.scales, self.delta)
        trgt, n2 = calc_vorticity_vector(masks, targets, self.scales, self.delta)

        diff = torch.linalg.norm(pred - trgt, dim=1, keepdim=True)

        return torch.sum(diff) / n1


class Ssim3dLoss(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        size_average: bool = True,
        max_val: float = 1.0,
        eps: float = 1e-3,
        use_gaussian=True,
    ):
        super().__init__()
        self.ssim = SSIM3D(
            window_size=window_size,
            sigma=sigma,
            size_average=size_average,
            max_val=max_val,
            eps=eps,
            use_gaussian=use_gaussian,
        )

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        assert predicts.shape == targets.shape
        _masks = torch.broadcast_to(masks, predicts.shape)

        return self.ssim(predicts, targets, _masks)


class ChannelwiseMse(nn.Module):
    def __init__(self, i_channel):
        super().__init__()
        self.i_channel = i_channel

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        assert predicts.shape[1] == targets.shape[1] == 4

        diffs = (predicts[:, self.i_channel] - targets[:, self.i_channel]) ** 2

        return torch.mean(diffs)