import numpy as np
import scipy
import torch


def _rotate_temperature_velocity(
    Xs: np.ndarray, means: list, scales: list, angle: float, rescaled: bool = True
):

    # Here, add batch, z, y, and x dims
    means = np.array(means)[None, :, None, None, None]
    scales = np.array(scales)[None, :, None, None, None]

    assert Xs.ndim == means.ndim == scales.ndim
    assert Xs.shape[1] == means.shape[1] == scales.shape[1] == 4  # channel num

    theta = np.deg2rad(angle)
    rot_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), np.sin(theta), 0],
            [0, -np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    scaled_Xs = Xs * scales + means

    rotated_Xs = scipy.ndimage.rotate(
        scaled_Xs, angle=angle, axes=(-2, -1), reshape=False
    )

    rotated_Xs = np.einsum("ij,bjklm->biklm", rot_matrix, rotated_Xs)

    assert rotated_Xs.shape == Xs.shape

    if rescaled:
        rotated_Xs = np.clip((rotated_Xs - means) / scales, a_min=0.0, a_max=1.0)

    return rotated_Xs, scaled_Xs


def rotate_temperature_velocity(
    Xs: np.ndarray, means: list, scales: list, angle: float, rescaled: bool = True
):
    rotated_Xs, _ = _rotate_temperature_velocity(Xs, means, scales, angle, rescaled)
    return rotated_Xs


def rotate_building_height(bs: np.ndarray, angle: float, th: float = 0.1):

    # batch, channel, z, y, x dims
    assert bs.ndim == 5

    rotated_bs = scipy.ndimage.rotate(bs, angle=angle, axes=(-2, -1), reshape=False)

    rotated_bs = np.where(
        rotated_bs > th, np.ones_like(rotated_bs), np.zeros_like(rotated_bs)
    )

    return rotated_bs


def calc_error_ratio(diff: np.ndarray, sx: int, ex: int, sy: int, ey: int):
    assert diff.ndim == 4
    # batch, z, y, x dims

    # Perform mean along space and batch
    return np.mean(np.abs(diff)[..., sy:ey, sx:ex])


def calc_preds_for_equivariance_errors(
    *,
    Xs: torch.Tensor,
    bs: torch.Tensor,
    model: torch.nn.Module,
    means: list,
    scales: list,
    angle: float,
    device: str,
    dtype: torch.dtype = torch.float32,
):
    with torch.no_grad():
        preds = model(Xs.to(device), bs.to(device)).cpu().numpy()

    rotated_Xs = torch.from_numpy(
        rotate_temperature_velocity(Xs.numpy(), means, scales, angle)
    ).to(dtype)

    rotated_bs = torch.from_numpy(rotate_building_height(bs, angle))

    assert rotated_Xs.shape == Xs.shape
    assert rotated_bs.shape == bs.shape

    # plt.imshow(Xs[0, 0, 0], vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(rotated_Xs[0, 0, 0], vmin=0, vmax=1)
    # plt.show()

    with torch.no_grad():
        preds_after_rot = (
            model(rotated_Xs.to(device), rotated_bs.to(device)).cpu().numpy()
        )

    rotated_preds = rotate_temperature_velocity(preds, means, scales, angle)

    means = np.array(means)[None, :, None, None, None]
    scales = np.array(scales)[None, :, None, None, None]

    assert means.ndim == scales.ndim == rotated_preds.ndim == preds_after_rot.ndim
    assert Xs.shape[1] == means.shape[1] == scales.shape[1] == 4  # channel num

    rotated_preds = scales * rotated_preds + means
    preds_after_rot = scales * preds_after_rot + means

    return rotated_preds, preds_after_rot


def calc_equivariance_errors(
    *,
    Xs: torch.Tensor,
    bs: torch.Tensor,
    model: torch.nn.Module,
    means: list,
    scales: list,
    angle: float,
    sx: int,
    ex: int,
    sy: int,
    ey: int,
    device: str,
    dtype: torch.dtype = torch.float32,
):
    rotated_preds, preds_after_rot = calc_preds_for_equivariance_errors(
        Xs=Xs,
        bs=bs,
        model=model,
        means=means,
        scales=scales,
        angle=angle,
        device=device,
    )

    dict_err = {}

    error_tm = rotated_preds[:, 0] - preds_after_rot[:, 0]
    dict_err["tm"] = calc_error_ratio(error_tm, sx, ex, sy, ey)

    error_vr = rotated_preds[:, 3] - preds_after_rot[:, 3]
    dict_err["vr"] = calc_error_ratio(error_vr, sx, ex, sy, ey)

    error_vl_vp = np.sqrt(
        (rotated_preds[:, 1] - preds_after_rot[:, 1]) ** 2
        + (rotated_preds[:, 2] - preds_after_rot[:, 2]) ** 2
    )

    dict_err["vl_vp"] = calc_error_ratio(error_vl_vp, sx, ex, sy, ey)

    return dict_err