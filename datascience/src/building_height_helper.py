import cv2
import numpy as np
import pandas as pd


def read_building_height(
    building_path: str, target_col: str, margin: int = 20
) -> np.ndarray:

    with open(building_path, "r") as file:
        lines = file.readlines()

    cols = ["i", "j", "Ez", "Tz", "Tzl"]
    _dict = {}
    for i, line in enumerate(lines[1:]):  # skip header
        splits = list(
            map(lambda s: s.strip(), filter(lambda s: s != "", line.split(" ")))
        )
        _dict[i] = {k: v for k, v in zip(cols, splits)}

    df_topography = pd.DataFrame.from_dict(_dict).T

    for col in cols:
        if col == "i" or col == "j":
            df_topography[col] = df_topography[col].astype(int)
        else:
            df_topography[col] = df_topography[col].astype(float)

    ret = pd.pivot_table(
        data=df_topography[["i", "j", target_col]],
        values=target_col,
        index="i",
        columns="j",
        aggfunc="max",
    ).values

    if margin == 0:
        return ret
    else:
        return ret[margin:-margin, margin:-margin]


def calc_is_in_building(
    tz: np.ndarray, ez: np.ndarray, actual_levs: np.ndarray
) -> np.ndarray:

    # tz = build height, ez = ground height, both from sea surface

    assert tz.shape == ez.shape
    assert len(tz.shape) == 2  # y and x
    assert len(actual_levs.shape) == 1  # z

    _shape = actual_levs.shape + tz.shape  # dims = (z, y, x)

    is_in_building = np.zeros(_shape)
    for j in range(is_in_building.shape[1]):  # y dim
        for i in range(is_in_building.shape[2]):  # x dim
            t, e = tz[j, i], ez[j, i]
            if t <= e:  # BH is lower than or equal to the ground.
                continue  # This means there is no building.

            idx_top_of_build = (actual_levs < t).argmin()
            is_in_building[:idx_top_of_build, j, i] = 1

    return is_in_building


def make_resized_lr_tz(
    lr_tz: np.ndarray,
    lr_ez: np.ndarray,
    hr_is_in_build: np.ndarray,
    actual_hr_levs: list,
):
    assert lr_tz.ndim == lr_ez.ndim == 2  # y and x
    assert hr_is_in_build.ndim == 4  # channel, z, y, x

    size = hr_is_in_build.shape[-2:]

    # [..., None] adds channel dim
    resized_lr_tz = cv2.resize(
        lr_tz[..., None], dsize=size, interpolation=cv2.INTER_NEAREST
    )

    # [..., None] adds channel dim
    resized_lr_ez = cv2.resize(
        lr_ez[..., None], dsize=size, interpolation=cv2.INTER_NEAREST
    )

    resized_lr_is_in_build = calc_is_in_building(
        tz=resized_lr_tz, ez=resized_lr_ez, actual_levs=actual_hr_levs
    )

    # [None, ...] adds channel dim
    return np.broadcast_to(
        resized_lr_is_in_build[None, ...], shape=hr_is_in_build.shape
    )


def calc_ave_pooling_weights(
    lr_is_in_build: np.ndarray,
    lev_window_width: int,
    lat_window_width: int,
    lon_window_width: int,
):

    assert lr_is_in_build.ndim == 3  # z, y, x

    weights = np.where(
        lr_is_in_build == 1,
        np.zeros_like(lr_is_in_build),
        np.ones_like(lr_is_in_build),
    )
    weights = np.lib.stride_tricks.sliding_window_view(
        weights,
        window_shape=(lev_window_width, lat_window_width, lon_window_width),
        axis=(0, 1, 2),
    )

    sum_weights = np.sum(weights, axis=(-3, -2, -1), keepdims=True)
    sum_weights = np.broadcast_to(sum_weights, shape=weights.shape)

    # ignore invalid value encounter due to zero-division
    with np.errstate(invalid="ignore"):
        weights = np.where(sum_weights != 0, weights / sum_weights, np.nan)

    return weights
