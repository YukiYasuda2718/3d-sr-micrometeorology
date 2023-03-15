import copy
import pathlib
import typing

import numpy as np
import xarray as xr
from io_grads import align_nan_grids, read_xarray


def _load_atmos_data(
    grads_dir: pathlib.Path,
    min_index_height: int = 4,
    max_index_height: int = 101,
    margin: int = 20,
) -> typing.Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:

    vl = read_xarray(
        grads_dir,
        "vl",
        margin=margin,
        min_index_height=min_index_height,
        max_index_height=max_index_height,
    )
    vp = read_xarray(
        grads_dir,
        "vp",
        margin=margin,
        min_index_height=min_index_height,
        max_index_height=max_index_height,
    )
    vr = read_xarray(
        grads_dir,
        "vr",
        margin=margin,
        min_index_height=min_index_height,
        max_index_height=max_index_height,
    )
    tm = read_xarray(
        grads_dir,
        "tm",
        margin=margin,
        min_index_height=min_index_height,
        max_index_height=max_index_height,
    )

    vl = align_nan_grids(vl, tm)
    vp = align_nan_grids(vp, tm)
    vr = align_nan_grids(vr, tm)

    return vl, vp, vr, tm


def load_atmos_data(
    grads_dir: pathlib.Path, data_shape: tuple = (50, 96, 400, 400)
) -> xr.Dataset:

    assert data_shape[-2] == data_shape[-1], "x and y grid numbers are not equal"
    max_index_height = 4 + data_shape[1]
    # `4` indicates lev = 17.5 meters, which is the ground level around Tokyo station.

    vl, vp, vr, tm = _load_atmos_data(
        grads_dir, max_index_height=max_index_height, margin=0
    )
    assert vl.shape == data_shape  # t,z,y,x
    assert vl.shape == vp.shape == vr.shape == tm.shape
    assert vl.dims == ("time", "lev", "lat", "lon")  # t,z,y,x
    assert vl.dims == vp.dims == vr.dims == tm.dims

    vertical_grid_spaces = np.unique(np.diff(tm.lev.values))
    assert len(vertical_grid_spaces) == 1, "Vertical grid spacing is not unique"
    assert (
        vertical_grid_spaces[0] == 5.0
    ), "Vertical grid spacing is not equal to 5 meters"

    return xr.Dataset({"tm": tm, "vl": vl, "vp": vp, "vr": vr})


def make_coarse_grained(
    da: xr.DataArray,
    lev_window_width: int = 4,
    lat_window_width: int = 4,
    lon_window_width: int = 4,
) -> xr.DataArray:

    data = da

    data = data.interpolate_na(dim="lat", method="nearest", fill_value="extrapolate")
    data = data.interpolate_na(dim="lon", method="nearest", fill_value="extrapolate")

    data = (
        data
        #
        .rolling(lev=lev_window_width, min_periods=None, center=True)
        .mean()
        .rolling(lat=lat_window_width, min_periods=None, center=True)
        .mean()
        .rolling(lon=lon_window_width, min_periods=None, center=True)
        .mean()
    )

    data = data.sel(
        lev=data.lev.values[lev_window_width // 2 :: lev_window_width],
        lat=data.lat.values[lat_window_width // 2 :: lat_window_width],
        lon=data.lon.values[lon_window_width // 2 :: lon_window_width],
    )

    data.coords["lev"] = copy.deepcopy(da.lev[::lev_window_width])
    data.coords["lat"] = copy.deepcopy(da.lat[::lat_window_width])
    data.coords["lon"] = copy.deepcopy(da.lon[::lon_window_width])

    return data


def make_coarse_grained_dataset(
    hr_ds: xr.Dataset,
    lev_window_width: int = 4,
    lat_window_width: int = 4,
    lon_window_width: int = 4,
) -> xr.Dataset:

    var_names = list(hr_ds.data_vars)

    lr_ds = {}

    for var_name in var_names:
        lr_ds[var_name] = make_coarse_grained(
            hr_ds[var_name],
            lev_window_width=lev_window_width,
            lat_window_width=lat_window_width,
            lon_window_width=lon_window_width,
        )

    return xr.Dataset(lr_ds)


def make_coarse_grained_dataarray_with_outside_lr_buildings(
    *,
    da: xr.DataArray,
    lr_is_in_build: np.ndarray,
    hr_is_in_build: np.ndarray,
    avg_pooling_weights: np.ndarray,
    lev_window_width: int,
    lat_window_width: int,
    lon_window_width: int,
):
    assert lr_is_in_build.ndim == 3  # z, y, x
    assert da.shape == lr_is_in_build.shape == hr_is_in_build.shape

    hr_data = xr.where(hr_is_in_build == 1, np.nan, da)

    hr_data = hr_data.chunk(dict(lev=-1)).interpolate_na(
        dim="lev", method="nearest", fill_value="extrapolate"
    )
    if np.sum(np.isnan(hr_data.values)) > 0:
        hr_data = hr_data.interpolate_na(
            dim="lat", method="nearest", fill_value="extrapolate"
        )
        hr_data = hr_data.interpolate_na(
            dim="lon", method="nearest", fill_value="extrapolate"
        )
    hr_data = xr.where(lr_is_in_build == 1, np.nan, hr_data)

    _data = np.lib.stride_tricks.sliding_window_view(
        hr_data.values,
        window_shape=(lev_window_width, lat_window_width, lon_window_width),
        axis=(0, 1, 2),
    )
    assert _data.shape == avg_pooling_weights.shape

    _data = np.sum((_data * avg_pooling_weights), axis=(-3, -2, -1))

    lr_data = np.zeros_like(hr_data.values)
    lr_data[:, :, :] = np.nan
    lr_data[
        lev_window_width // 2 : -lev_window_width // 2 + 1,
        lat_window_width // 2 : -lat_window_width // 2 + 1,
        lon_window_width // 2 : -lon_window_width // 2 + 1,
    ] = _data

    lr_da = xr.DataArray(lr_data, coords=copy.deepcopy(da.coords))

    lr_da = lr_da.sel(
        lev=lr_da.lev.values[lev_window_width // 2 :: lev_window_width],
        lat=lr_da.lat.values[lat_window_width // 2 :: lat_window_width],
        lon=lr_da.lon.values[lon_window_width // 2 :: lon_window_width],
    )
    lr_da.coords["lev"] = copy.deepcopy(da.lev[::lev_window_width])
    lr_da.coords["lat"] = copy.deepcopy(da.lat[::lat_window_width])
    lr_da.coords["lon"] = copy.deepcopy(da.lon[::lon_window_width])

    return lr_da


def make_coarse_grained_dataset_with_outside_lr_buildings(
    *,
    hr_ds: xr.Dataset,
    lr_is_in_build: np.ndarray,
    hr_is_in_build: np.ndarray,
    avg_pooling_weights: np.ndarray,
    lev_window_width: int = 4,
    lat_window_width: int = 4,
    lon_window_width: int = 4,
) -> xr.Dataset:

    assert list(hr_ds.dims.keys()) == ["lev", "lat", "lon"]

    var_names = list(hr_ds.data_vars)

    lr_ds = {}

    for var_name in var_names:
        lr_ds[var_name] = make_coarse_grained_dataarray_with_outside_lr_buildings(
            da=hr_ds[var_name],
            lr_is_in_build=lr_is_in_build,
            hr_is_in_build=hr_is_in_build,
            avg_pooling_weights=avg_pooling_weights,
            lev_window_width=lev_window_width,
            lat_window_width=lat_window_width,
            lon_window_width=lon_window_width,
        )

    return xr.Dataset(lr_ds)
