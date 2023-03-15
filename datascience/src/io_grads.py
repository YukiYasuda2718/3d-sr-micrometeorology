import pathlib
from datetime import datetime
from logging import getLogger

import dask
import numpy as np
import pandas as pd
import xarray as xr
from xgrads import open_CtlDataset

logger = getLogger()


def get_grads_dir_micro_meteorology(
    root_dir: pathlib.Path, dt: datetime
) -> pathlib.Path:
    str_dt = f"{dt:%Y-%m-%dT%H:%M:%S}"
    return root_dir / str_dt / "micro-meteorology" / "out" / "grads"


def get_grads_dir_meteorology(
    root_dir: pathlib.Path, dt: datetime, offset_hours: int = 5
) -> pathlib.Path:
    _dt = pd.Timestamp(dt) - pd.Timedelta(hours=offset_hours)
    str_dt = f"{dt:%Y-%m-%dT%H:%M:%S}"
    case_dir = f"1km_300m_100m-tokyo-many-grids_{_dt:%Y%m%d}_{_dt:%H}_{_dt:%M}UTC.000"
    return root_dir / str_dt / "mssg-auto" / "MSSG" / case_dir / "out" / "grads"


def read_xarray(
    dir_path: pathlib.Path,
    variable_name: str,
    margin: int = 20,
    nx: int = 400,
    ny: int = 400,
    discarded_initial_period: int = 10,
    min_index_height: int = 0,
    max_index_height: int = None,
    nest_level: str = "0n",
) -> xr.DataArray:

    if min_index_height != 0 and max_index_height is None:
        logger.warning("min_index_height is not used.")

    # Read xr.Dataset and extract xr.DataArray
    ds = open_CtlDataset(str(dir_path / f"atmos_{nest_level}_{variable_name}.ctl"))
    da = ds[variable_name]

    # Fill missing values with nan
    assert isinstance(ds.undef, float)
    logger.debug(f"{variable_name}: undef = {ds.undef}")
    da = xr.where(da == ds.undef, np.nan, da)

    # Drop margin regions
    if margin > 0:
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            da = da.isel(
                lon=np.arange(margin, nx - margin), lat=np.arange(margin, ny - margin)
            )

    # Discard the initial period
    times = da.time[discarded_initial_period:]
    da = da.sel(time=times)

    if max_index_height is not None:
        levs = da.lev[min_index_height:max_index_height]
        logger.debug(
            f"{variable_name}, height range = {levs.values[0]} - {levs.values[-1]} m"
        )
        da = da.sel(lev=levs)

    return da


def align_nan_grids(target_da: xr.DataArray, source_da: xr.DataArray) -> xr.DataArray:
    ret = xr.where(np.isnan(source_da), np.nan, target_da)
    ret.name = target_da.name
    return ret


def calc_std_fields(
    ave_1st_moment: xr.DataArray, ave_2nd_moment: xr.DataArray
) -> xr.DataArray:
    assert (
        ave_1st_moment.name == ave_2nd_moment.name[:2]
    ), "Variable name header is different"

    _vars = ave_2nd_moment - ave_1st_moment**2

    stds = np.sqrt(_vars)
    stds = xr.where(stds > 0.0, stds, 0)
    stds = xr.where(np.isnan(ave_1st_moment), np.nan, stds)

    stds.name = f"{ave_1st_moment.name}_std"

    return stds
