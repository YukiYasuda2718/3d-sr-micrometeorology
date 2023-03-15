ROOT_DIR = "/S/home00/G4012/y0630/workspace_lab/3d-scalar-sr"

import argparse
import os
import pathlib
import sys
import traceback
from datetime import datetime
from logging import INFO, FileHandler, StreamHandler, getLogger

import dask
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

if f"{ROOT_DIR}/datascience/src" not in sys.path:
    sys.path.append(f"{ROOT_DIR}/datascience/src")

from building_height_helper import (
    calc_ave_pooling_weights,
    calc_is_in_building,
    make_resized_lr_tz,
    read_building_height,
)
from io_grads import get_grads_dir_micro_meteorology
from training_data_maker import (
    load_atmos_data,
    make_coarse_grained_dataset_with_outside_lr_buildings,
)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

dask.config.set(**{"array.slicing.split_large_chunks": True})

ROOT_RESUT_DIR = pathlib.Path("/data/mssg-results/tokyo_05m")
DL_DATA_ROOT_DIR = pathlib.Path(f"{ROOT_DIR}/data/DL_data")
HR_BUILDING_TXT_PATH = f"{ROOT_DIR}/datascience/script/EleTopoZ_HR.txt"
LR_BUILDING_TXT_PATH = f"{ROOT_DIR}/datascience/script/EleTopoZ_LR.txt"
LOG_DIR = f"{ROOT_DIR}/datascience/script/make_dl_data_using_outside_lr_builds/log"
os.makedirs(LOG_DIR, exist_ok=True)


SR_SCALE = 4
assert SR_SCALE == 4

TIME_SAMPLE_INTERVAL = 2
HR_MARGIN = 40
LR_MARGIN = HR_MARGIN // SR_SCALE
VAR_NAMES = ["tm", "vl", "vp", "vr"]
DATA_GROUP = "10"


def save(file_path: str, data: np.ndarray, margin: int):
    assert margin > 0
    out = data[..., margin:-margin, margin:-margin]  # cut margins along y and x dims
    if not os.path.exists(file_path):
        np.save(file_path, out)


def make_hr_build_data(
    hr_tz: np.ndarray, hr_ez: np.ndarray, hr_ds: xr.DataArray, dl_data_dir: pathlib.Path
) -> np.ndarray:

    hr_is_in_build = calc_is_in_building(
        tz=hr_tz, ez=hr_ez, actual_levs=hr_ds.lev.values
    )
    _hr_is_in_build = np.isnan(hr_ds["tm"].isel(time=0).values).astype(float)
    hr_is_in_build = (hr_is_in_build + _hr_is_in_build > 0).astype(float)

    # Replicate over the channel dim
    hr_is_in_build = np.expand_dims(hr_is_in_build, axis=0)
    hr_is_in_build = np.tile(hr_is_in_build, (len(VAR_NAMES), 1, 1, 1))
    save(
        str(dl_data_dir / "hr_is_in_build.npy"),
        hr_is_in_build,
        HR_MARGIN,
    )

    return hr_is_in_build


def make_lr_build_data(
    lr_tz: np.ndarray, lr_ez: np.ndarray, lr_ds: xr.DataArray, dl_data_dir: pathlib.Path
) -> np.ndarray:

    lr_is_in_build = calc_is_in_building(
        tz=lr_tz, ez=lr_ez, actual_levs=lr_ds.lev.values
    )
    lr_is_in_build = np.expand_dims(lr_is_in_build, axis=0)
    lr_is_in_build = np.tile(lr_is_in_build, (len(VAR_NAMES), 1, 1, 1))
    save(
        str(dl_data_dir / "lr_is_in_build.npy"),
        lr_is_in_build,
        LR_MARGIN,
    )

    return lr_is_in_build


parser = argparse.ArgumentParser()

parser.add_argument(
    "--target_datetime",
    type=str,
    help="target UTC datetime in ISO 8601 format, e.g., 2015-07-31T05:00:00",
)

if __name__ == "__main__":
    try:
        target_datetime = datetime.strptime(
            parser.parse_args().target_datetime, "%Y-%m-%dT%H:%M:%S"
        )

        logger.addHandler(
            FileHandler(f"{LOG_DIR}/{parser.parse_args().target_datetime}.txt")
        )
        logger.info("\n*********************************************************")
        logger.info(f"Start: {datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")
        logger.info(f"\ntarget_datetimes = {target_datetime}")

        hr_tz = read_building_height(HR_BUILDING_TXT_PATH, "Tz", margin=0).transpose()
        hr_ez = read_building_height(HR_BUILDING_TXT_PATH, "Ez", margin=0).transpose()
        assert (400, 400) == hr_tz.shape == hr_ez.shape

        lr_tz = read_building_height(LR_BUILDING_TXT_PATH, "Tz", margin=0).transpose()
        lr_ez = read_building_height(LR_BUILDING_TXT_PATH, "Ez", margin=0).transpose()
        assert (100, 100) == lr_tz.shape == lr_ez.shape

        dl_data_dir = DL_DATA_ROOT_DIR / DATA_GROUP

        hr_is_in_build = None
        lr_is_in_build = None
        resized_lr_is_in_build = None
        avg_pooling_weights = None

        grads_dir = get_grads_dir_micro_meteorology(ROOT_RESUT_DIR, target_datetime)

        logger.info("Start to calculate HR data")

        hr_ds = load_atmos_data(grads_dir, data_shape=(50, 40, 400, 400))

        # Sample times from the "last" index
        if TIME_SAMPLE_INTERVAL > 1:
            times = hr_ds.time.values[::-1][::TIME_SAMPLE_INTERVAL][::-1]
            hr_ds = hr_ds.sel(time=times)
            assert len(hr_ds.time) == 50 // TIME_SAMPLE_INTERVAL

        for time in tqdm(hr_ds.time.values):
            ts = pd.Timestamp(time)
            data_dir = dl_data_dir / f"{ts:%Y%m%d}"
            os.makedirs(data_dir, exist_ok=True)

            if hr_is_in_build is None:
                hr_is_in_build = make_hr_build_data(hr_tz, hr_ez, hr_ds, dl_data_dir)

                resized_lr_is_in_build = make_resized_lr_tz(
                    lr_tz=lr_tz,
                    lr_ez=lr_ez,
                    hr_is_in_build=hr_is_in_build,
                    actual_hr_levs=hr_ds["lev"].values,
                )

                # `0` specifies channel num. all `lr_is_in_build` are same over channel
                avg_pooling_weights = calc_ave_pooling_weights(
                    lr_is_in_build=resized_lr_is_in_build[0],
                    lev_window_width=SR_SCALE,
                    lat_window_width=SR_SCALE,
                    lon_window_width=SR_SCALE,
                )

            hr_file = data_dir / f"{ts:%Y%m%dT%H%M%S}_HR.npy"
            if not hr_file.exists():
                _ds = hr_ds.sel(time=time)
                hr_out = np.stack([_ds[name].values for name in VAR_NAMES])

                assert hr_out.shape == hr_is_in_build.shape
                hr_out = np.where(hr_is_in_build == 1.0, np.nan, hr_out)

                save(str(hr_file), hr_out, HR_MARGIN)
                logger.info(f"{hr_file} has been made")

            lr_file = data_dir / f"{ts:%Y%m%dT%H%M%S}_LR_x04.npy"
            if not lr_file.exists():
                # `0` selects the first channel, is_in_build does not deped on channel number.
                lr_ds = make_coarse_grained_dataset_with_outside_lr_buildings(
                    hr_ds=hr_ds.sel(time=time),
                    lr_is_in_build=resized_lr_is_in_build[0],
                    hr_is_in_build=hr_is_in_build[0],
                    avg_pooling_weights=avg_pooling_weights,
                    lev_window_width=SR_SCALE,
                    lat_window_width=SR_SCALE,
                    lon_window_width=SR_SCALE,
                )

                if lr_is_in_build is None:
                    lr_is_in_build = make_lr_build_data(
                        lr_tz, lr_ez, lr_ds, dl_data_dir
                    )

                lr_out = np.stack([lr_ds[name].values for name in VAR_NAMES])

                assert lr_out.shape == lr_is_in_build.shape
                lr_out = np.where(lr_is_in_build == 1.0, np.nan, lr_out)

                save(str(lr_file), lr_out, LR_MARGIN)
                logger.info(f"{lr_file} has been made.")

        logger.info("\n*********************************************************")
        logger.info(f"End: {datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())
