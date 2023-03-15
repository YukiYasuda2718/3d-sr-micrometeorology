import glob
import os
import pathlib
import typing
from logging import getLogger

from sklearn.model_selection import train_test_split
from src.dataset import DatasetWithoutAligningResolution
from src.utils import get_torch_generator, seed_worker
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = getLogger()


def get_all_data_dir_paths(root_dir: pathlib.Path) -> typing.List[pathlib.Path]:
    logger.info("This is newly implemented get_all_data_dir_paths method.")

    # 14 pm (JST) - 15 pm (JST)
    all_03_dirs = [
        path
        for path in sorted(glob.glob(str(root_dir / "03" / "*")))
        if os.path.isdir(path)
    ]

    # 13 pm (JST) - 14 pm (JST)
    all_04_dirs = [
        path
        for path in sorted(glob.glob(str(root_dir / "04" / "*")))
        if os.path.isdir(path)
    ]

    # 15 pm (JST) - 16 pm (JST)
    all_05_dirs = [
        path
        for path in sorted(glob.glob(str(root_dir / "05" / "*")))
        if os.path.isdir(path)
    ]

    assert len(all_03_dirs) == len(all_04_dirs) == len(all_05_dirs)

    all_data_dirs = []

    for dir_04, dir_03, dir_05 in zip(all_04_dirs, all_03_dirs, all_05_dirs):

        # Check dates are the same.
        assert (
            os.path.basename(dir_04)
            == os.path.basename(dir_03)
            == os.path.basename(dir_05)
        )

        # dir_04 must be appended first to keep the chronological order
        all_data_dirs.append(pathlib.Path(dir_04))
        all_data_dirs.append(pathlib.Path(dir_03))
        all_data_dirs.append(pathlib.Path(dir_05))

    return all_data_dirs


def get_all_new_lr_data_dir_paths(
    root_dir: pathlib.Path, dir_name: str = "10"
) -> typing.List[pathlib.Path]:
    return [
        pathlib.Path(path)
        for path in sorted(glob.glob(str(root_dir / dir_name / "*")))
        if os.path.isdir(path)
    ]


def _get_all_data_dir_paths(
    dir_names: typing.List[str], root_dir: pathlib.Path
) -> typing.List[pathlib.Path]:

    logger.warning("This method is old. Use get_all_data_dir_paths")

    all_data_dirs = []

    for dir_name in dir_names:
        all_data_dirs += [
            pathlib.Path(path)
            for path in sorted(glob.glob(str(root_dir / dir_name / "*")))
            if os.path.isdir(path)
        ]
    return all_data_dirs


def split_into_train_valid_test_dirs(
    all_data_dirs: typing.List[pathlib.Path], train_valid_test_ratios: typing.List[int]
) -> typing.Dict[str, typing.List[pathlib.Path]]:

    test_size = train_valid_test_ratios[-1]
    _data_dirs, test_data_dirs = train_test_split(
        all_data_dirs, test_size=test_size, shuffle=False
    )

    valid_size = train_valid_test_ratios[1] / (
        train_valid_test_ratios[0] + train_valid_test_ratios[1]
    )
    train_data_dirs, valid_data_dirs = train_test_split(
        _data_dirs, test_size=valid_size, shuffle=False
    )

    return {"train": train_data_dirs, "valid": valid_data_dirs, "test": test_data_dirs}


def make_dataloaders(
    data_dirs: typing.Dict[str, typing.List[pathlib.Path]],
    hr_3d_build_path: pathlib.Path,
    means: typing.List[float] = [0.0, 0.0, 0.0, 0.0],
    stds: typing.List[float] = [1.0, 1.0, 1.0, 1.0],
    nan_value: float = 0.0,
    hr_org_size: tuple = (32, 320, 320),
    hr_crop_size: tuple = (16, 64, 64),
    rank: int = None,
    world_size: int = None,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 0,
    datasizes: typing.Dict[str, int] = {},
    use_clipping: bool = True,
    lr_scaling: float = None,
    max_discarded_lr_z_index: int = None,
    scale_factor: int = 4,
    **kwargs,
) -> typing.Tuple[dict, dict]:

    logger.info(f"Seed passed to make_dataloaders = {seed}")
    logger.info(f"Batch size = {batch_size}")

    dict_dataloaders, dict_samplers = {}, {}

    for kind in ["train", "valid", "test"]:

        dataset = DatasetWithoutAligningResolution(
            data_dirs=data_dirs[kind],
            hr_3d_build_path=hr_3d_build_path,
            means=means,
            stds=stds,
            nan_value=nan_value,
            hr_org_size=hr_org_size,
            hr_crop_size=hr_crop_size,
            datasize=datasizes.get(kind, None),
            seed=seed,
            use_clipping=use_clipping,
            lr_scaling=lr_scaling,
            max_discarded_lr_z_index=max_discarded_lr_z_index,
            scale_factor=scale_factor,
        )

        if world_size is None or rank is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}"
                )

    return dict_dataloaders, dict_samplers


def make_evaluation_dataloader_without_random_cropping(
    config: dict,
    data_dir_root: pathlib.Path,
    batch_size: int = 1,
    num_workers: int = 2,
    max_height_index: int = 32,
):

    if config["data"]["data_dir_names"] == ["03", "04", "05"]:
        logger.info("Data dirs are [03, 04, 05]")
        all_data_dirs = get_all_data_dir_paths(data_dir_root)
    elif config["data"]["data_dir_names"] == ["10"]:
        logger.info("Data dirs are [10]")
        all_data_dirs = get_all_new_lr_data_dir_paths(data_dir_root)
    elif config["data"]["data_dir_names"] == ["20"]:
        logger.info("Data dirs are [20]")
        all_data_dirs = get_all_new_lr_data_dir_paths(data_dir_root, dir_name="20")
    else:
        raise Exception(
            f'Data dirs {config["data"]["data_dir_names"]} are not supported.'
        )

    test_dirs = split_into_train_valid_test_dirs(
        all_data_dirs, config["data"]["train_valid_test_ratios"]
    )["test"]

    test_datasets = DatasetWithoutAligningResolution(
        data_dirs=test_dirs,
        hr_3d_build_path=all_data_dirs[0].parent / "hr_is_in_build.npy",
        means=config["data"]["means"],
        stds=config["data"]["stds"],
        nan_value=config["data"]["nan_value"],
        hr_org_size=tuple(config["data"]["hr_org_size"]),
        hr_crop_size=tuple(config["data"]["hr_crop_size"]),
        use_cropping=False,
        use_clipping=False,
        datasize=config["data"]["datasizes"]["test"],
        seed=config["data"]["seed"],
        lr_scaling=config["data"].get("lr_scaling", None),
        max_height_index=max_height_index,
        max_discarded_lr_z_index=config["data"].get("max_discarded_lr_z_index", None),
        scale_factor=config["data"].get("scale_factor", 4),
    )

    return DataLoader(
        test_datasets,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
        num_workers=num_workers,
    )


def make_evaluation_dataloader_with_random_cropping(
    config: dict,
    data_dir_root: pathlib.Path,
    batch_size: int = 32,
    num_workers: int = 4,
):

    if config["data"]["data_dir_names"] == ["03", "04", "05"]:
        all_data_dirs = get_all_data_dir_paths(data_dir_root)
    elif config["data"]["data_dir_names"] == ["10"]:
        all_data_dirs = get_all_new_lr_data_dir_paths(data_dir_root)
    elif config["data"]["data_dir_names"] == ["20"]:
        logger.info("Data dirs are [20]")
        all_data_dirs = get_all_new_lr_data_dir_paths(data_dir_root, dir_name="20")
    else:
        raise Exception(
            f'Data dirs {config["data"]["data_dir_names"]} are not supported.'
        )

    train_valid_test_dirs = split_into_train_valid_test_dirs(
        all_data_dirs, config["data"]["train_valid_test_ratios"]
    )

    dataloaders, _ = make_dataloaders(
        data_dirs=train_valid_test_dirs,
        hr_3d_build_path=all_data_dirs[0].parent / "hr_is_in_build.npy",
        hr_org_size=tuple(config["data"]["hr_org_size"]),
        hr_crop_size=tuple(config["data"]["hr_crop_size"]),
        batch_size=batch_size,
        means=config["data"]["means"],
        stds=config["data"]["stds"],
        nan_value=config["data"]["nan_value"],
        num_workers=num_workers,
        use_clipping=False,
        datasizes=config["data"]["datasizes"],
        seed=config["data"]["seed"],
        lr_scaling=config["data"].get("lr_scaling", None),
        scale_factor=config["data"].get("scale_factor", 4),
    )

    return dataloaders["test"]