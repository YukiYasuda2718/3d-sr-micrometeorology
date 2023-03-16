import argparse
import copy
import datetime
import gc
import os
import pathlib
import shutil
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from src.dataloader import (
    get_all_data_dir_paths,
    get_all_new_lr_data_dir_paths,
    make_dataloaders,
    make_evaluation_dataloader_without_random_cropping,
    split_into_train_valid_test_dirs,
)
from src.gradnorm import GradNorm
from src.loss_maker import (
    AbsDiffDivergence,
    AbsDiffTemperature,
    DiffOmegaVectorNorm,
    DiffVelocityVectorNorm,
    MaskedL1Loss,
    MaskedL1LossNearWall,
    MyL1Loss,
    ResidualContinuity,
    make_loss,
)
from src.model_maker import make_model
from src.optim_helper import evaluate, test_ddp, train_ddp
from src.utils import set_seeds
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=True)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--world_size", type=int, default=2)

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())
DL_DATA_DIR = pathlib.Path(f"{ROOT_DIR}/data/DL_data")

mlflow.set_tracking_uri(f"{ROOT_DIR}/mlruns/")


def check_hr_data_paths_in_dataloader(dataloader):
    for path in dataloader.dataset.hr_files:
        logger.info(os.path.basename(path))


def _log_params(config: dict):
    for k1, v1 in config.items():
        if isinstance(v1, list):
            mlflow.log_param(k1, "/".join(map(str, v1)))
        elif isinstance(v1, dict):
            for k2, v2 in v1.items():
                mlflow.log_param(f"{k1}/{k2}", v2)
        else:
            mlflow.log_param(k1, v1)


def mlflow_log_params(config: dict):
    _log_params(config["model"])
    _log_params(config["train"])
    _log_params(config["data"])


def write_out_inferences(*, test_loader, model, inference_dir, device):
    l1_loss = torch.nn.L1Loss()

    for hr_file_path, (Xs, bs, ys) in zip(test_loader.dataset.hr_files, test_loader):
        bs = bs.unsqueeze(1)  # add channel dim
        assert Xs.shape == (1, 4, 8, 80, 80)
        assert bs.shape == (1, 1, 32, 320, 320)
        assert ys.shape == (1, 4, 32, 320, 320)

        with torch.no_grad():
            preds = model(Xs.to(device), bs.to(device)).cpu().detach()
        assert preds.shape == ys.shape

        str_datetime = os.path.basename(hr_file_path).split("_")[0]

        for label, data in zip(["LR", "BM", "HR", "SR"], [Xs, bs, ys, preds]):
            np.save(f"{inference_dir}/{str_datetime}_{label}.npy", data.numpy())

        logger.info(f"{str_datetime}, l1 = {l1_loss(preds, ys).item():.7f}")


def setup(rank: int, world_size: int, backend: str = "nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def train_and_validate(
    rank: int,
    world_size: int,
    config: dict,
    weight_path: str,
    learning_history_path: str,
):

    setup(rank, world_size)
    set_seeds(config["train"]["seed"])
    use_grad_norm = "grad_norm" in config["train"]

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Make dataloaders and samplers")
        logger.info("################################\n")

    if config["data"]["data_dir_names"] == ["03", "04", "05"]:
        all_data_dirs = get_all_data_dir_paths(DL_DATA_DIR)
    elif config["data"]["data_dir_names"] == ["10"]:
        all_data_dirs = get_all_new_lr_data_dir_paths(DL_DATA_DIR)
    elif config["data"]["data_dir_names"] == ["20"]:
        logger.info("LR data (8xSR) are used.")
        all_data_dirs = get_all_new_lr_data_dir_paths(DL_DATA_DIR, dir_name="20")
    else:
        raise Exception(
            f'data_dir_names == {config["data"]["data_dir_names"]} are not supported.'
        )

    train_valid_test_dirs = split_into_train_valid_test_dirs(
        all_data_dirs, config["data"]["train_valid_test_ratios"]
    )

    dataloaders, samplers = make_dataloaders(
        rank=rank,
        world_size=world_size,
        data_dirs=train_valid_test_dirs,
        hr_3d_build_path=all_data_dirs[0].parent / "hr_is_in_build.npy",
        hr_org_size=tuple(config["data"]["hr_org_size"]),
        hr_crop_size=tuple(config["data"]["hr_crop_size"]),
        batch_size=config["data"]["batch_size"],
        means=config["data"]["means"],
        stds=config["data"]["stds"],
        nan_value=config["data"]["nan_value"],
        num_workers=2,
        datasizes=config["data"]["datasizes"],
        seed=config["data"]["seed"],
        lr_scaling=config["data"].get("lr_scaling", None),
        max_discarded_lr_z_index=config["data"].get("max_discarded_lr_z_index", None),
        scale_factor=config["data"].get("scale_factor", 4),
    )

    if rank == 0:
        logger.info("\nCheck train_loader")
        check_hr_data_paths_in_dataloader(dataloaders["train"])
        logger.info("\nCheck valid_loader")
        check_hr_data_paths_in_dataloader(dataloaders["valid"])

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Make model and optimizer")
        logger.info("###############################\n")
        logger.info(f"Use GradNorm = {use_grad_norm}")

    model = make_model(config)
    model = DDP(model.to(rank), device_ids=[rank])
    loss_fn = make_loss(config)

    if not use_grad_norm:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
        grad_norm = None
    else:
        grad_norm = GradNorm(
            n_tasks=config["train"]["grad_norm"]["n_tasks"],
            alpha=config["train"]["grad_norm"]["alpha"],
            output_dir_path=os.path.dirname(weight_path),
            device=rank,
            clipping_weight_min=config["train"]["grad_norm"].get(
                "clipping_weight_min", None
            ),
        )
        param_groups = [
            {"params": model.parameters()},
            {"params": grad_norm.weights, "lr": config["train"]["grad_norm"]["lr"]},
        ]
        optimizer = torch.optim.Adam(param_groups, lr=config["train"]["lr"])

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Train model")
        logger.info("###############################\n")
    all_scores = []
    best_epoch = 0
    best_loss = np.inf
    best_weights = copy.deepcopy(model.module.state_dict())

    for epoch in range(config["train"]["num_epochs"]):
        _time = time.time()
        dist.barrier()
        loss = train_ddp(
            dataloader=dataloaders["train"],
            sampler=samplers["train"],
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            num_loops=config["train"]["num_loops_train"],
            grad_norm=grad_norm,
        )
        dist.barrier()
        val_loss = test_ddp(
            dataloader=dataloaders["valid"],
            sampler=samplers["valid"],
            model=model,
            loss_fn=loss_fn,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            num_loops=config["train"]["num_loops_valid"],
            grad_norm=grad_norm,
        )
        dist.barrier()

        losses = {"loss": loss, "val_loss": val_loss}
        all_scores.append(losses)

        if use_grad_norm:
            grad_norm.record_and_write_out_weights_and_losses()

        if rank == 0:
            logger.info(
                f"Epoch: {epoch + 1}, loss = {loss:.8f}, val_loss = {val_loss:.8f}"
            )
            # mlflow.log_metrics(losses, step=epoch + 1)

        if rank == 0 and val_loss <= best_loss:
            best_epoch = epoch + 1
            best_loss = val_loss
            best_weights = copy.deepcopy(model.module.state_dict())

            torch.save(best_weights, weight_path)
            logger.info("Best loss is updated.")

        if rank == 0 and epoch % 10 == 0:
            pd.DataFrame(all_scores).to_csv(learning_history_path, index=False)

        logger.info(f"Elapsed time = {time.time() - _time} sec")
        logger.info("-----")

    if rank == 0:
        torch.save(best_weights, weight_path)
        pd.DataFrame(all_scores).to_csv(learning_history_path, index=False)

    cleanup()


if __name__ == "__main__":
    try:

        os.environ["MASTER_ADDR"] = "localhost"

        # Port is arbitrary, but set random value to avoid collision
        np.random.seed(datetime.datetime.now().microsecond)
        port = str(np.random.randint(12000, 65535))
        os.environ["MASTER_PORT"] = port
        # max port 65535 is determined by Earth Simulator.
        # min port 12000 may be arbitrary.

        world_size = parser.parse_args().world_size
        config_path = parser.parse_args().config_path

        with open(config_path) as file:
            config = yaml.safe_load(file)

        experiment_name = "unet_model"
        config_name = os.path.basename(config_path).split(".")[0]

        _dir = f"{ROOT_DIR}/data/DL_results/{experiment_name}/{config_name}"
        os.makedirs(_dir, exist_ok=False)

        weight_path = f"{_dir}/weights.pth"
        learning_history_path = f"{_dir}/learning_history.csv"
        logger.addHandler(FileHandler(f"{_dir}/log.txt"))

        INFERENCE_DIR = f"{ROOT_DIR}/data/DL_inferences/{experiment_name}/{config_name}"
        if os.path.exists(INFERENCE_DIR):
            shutil.rmtree(INFERENCE_DIR)
        os.makedirs(INFERENCE_DIR, exist_ok=False)

        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=config_name)
        mlflow.set_tags(
            {
                "config_name": config_name,
                "config_path": config_path,
                "weight_path": weight_path,
                "inference_dir": INFERENCE_DIR,
                "note": "multiple_gpus",
                "world_size": str(world_size),
            }
        )
        mlflow_log_params(config)

        logger.info("\n*********************************************************")
        logger.info(f"Start DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        logger.info(f"config path = {config_path}")

        if not torch.cuda.is_available():
            logger.error("No GPU.")
            raise Exception("No GPU.")

        logger.info(f"Num available GPUs = {torch.cuda.device_count()}")
        logger.info(f"Names of GPUs = {torch.cuda.get_device_name()}")
        logger.info(f"Device capability = {torch.cuda.get_device_capability()}")
        logger.info(f"World size = {world_size}")

        start_time = time.time()

        mp.spawn(
            train_and_validate,
            args=(world_size, config, weight_path, learning_history_path),
            nprocs=world_size,
            join=True,
        )

        end_time = time.time()

        logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min")

        logger.info("\n*********************************************************")
        logger.info("Evaluate model and make inferences")
        logger.info("*********************************************************\n")

        device = "cuda:0"

        test_loader = make_evaluation_dataloader_without_random_cropping(
            config, DL_DATA_DIR, batch_size=1
        )
        logger.info("check test_loader")
        check_hr_data_paths_in_dataloader(test_loader)

        # Re-load best weights
        model = make_model(config).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        loss_fns = {
            "L1": MyL1Loss(),
            "MaskedL1": MaskedL1Loss(),
            "MaskedL1NearWall": MaskedL1LossNearWall(),
            "ResidualContinuityEq": ResidualContinuity(config["data"]["stds"][1:]),
            "AbsDiffTemperature": AbsDiffTemperature(config["data"]["stds"][0]),
            "DiffVelocityNorm": DiffVelocityVectorNorm(config["data"]["stds"][1:]),
            "AbsDiffTemperatureLevZero": AbsDiffTemperature(
                config["data"]["stds"][0], lev=0
            ),
            "DiffVelocityNormLevZero": DiffVelocityVectorNorm(
                config["data"]["stds"][1:], lev=0
            ),
            "AbsDiffDivergence": AbsDiffDivergence(config["data"]["stds"][1:]),
            "DiffOmegaVectorNorm": DiffOmegaVectorNorm(config["data"]["stds"][1:]),
        }

        results = evaluate(
            dataloader=test_loader,
            model=model,
            loss_fns=loss_fns,
            device=device,
            hide_progress_bar=True,
        )

        for k, v in results.items():
            mlflow.log_metric(k, v.avg)

        if config["train"].get("write_out_inferences", False):
            write_out_inferences(
                test_loader=test_loader,
                model=model,
                inference_dir=INFERENCE_DIR,
                device=DEVICE,
            )

        mlflow.end_run()

        logger.info("\n*********************************************************")
        logger.info(f"End DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())