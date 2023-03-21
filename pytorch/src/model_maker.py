from logging import getLogger

import torch

from model.unet import UNetSR

logger = getLogger()


def make_model(config: dict) -> torch.nn.Module:

    if config["model"]["model_name"] == "unet":
        logger.info("UNetSR is created")
        return UNetSR(**config["model"])
    else:
        raise NotImplementedError(f"{config['model']['model_name']} is not supported")