from logging import getLogger

import torch

from model.unet import UNetSR

logger = getLogger()


def make_model(config: dict) -> torch.nn.Module:

    if config["model"]["model_name"] == "unet":
        logger.info("UNetSR is created")
        return UNetSR(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            num_feat0=config["model"]["num_feat0"],
            num_feat1=config["model"]["num_feat1"],
            num_feat2=config["model"]["num_feat2"],
            num_feat3=config["model"]["num_feat3"],
            num_feat4=config["model"].get("num_feat4", None),
            num_x2upsample=config["model"]["num_x2upsample"],
            num_latent_layers=config["model"]["num_latent_layers"],
            n_layers_in_block=config["model"]["n_layers_in_block"],
            bias_feat_extraction=config["model"]["bias_feat_extraction"],
            conv_mode_feat_extraction=config["model"].get(
                "conv_mode_feat_extraction", None
            ),
            conv_mode_down_block=config["model"].get("conv_mode_down_block", None),
            conv_mode_up_block=config["model"].get("conv_mode_up_block", None),
        )
    else:
        raise NotImplementedError(f"{config['model']['model_name']} is not supported")