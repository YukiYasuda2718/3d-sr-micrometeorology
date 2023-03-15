from logging import getLogger

import torch

from model.rrdn_v1 import ResidualInResidualDenseNetworkVer01
from model.rrdn_v2 import ResidualInResidualDenseNetworkVer02
from model.unet_v1 import UNetSRVer01
from model.unet_v2 import UNetSRVer02
from model.unet_v3 import UNetSRVer03
from model.unet_v4 import UNetSRVer04

logger = getLogger()


def make_model(config: dict) -> torch.nn.Module:

    if config["model"]["model_name"] == "rrdn_4ch_v1":
        logger.info("ResidualInResidualDenseNetworkVer01 is created")
        return ResidualInResidualDenseNetworkVer01(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            num_filters_rrdb=config["model"]["num_filters_rrdb"],
            num_layers_rrdb=config["model"]["num_layers_rrdb"],
            num_blocks_rrdb=config["model"]["num_blocks_rrdb"],
            num_rrdbs=config["model"]["num_rrdbs"],
            num_x2upsample=config["model"]["num_x2upsample"],
            num_filters_reconstruct=config["model"]["num_filters_reconstruct"],
            num_layers_reconstruct=config["model"]["num_layers_reconstruct"],
            kernel_size=config["model"]["kernel_size"],
            rescale_param=config["model"]["rescale_param"],
        )
    elif config["model"]["model_name"] == "rrdn_4ch_v2":
        logger.info("ResidualInResidualDenseNetworkVer02 is created")
        return ResidualInResidualDenseNetworkVer02(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            num_filters_rrdb=config["model"]["num_filters_rrdb"],
            num_layers_rrdb=config["model"]["num_layers_rrdb"],
            num_blocks_rrdb=config["model"]["num_blocks_rrdb"],
            num_rrdbs=config["model"]["num_rrdbs"],
            num_x2upsample=config["model"]["num_x2upsample"],
            num_filters_reconstruct=config["model"]["num_filters_reconstruct"],
            num_res_blocks_reconstruct=config["model"]["num_res_blocks_reconstruct"],
            kernel_size=config["model"]["kernel_size"],
            rescale_param_rrdn=config["model"]["rescale_param_rrdn"],
            rescale_param_rb=config["model"]["rescale_param_rb"],
        )
    elif config["model"]["model_name"] == "unet_4ch_v1":
        logger.info("UNetSRVer01 is created")
        return UNetSRVer01(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            num_feat0=config["model"]["num_feat0"],
            num_feat1=config["model"]["num_feat1"],
            num_feat2=config["model"]["num_feat2"],
            num_feat3=config["model"]["num_feat3"],
            num_x2upsample=config["model"]["num_x2upsample"],
            num_latent_layers=config["model"]["num_latent_layers"],
            conv_mode=config["model"].get("conv_mode", None),
        )
    elif config["model"]["model_name"] == "unet_4ch_v2":
        logger.info("UNetSRVer02 is created")
        return UNetSRVer02(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            num_feat0=config["model"]["num_feat0"],
            num_feat1=config["model"]["num_feat1"],
            num_feat2=config["model"]["num_feat2"],
            num_feat3=config["model"]["num_feat3"],
            num_x2upsample=config["model"]["num_x2upsample"],
            num_latent_layers=config["model"]["num_latent_layers"],
            conv_mode=config["model"].get("conv_mode", None),
        )
    elif config["model"]["model_name"] == "unet_4ch_v3":
        logger.info("UNetSRVer03 is created")
        return UNetSRVer03(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            num_feat0=config["model"]["num_feat0"],
            num_feat1=config["model"]["num_feat1"],
            num_feat2=config["model"]["num_feat2"],
            num_feat3=config["model"]["num_feat3"],
            num_feat4=config["model"].get("num_feat4", None),
            num_x2upsample=config["model"]["num_x2upsample"],
            num_latent_layers=config["model"]["num_latent_layers"],
            conv_mode=config["model"].get("conv_mode", None),
            n_layers_in_block=config["model"]["n_layers_in_block"],
        )
    elif config["model"]["model_name"] == "unet_4ch_v4":
        logger.info("UNetSRVer04 is created")
        return UNetSRVer04(
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