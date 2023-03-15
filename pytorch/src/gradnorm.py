from copy import deepcopy
from logging import getLogger
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn

logger = getLogger()


class GradNorm:
    """
    https://github.com/AvivNavon/AuxiLearn/blob/8ff7dd28ab045a817757a5970cd02f14af983e3d/experiments/weight_methods.py
    """

    def __init__(
        self,
        n_tasks: int,
        alpha: float = 1.5,
        device: str = None,
        output_dir_path: str = ".",
        clipping_weight_min: float = None,
        **kwargs,
    ):
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.weights = torch.ones((n_tasks,), requires_grad=True, device=device)
        self.dir_path = output_dir_path
        self.device = device
        self.clipping_min = clipping_weight_min
        logger.info(f"GradNorm clipping_weight_min = {self.clipping_min}")

        self.init_losses = None
        self._losses = []
        self.recorded_weights = []
        self.recorded_losses = []

        logger.info(f"GradNorm params: n_tasks = {self.n_tasks}, alpha = {self.alpha}")

    def renormalize_weights(self):
        with torch.no_grad():
            if self.clipping_min is not None:
                self.weights = self.weights.clamp_(min=self.clipping_min)
            renormalize_coeff = self.n_tasks / self.weights.sum()
            self.weights *= renormalize_coeff
        self.weights.requires_grad = True

    def calc_total_weighted_loss_for_test(
        self,
        loss_list: List[nn.Module],
    ):
        with torch.no_grad():
            losses = torch.stack(loss_list)
            weighted_losses = self.weights * losses
            self._losses.append(losses.detach().cpu().numpy())
            return weighted_losses.sum()

    def record_and_write_out_weights_and_losses(self):
        self.recorded_weights.append(deepcopy(self.weights.detach().cpu().numpy()))

        mean_losses = np.mean(np.stack(self._losses, axis=0), axis=0)
        self._losses = []
        self.recorded_losses.append(mean_losses)

        pd.DataFrame(self.recorded_weights).to_csv(
            f"{self.dir_path}/grad_norm_weights_{self.device}.csv"
        )
        pd.DataFrame(self.recorded_losses).to_csv(
            f"{self.dir_path}/grad_norm_losses_{self.device}.csv"
        )

    def backward(
        self,
        loss_list: List[nn.Module],
        last_shared_params: List[torch.nn.Parameter],
        return_total_weighted_loss: bool = True,
        **kwargs,
    ):
        losses = torch.stack(loss_list)

        if self.init_losses is None:
            self.init_losses = losses.detach().clone().data

        weighted_losses = self.weights * losses
        total_weighted_loss = weighted_losses.sum()
        total_weighted_loss.backward(retain_graph=True)

        # Reset the gradients of `self.weights`
        # These gradients are calculated in the above `backward`
        self.weights.grad = torch.zeros_like(self.weights.grad)

        norms = []
        for w_i, L_i in zip(self.weights, losses):
            # The first element is extracted from the returned tuple of `torch.autograd.grad`
            # `last_shared_params` is generally a list, so the returned tuple contains a gradient for each element of this param list.
            # `retain_graph=True` is necessary because each `grd_L_i` is a derivative with respect to the common params of the model.
            grd_L_i = torch.autograd.grad(L_i, last_shared_params, retain_graph=True)[0]
            norms.append(torch.norm(w_i * grd_L_i))

        norms = torch.stack(norms)

        with torch.no_grad():
            loss_ratios = losses / self.init_losses
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            constant_term = norms.mean() * (inverse_train_rates ** self.alpha)
            constant_term = constant_term.detach().clone()

        grad_norm_loss = (norms - constant_term).abs().sum()
        self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
        # `self.weights` is just a torch.Tensor, so the regurned tuple contains only the first element.

        if return_total_weighted_loss:
            return total_weighted_loss