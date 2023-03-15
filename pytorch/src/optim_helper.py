import sys
import typing
from logging import getLogger

import torch
import torch.distributed as dist
from src.gradnorm import GradNorm
from src.utils import AverageMeter
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    num_loops: int = 1,
    hide_progress_bar: bool = False,
    grad_norm: GradNorm = None,
) -> float:

    train_loss = AverageMeter()
    model.train()

    with tqdm(total=len(dataloader) * num_loops, disable=hide_progress_bar) as t:
        for _ in range(num_loops):
            for Xs, bs, ys in dataloader:
                bs = bs.unsqueeze(1)  # add channel dim
                Xs, bs, ys = Xs.to(device), bs.to(device), ys.to(device)

                preds = model(Xs, bs)

                if grad_norm is None:
                    loss = loss_fn(preds, ys, bs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    logger.debug("GradNorm is used in training.")
                    _losses = loss_fn.calc_loss_terms(
                        predicts=preds, targets=ys, masks=bs
                    )
                    optimizer.zero_grad()
                    loss = grad_norm.backward(
                        loss_list=list(_losses),
                        last_shared_params=model.get_last_params(),
                    )
                    optimizer.step()
                    grad_norm.renormalize_weights()

                train_loss.update(loss.item(), n=len(Xs))
                t.update(1)

    logger.info(f"Train error: avg loss = {train_loss.avg:.8f}")

    return train_loss.avg


def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    device: str,
    num_loops: int = 1,
    hide_progress_bar: bool = False,
    grad_norm: GradNorm = None,
) -> float:

    val_loss = AverageMeter()
    model.eval()

    with tqdm(total=len(dataloader) * num_loops, disable=hide_progress_bar) as t:
        with torch.no_grad():
            for _ in range(num_loops):
                for Xs, bs, ys in dataloader:
                    bs = bs.unsqueeze(1)  # add channel dim
                    Xs, bs, ys = Xs.to(device), bs.to(device), ys.to(device)

                    preds = model(Xs, bs)

                    if grad_norm is None:
                        loss = loss_fn(preds, ys, bs)
                    else:
                        logger.debug("GradNorm is used in test.")
                        _losses = loss_fn.calc_loss_terms(
                            predicts=preds, targets=ys, masks=bs
                        )
                        loss = grad_norm.calc_total_weighted_loss_for_test(
                            list(_losses)
                        )

                    val_loss.update(loss.item(), n=len(Xs))
                    t.update(1)

    logger.info(f"Valid error: avg loss = {val_loss.avg:.8f}")

    return val_loss.avg


def evaluate(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fns: typing.Dict[str, typing.Callable],
    device: str,
    hide_progress_bar: bool = True,
) -> dict:

    dict_loss = {k: AverageMeter() for k in loss_fns.keys()}

    with torch.no_grad(), tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        for Xs, bs, ys in dataloader:
            bs = bs.unsqueeze(1)  # add channel dim

            Xs, bs, ys = Xs.to(device), bs.to(device), ys.to(device)
            preds = model(Xs, bs)

            for loss_name, loss_fn in loss_fns.items():
                lss = loss_fn(preds, ys, bs).item()
                dict_loss[loss_name].update(lss, n=len(Xs))
            t.update(1)

    return dict_loss


def train_ddp(
    dataloader: DataLoader,
    sampler: Sampler,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    epoch: int,
    rank: int,
    world_size: int,
    num_loops: int,
    grad_norm: GradNorm = None,
) -> float:

    mean_loss, cnt = 0.0, 0

    sampler.set_epoch(epoch)
    model.train()

    for _ in range(num_loops):
        for Xs, bs, ys in dataloader:
            bs = bs.unsqueeze(1)  # add channel dim
            Xs, bs, ys = Xs.to(rank), bs.to(rank), ys.to(rank)

            preds = model(Xs, bs)

            if grad_norm is None:
                loss = loss_fn(preds, ys, bs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                _losses = loss_fn.calc_loss_terms(predicts=preds, targets=ys, masks=bs)
                optimizer.zero_grad()
                loss = grad_norm.backward(
                    loss_list=list(_losses),
                    last_shared_params=model.module.get_last_params(),
                )
                optimizer.step()
                grad_norm.renormalize_weights()

            mean_loss += loss * Xs.shape[0]
            cnt += Xs.shape[0]
    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size


def test_ddp(
    dataloader: DataLoader,
    sampler: Sampler,
    model: nn.Module,
    loss_fn: nn.functional,
    epoch: int,
    rank: int,
    world_size: int,
    num_loops: int,
    grad_norm: GradNorm = None,
) -> float:

    mean_loss, cnt = 0.0, 0

    sampler.set_epoch(epoch)
    model.eval()

    with torch.no_grad():
        for _ in range(num_loops):
            for Xs, bs, ys in dataloader:
                bs = bs.unsqueeze(1)  # add channel dim
                Xs, bs, ys = Xs.to(rank), bs.to(rank), ys.to(rank)

                preds = model(Xs, bs)

                if grad_norm is None:
                    loss = loss_fn(preds, ys, bs)
                else:
                    _losses = loss_fn.calc_loss_terms(
                        predicts=preds, targets=ys, masks=bs
                    )
                    loss = grad_norm.calc_total_weighted_loss_for_test(list(_losses))

                mean_loss += loss * Xs.shape[0]
                cnt += Xs.shape[0]
    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size