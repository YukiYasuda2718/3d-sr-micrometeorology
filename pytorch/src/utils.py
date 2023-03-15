import os
import pickle
import random
import typing
from logging import getLogger

import numpy as np
import pandas as pd
import torch

logger = getLogger()


class RandomCrop3D:
    """
    https://discuss.pytorch.org/t/efficient-way-to-crop-3d-image-in-pytorch/78421
    """

    def __init__(self, img_sz, crop_sz):
        assert img_sz[0] >= crop_sz[0]
        assert img_sz[1] >= crop_sz[1]
        assert img_sz[2] >= crop_sz[2]
        self.img_sz = tuple(img_sz)
        self.crop_sz = tuple(crop_sz)

    def __call__(self, x):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *slice_hwd)

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            if sz == crop_sz:
                lower_bound = 0
            else:
                lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        logger.debug(f"slice_h = {slice_h}, slice_w = {slice_w}, slice_d = {slice_d}")
        return x[
            ...,
            slice_h[0] : slice_h[1],
            slice_w[0] : slice_w[1],
            slice_d[0] : slice_d[1],
        ]


class AverageMeter(object):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seeds(seed: int = 42, use_deterministic: bool = False) -> None:
    """
    Do not forget to run `torch.use_deterministic_algorithms(True)`
    just after importing torch in your main program.

    # How to reproduce the same results using pytorch.
    # https://pytorch.org/docs/stable/notes/randomness.html
    """
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if use_deterministic:
            torch.use_deterministic_algorithms(True)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        logger.error(e)


def seed_worker(worker_id: int):
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_torch_generator(seed: int = 42) -> torch.Generator:
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def count_model_params(model: torch.nn.Module) -> int:
    num_params, total_num_params = 0, 0
    for p in model.parameters():
        total_num_params += p.numel()
        if p.requires_grad:
            num_params += p.numel()

    assert num_params == total_num_params

    return num_params


def calc_early_stopping_patience(
    df: pd.DataFrame, col: str = "val_loss", th_max_cnt: int = 50
) -> int:

    th_val = np.inf
    cnt = 0
    max_cnt = 0
    is_over = False
    for i, val in enumerate(df[col].values):

        if val <= th_val:
            th_val = val
            cnt = 0
            # print(f"Best loss is updated. Epoch = {i+1}, loss = {val}")
        else:
            cnt += 1
            if cnt > max_cnt:
                max_cnt = cnt

                if not is_over and max_cnt > th_max_cnt:
                    print(f"Epoch = {i+1}, max_cnt = {max_cnt}, loss = {val}")
                    is_over = True
                # print(f"Epoch = {i+1}, max_cnt = {max_cnt}, loss = {val}")
    if not is_over:
        raise Exception("Val loss is never over threshold.")
    return max_cnt


def read_pickle(file_path: str) -> typing.Any:
    with open(str(file_path), "rb") as p:
        return pickle.load(p)


def write_pickle(data: typing.Any, file_path: str) -> None:
    with open(str(file_path), "wb") as p:
        pickle.dump(data, p)