from typing import Tuple, Iterator

import numpy as np


def crop3D(
    xs: np.ndarray, window_shape: Tuple[int, int, int], strides: Tuple[int, int, int]
) -> np.ndarray.view:
    assert xs.ndim == 3
    return np.lib.stride_tricks.sliding_window_view(xs, window_shape=window_shape, writeable=False)[
        :: strides[0], :: strides[1], :: strides[2], :, :, :
    ]


def average3D(xs: np.ndarray, scale_factor: int) -> np.ndarray:
    assert xs.ndim == 3
    w = (scale_factor, scale_factor, scale_factor)
    # stride == window, so any window is not overlapped.
    return crop3D(xs, w, w).mean(axis=(3, 4, 5))


def get_cropped_size(org_size: int, window_width: int, stride: int) -> int:
    return (org_size - window_width) // stride + 1


def check_crop3D_result(
    org_shape: Tuple[int, int, int],
    cropped_shape: Tuple[int, int, int],
    window_shape: Tuple[int, int, int],
    strides: Tuple[int, int, int],
):
    for org_size, cropped_size, window_width, stride in zip(
        org_shape, cropped_shape, window_shape, strides
    ):
        actual_cropped_size = get_cropped_size(org_size, window_width, stride)
        assert actual_cropped_size == cropped_size, "Invalid cropped size!"


def convert_strided_index_to_org_index(strided_index: int, stride: int) -> int:
    return strided_index * stride


def generate_patch_from_cropped_data(
    cropped: np.ndarray.view, strides: Tuple[int, int, int], box_shape: Tuple[int, int, int]
) -> Iterator[Tuple[np.ndarray.view, Tuple[int, int, int]]]:

    assert cropped.ndim == 6

    for i in range(cropped.shape[0]):
        org_i = convert_strided_index_to_org_index(i, strides[0])
        for j in range(cropped.shape[1]):
            org_j = convert_strided_index_to_org_index(j, strides[1])
            for k in range(cropped.shape[2]):
                org_k = convert_strided_index_to_org_index(k, strides[2])
                out_arry = cropped[i, j, k, :, :, :]
                assert out_arry.shape == box_shape, "Box shape is invalid!"
                yield (out_arry, (org_i, org_j, org_k))


def cut_2Dmargins(xs: np.ndarray, margins: Tuple[int, int]) -> np.ndarray:
    assert xs.ndim == 3
    return xs[margins[0] : -margins[0], margins[1] : -margins[1], :]
