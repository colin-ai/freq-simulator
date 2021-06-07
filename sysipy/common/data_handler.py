from typing import Callable

import numpy.typing as npt
from numpy import loadtxt, zeros

ndarray = npt.ArrayLike


def add_derivation(to_derive: ndarray, n_derived: int, func_derivation: Callable, dt: float):
    dataset_expanded = zeros((to_derive.shape[0], n_derived + 1, to_derive.shape[-1]))
    dataset_expanded[:, [0], :] = to_derive
    for i in range(n_derived):
        dataset_expanded[:, [i + 1], 1:] = func_derivation(
            dataset_expanded[:, [i], 1:] - dataset_expanded[:, [i], :-1], dt
        )
    return dataset_expanded


def finite_diff(diff: ndarray, dt: float) -> ndarray:
    return diff / dt


def batch_dataset(filename, batch_size, seq_size=None):
    if seq_size is None:
        seq_size = -1
    dataset = loadtxt(filename, delimiter=";")
    dataset = dataset[:, :seq_size].reshape(batch_size, 2, seq_size)
    return dataset


