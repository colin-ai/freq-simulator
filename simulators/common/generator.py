from typing import Callable

import numpy as np
import numpy.typing as npt
from numpy import arange, asarray, interp, loadtxt, polyfit, polyval, savetxt, zeros
from numpy.random import randint, default_rng

ndarray = npt.ArrayLike


def negative_value_thresholder(velocity):
    """
    Force all negative position to zero value.
    """
    velocity[velocity < 0] = 0
    return velocity


def linear_interp(xp, yp):
    x = arange(0.0, xp[-1])
    velocities = interp(x, xp, yp)
    velocities = negative_value_thresholder(velocities)
    return velocities


def polyn_interp(xp, yp):
    x = arange(0.0, xp[-1])
    p = polyfit(xp, yp, deg=int(len(yp) * 0.7))
    velocities = polyval(p, x)
    velocities = negative_value_thresholder(velocities)
    return velocities



def _init_dataset(input):
    n_seq = input.shape[0]
    x_limit = int(input[0, -1])
    return zeros([n_seq, 1, x_limit])


# def data_interpolation(position, func_interp, dt):
#     N_DERIVED = 3
#     data = _init_dataset(position, n_derived=N_DERIVED)
#     for i, set in enumerate(position):
#         data[i * N_DERIVED] = func_interp(set[0], set[1])
#         data[(i * N_DERIVED + 1)], *_ = derived(data[i * N_DERIVED], dt)
#         data[(i * N_DERIVED + 2)], *_ = derived(data[i * N_DERIVED + 1], dt)
#     return data


def data_interpolation(x, y, func_interp):
    dataset = _init_dataset(x)
    for i in range(len(x)):
        dataset[i] = func_interp(x[0], y[0])
    return dataset


def add_derivation(data: asarray, n_derived: int, func_derivation: Callable, dt: float):
    dataset_expanded = zeros((data.shape[0], n_derived + 1, data.shape[-1]))
    dataset_expanded[:, [0], :] = data
    for i in range(n_derived):
        dataset_expanded[:, [i + 1], 1:] = func_derivation(
            dataset_expanded[:, [i], 1:] - dataset_expanded[:, [i], :-1], dt
        )
    return dataset_expanded


def finite_diff(diff: ndarray, dt: float) -> ndarray:
    return diff / dt


def to_csv(X, filename, **kwargs):
    savetxt(filename, X, delimiter=";", **kwargs)


def read_dataset(filename):
    return loadtxt(filename, delimiter=";")


def batch_dataset(filename, batch_size, seq_size=None):
    if seq_size is None:
        seq_size = -1
    dataset = loadtxt(filename, delimiter=";")
    dataset = dataset[:, :seq_size].reshape(batch_size, 2, seq_size)
    return dataset


def seq_generator_freq(input):  # sourcery skip: inline-immediately-returned-variable
    """
    phase = φ(t) = 2π * Int(f(t) dt)
    freq = f(t) = 1/2π * ∂φ(t)/∂t
    pulsation = "angular velocity" = ω(t) = 2π*f(t) = ∂φ(t)/∂t

    :param input:
    :return:
    """

    N_DERIVED = 1

    n_seq = input.shape[0]
    dataset = add_derivation(
        data=input, n_derived=N_DERIVED, func_derivation=finite_diff, dt=10.0
    )
    dataset = dataset.reshape(n_seq * (N_DERIVED + 1), -1)
    return dataset
