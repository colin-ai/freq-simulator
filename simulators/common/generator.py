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


def randomizer(
    x_breakpoints,
    y_breakpoints,
    n_seq: int,
    y_delta: int,
    x_delta: int,
    x_excluded: int,
):
    if x_excluded < 1.0:
        raise ValueError(
            f"Value of x_excluded should be higher or equal than 1 Value found {x_excluded}."
        )

    rng = default_rng()
    xp = np.array(x_breakpoints)
    yp = np.array(y_breakpoints)
    seq_len = len(xp)

    yp = np.tile(yp, reps=(n_seq, 1))
    xp = np.tile(xp, reps=(n_seq, 1))

    if y_delta != 0:
        yp[:, x_excluded:-x_excluded] += rng.integers(
            -y_delta, y_delta, (n_seq, seq_len - x_excluded * 2)
        )
    if x_delta != 0:

        xp[:, x_excluded:-x_excluded] += rng.integers(
            -x_delta, x_delta, (n_seq, seq_len - x_excluded * 2)
        )

    return xp, yp


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


def _get_xlim(n_breakpoints, n_points_interval):
    return int(n_breakpoints * n_points_interval)


def _get_n_breakpoints(y_breakpoints):
    return len(y_breakpoints)


def _get_n_points_interval(n_points_interval):
    return int(n_points_interval)


def _get_x_mesh(y_breakpoints, n_points_interval):
    n_breakpoints = _get_n_breakpoints(y_breakpoints)
    x_limit = _get_xlim(n_breakpoints, n_points_interval)
    n_points_interval = _get_n_points_interval(n_points_interval)
    return arange(0, x_limit, n_points_interval)


def randomize_breakpoints(
    y_breakpoints,
    n_seq,
    y_delta,
    x_delta,
    x_excluded,
    n_points_interval,
):
    x_mesh = _get_x_mesh(y_breakpoints, n_points_interval)
    return randomizer(
        x_breakpoints=x_mesh,
        y_breakpoints=y_breakpoints,
        n_seq=n_seq,
        y_delta=y_delta,
        x_delta=x_delta,
        x_excluded=x_excluded,
    )


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
