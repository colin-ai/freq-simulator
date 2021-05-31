from typing import Callable

from numpy import arange, asarray, interp, loadtxt, polyfit, polyval, savetxt, zeros
from numpy.random import randint
from torch import Tensor

from simulator.model import slip_velocities


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


def breakpoints_randomize(
    x_breakpoints, y_breakpoints, n_seq, y_delta=30, xp_rand=True, x_delta=5
):
    xp = asarray(x_breakpoints)
    yp = asarray(y_breakpoints)
    sets = zeros([n_seq, 2, len(yp)])
    for i in range(n_seq):
        yp_rand = yp.copy()
        yp_rand[2:-2] += randint(
            -y_delta, y_delta, len(yp) - 4
        )  # Avoid randomization of the two first and last values
        if xp_rand is True:
            xp_rand = xp.copy()
            xp_rand[2:-2] += randint(-x_delta, x_delta, len(xp) - 4)  # Idem
            sets[i] = [xp_rand, negative_value_thresholder(yp_rand)]
        else:
            sets[i] = [xp, negative_value_thresholder(yp_rand)]

    return sets


def wheel_velocities_computer(velocities_vehicle, func_interp, m, freq, F, k):
    velocities_set = zeros(
        [2 * velocities_vehicle.shape[0], int(velocities_vehicle[0][0][-1])]
    )

    for i, set in enumerate(velocities_vehicle):
        velocities_set[i * 2] = func_interp(set[0], set[1])
        velocities_set[(i * 2 + 1)], *_ = slip_velocities(
            velocities_set[i * 2], m, freq, F, k, verbose=False
        )

    return velocities_set


def init_dataset(input):
    n_seq = input.shape[0]
    x_limit = int(input[0, 0, -1])
    return zeros([n_seq, 1, x_limit])


# def data_interpolation(position, func_interp, dt):
#     N_DERIVED = 3
#     dataset = init_dataset(position, n_derived=N_DERIVED)
#     for i, set in enumerate(position):
#         dataset[i * N_DERIVED] = func_interp(set[0], set[1])
#         dataset[(i * N_DERIVED + 1)], *_ = derived(dataset[i * N_DERIVED], dt)
#         dataset[(i * N_DERIVED + 2)], *_ = derived(dataset[i * N_DERIVED + 1], dt)
#     return dataset


def data_interpolation(position, func_interp, dt):
    dataset = init_dataset(position)
    for i, set in enumerate(position):
        dataset[i] = func_interp(set[0], set[1])
    return dataset


def add_derivation(
    dataset: asarray, n_derived: int, func_derivation: Callable, dt: float
):
    dataset_expanded = zeros((dataset.shape[0], n_derived + 1, dataset.shape[-1]))
    dataset_expanded[:, [0], :] = dataset
    for i in range(n_derived):
        dataset_expanded[:, [i + 1], 1:] = func_derivation(
            dataset_expanded[:, [i], 1:] - dataset_expanded[:, [i], :-1], dt
        )
    return dataset_expanded


def derivation(diff: Tensor, dt: Tensor) -> Tensor:
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


def seq_generator_tire_slip(
    y_breakpoints,
    n_seq,
    xp_rand,
    func_interp,
    m,
    freq,
    step_breakpoint,
    F_long,
    alpha,
):
    seq_end = int(len(y_breakpoints) * step_breakpoint / freq)
    step = int(step_breakpoint / freq)
    x_breakpoints = [x for x in range(0, seq_end, step)]
    v_vehicle = breakpoints_randomize(
        x_breakpoints, y_breakpoints, n_seq, xp_rand=xp_rand, x_delta=5
    )
    velocities = wheel_velocities_computer(
        v_vehicle, func_interp, m, freq, F_long, alpha
    )

    return velocities

def seq_generator_tracking(
    y_breakpoints,
    n_seq,
    xp_rand,
    func_interp,
    freq,
    n_points_interval,
):  # sourcery skip: inline-immediately-returned-variable

    N_DERIVED = 2
    n_breakpoints = len(y_breakpoints)
    x_limit = int(n_breakpoints * n_points_interval / freq)
    n_points_interval = int(n_points_interval / freq)
    x_breakpoints = arange(0, x_limit, n_points_interval)
    xy_breakpoints_rand = breakpoints_randomize(
        x_breakpoints, y_breakpoints, n_seq, xp_rand=xp_rand, x_delta=5
    )
    y_interpolated = data_interpolation(xy_breakpoints_rand, func_interp, dt=1.0)
    dataset = add_derivation(
        dataset=y_interpolated, n_derived=N_DERIVED, func_derivation=derivation, dt=10.0
    )
    dataset = dataset.reshape(n_seq*(N_DERIVED+1), -1)
    return dataset
