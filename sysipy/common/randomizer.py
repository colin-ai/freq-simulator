import numpy as np
from numpy.random import default_rng


def randomize(
    n_samples_interval,
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

    xp = _linspace(y_breakpoints, n_samples_interval)
    yp = np.array(y_breakpoints)

    xp = np.tile(xp, reps=(n_seq, 1))
    yp = np.tile(yp, reps=(n_seq, 1))

    if _is_randomize(y_delta):
        _add_jitter(yp, x_excluded, y_delta)
    if _is_randomize(x_delta):
        _add_jitter(xp, x_excluded, x_delta)

    return xp, yp


# LINSPACE
def _linspace(y_breakpoints, n_points_interval):
    n_breakpoints = _get_n_breakpoints(y_breakpoints)
    x_limit = _get_xlim(n_breakpoints, n_points_interval)
    n_points_interval = _get_n_points_interval(n_points_interval)
    return np.arange(0, x_limit, n_points_interval)


def _get_xlim(n_breakpoints, n_points_interval):
    return int(n_breakpoints * n_points_interval)


def _get_n_breakpoints(y_breakpoints):
    return len(y_breakpoints)


def _get_n_points_interval(n_points_interval):
    return int(n_points_interval)


# JITTER
def _is_randomize(a):
    return a != 0


def _add_jitter(a, x_excluded, delta_jitter):
    n_seq = a.shape[0]
    seq_len = a.shape[-1]
    return a[:, x_excluded:-x_excluded] + _uniform_jitter(
        delta_jitter, n_seq, seq_len, x_excluded
    )


def _uniform_jitter(delta_jitter, n_seq, seq_len, x_excluded):
    rng = default_rng()
    return rng.uniform(-delta_jitter, delta_jitter, (n_seq, seq_len - x_excluded * 2))
