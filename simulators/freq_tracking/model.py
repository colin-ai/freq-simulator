import numpy as np

from simulators.common.generator import (
    data_interpolation,
    add_derivation,
    finite_diff,
)
from simulators.common.randomizer import randomize


def compute_phase(frequencies, dt):
    return 0.5 * np.pi * np.cumsum(frequencies, axis=1) / dt


def freq_tracking_generator(
    freq_breakpoints,
    n_seq,
    y_delta,
    x_excluded,
    x_delta,
    n_samples_interval,
    func_interp,
):  # sourcery skip: inline-immediately-returned-variable
    x_rand, y_rand = randomize(
        y_breakpoints=freq_breakpoints,
        n_seq=n_seq,
        y_delta=y_delta,
        x_excluded=x_excluded,
        x_delta=x_delta,
        n_samples_interval=n_samples_interval,
    )
    frequencies = data_interpolation(x_rand, y_rand, func_interp=func_interp)
    dataset = add_derivation(
        data=frequencies, n_derived=1, func_derivation=finite_diff, dt=10.0
    )

    return dataset


def freq_phases_tracking_generator(
    freq_breakpoints,
    time_interval,
    n_samples_interval,
    n_seq,
    y_delta,
    x_excluded,
    x_delta,
    func_interp,
):

    x_rand, y_rand = randomize(
        y_breakpoints=freq_breakpoints,
        n_seq=n_seq,
        y_delta=y_delta,
        x_excluded=x_excluded,
        x_delta=x_delta,
        n_samples_interval=n_samples_interval,
    )
    frequencies = data_interpolation(x_rand, y_rand, func_interp=func_interp)
    dataset = add_derivation(
        data=frequencies, n_derived=1, func_derivation=finite_diff, dt=10.0
    )
    phases = compute_phase(
        dataset[:, 0, :],
        dt=n_samples_interval / ((len(freq_breakpoints) - 1) * time_interval),
    )

    return phases
