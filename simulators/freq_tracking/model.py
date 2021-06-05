import numpy as np

from simulators.common.data_handler import (
    add_derivation,
    finite_diff,
)
from simulators.common.interpolator import interpolate
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
    """
    phase = φ(t) = 2π * Int_0_t(f(t) dt)
    freq = f(t) = 1/2π * ∂φ(t)/∂t
    pulsation = "angular velocity" = ω(t) = 2π*f(t) = ∂φ(t)/∂t

    Args:
        freq_breakpoints:
        n_seq:
        y_delta:
        x_excluded:
        x_delta:
        n_samples_interval:
        func_interp:

    Returns:

    """
    x_rand, y_rand = randomize(
        y_breakpoints=freq_breakpoints,
        n_seq=n_seq,
        y_delta=y_delta,
        x_excluded=x_excluded,
        x_delta=x_delta,
        n_samples_interval=n_samples_interval,
    )
    frequencies = interpolate(x_rand, y_rand, func_interp=func_interp)
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
):  # sourcery skip: inline-immediately-returned-variable
    """
    phase = φ(t) = 2π * Int_0_t(f(t) dt)
    freq = f(t) = 1/2π * ∂φ(t)/∂t
    pulsation = "angular velocity" = ω(t) = 2π*f(t) = ∂φ(t)/∂t

    Args:
        freq_breakpoints:
        time_interval:
        n_samples_interval:
        n_seq:
        y_delta:
        x_excluded:
        x_delta:
        func_interp:

    Returns:

    """
    x_rand, y_rand = randomize(
        y_breakpoints=freq_breakpoints,
        n_seq=n_seq,
        y_delta=y_delta,
        x_excluded=x_excluded,
        x_delta=x_delta,
        n_samples_interval=n_samples_interval,
    )
    frequencies = interpolate(x_rand, y_rand, func_interp=func_interp)
    dataset = add_derivation(
        data=frequencies, n_derived=1, func_derivation=finite_diff, dt=10.0
    )
    phases = compute_phase(
        dataset[:, 0, :],
        dt=n_samples_interval / ((len(freq_breakpoints) - 1) * time_interval),
    )

    return phases


def seq_generator_freq(input):  # sourcery skip: inline-immediately-returned-variable
    """
    phase = φ(t) = 2π * Int_0_t(f(t) dt)
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
