import numpy as np

from sysipy.common.data_handler import (
    add_derivation,
    finite_diff,
)
from sysipy.common.interpolator import interpolate
from sysipy.common.randomizer import randomize


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
        to_derive=frequencies, n_derived=1, func_derivation=finite_diff, dt=10.0
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
        to_derive=frequencies, n_derived=1, func_derivation=finite_diff, dt=10.0
    )
    phases = compute_phase(
        dataset[:, 0, :],
        dt=n_samples_interval / ((len(freq_breakpoints) - 1) * time_interval),
    )

    return phases
