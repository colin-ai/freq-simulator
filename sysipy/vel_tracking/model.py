import numpy as np

from sysipy.common.data_handler import add_derivation, finite_diff
from sysipy.common.interpolator import interpolate
from sysipy.common.randomizer import randomize


def vel_tracking_generator(
    pos_breakpoints,
    n_seq,
    y_delta,
    x_excluded,
    x_delta,
    n_samples_interval,
    func_interp,
):  # sourcery skip: inline-immediately-returned-variable

    N_DERIVED = 2
    x_rand, y_rand = randomize(
        y_breakpoints=pos_breakpoints,
        n_seq=n_seq,
        y_delta=y_delta,
        x_excluded=x_excluded,
        x_delta=x_delta,
        n_samples_interval=n_samples_interval,
    )
    positions = interpolate(x_rand, y_rand, func_interp=func_interp)
    dataset = add_derivation(
        to_derive=positions, n_derived=N_DERIVED, func_derivation=finite_diff, dt=10.0
    )
    return dataset