# Vehicle position pos_breakpoints
from sysipy.common.data_handler import (
    add_derivation,
    data_interpolation,
    finite_diff,
    linear_interp,
    to_csv,
)
from sysipy.common.randomizer import randomize
from sysipy.freq_tracking.model import compute_phase
from sysipy.freq_tracking.viz import plot_signal, plot_freq_signal, get_x_mesh
import numpy as np

# Parameters
np.random.RandomState(12)
freq_breakpoints = (5, 5, 15, 15)
n_points_interval = 200  # number of points between two pos_breakpoints
time_interval = 1  # seconds
n_seq = 100

# Simulator
x_rand, y_rand = randomize(
    y_breakpoints=freq_breakpoints,
    n_seq=n_seq,
    y_delta=5,
    x_excluded=1,
    x_delta=5,
    n_samples_interval=n_points_interval,
)
frequencies = data_interpolation(x_rand, y_rand, func_interp=linear_interp)
dataset = add_derivation(
    to_derive=frequencies, n_derived=1, func_derivation=finite_diff, dt=10.0
)

n_samples = (len(freq_breakpoints) - 1) * n_points_interval
to_csv(
    dataset.reshape(n_seq * 2, n_samples),
    f"./datasets/freq_tracking/freq_tracking_linear_{n_seq}_seq_{n_samples}_samples.csv",
)

# plot_dataset_freq(
#     dataset,
#     n_rows=2,
#     n_cols=2,
# )
