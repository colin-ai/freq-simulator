# Vehicle position breakpoints
from simulators.common.generator import (
    add_derivation,
    data_interpolation,
    finite_diff,
    linear_interp,
    randomize_breakpoints,
)
from simulators.freq_tracking.model import compute_phase
from simulators.freq_tracking.viz import plot_signal, plot_freq_signal, get_x_mesh
import numpy as np

# Parameters
np.random.RandomState(12)
freq_breakpoints = (3, 3, 10, 10)
n_points_interval = 300  # number of points between two breakpoints
time_interval = 1  # seconds
n_seq = 1

# Simulator
x_rand, y_rand = randomize_breakpoints(
    y_breakpoints=freq_breakpoints,
    n_seq=n_seq,
    y_delta=2,
    x_excluded=1,
    x_delta=2,
    n_points_interval=n_points_interval,
)
frequencies = data_interpolation(x_rand, y_rand, func_interp=linear_interp)
dataset = add_derivation(
    data=frequencies, n_derived=1, func_derivation=finite_diff, dt=10.0
)
phases = compute_phase(
    dataset[:, 0, :],
    dt=n_points_interval / ((len(freq_breakpoints) - 1) * time_interval),
)

n_samples = (len(freq_breakpoints) - 1) * n_points_interval
x_linspace = np.linspace(start=0, stop=3, num=n_samples)
plot_freq_signal(x_linspace, frequencies[0, 0], phases[0])
# to_csv(data, f"./datasets/freq_tracking_linear_{n_seq}_seq_{step_breakpoint}_points.csv")

# plot_dataset_freq(
#     dataset,
#     n_rows=2,
#     n_cols=2,
# )
