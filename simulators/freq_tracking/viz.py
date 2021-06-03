from random import choice

import numpy as np
import matplotlib.pyplot as plt

from simulators.common.viz import csv_to_dataset


# UTILS

def get_x_mesh(start_time, freq_breakpoints, n_points_interval, time_interval):
    n_breakpoints = len(freq_breakpoints)
    stop_time = (n_breakpoints - 1) * time_interval
    n_samples = (n_breakpoints - 1) * n_points_interval
    return np.linspace(start_time, stop_time, n_samples)


# PLOTS HELPER

def plot_freq_signal(x, freq, phases):
    fig, ax = plt.subplots(1)
    ax2 = ax.twinx()
    ax.plot(x, freq, "g--")
    ax2.plot(x, np.sin(phases))
    fig.show()


def plot_signal(phases):
    fig, ax = plt.subplots(1)
    ax.plot(np.sin(phases))
    fig.show()

def plot_dataset_freq(filename, n_rows=4, n_cols=4):
    # TODO : adapt x-axis tick

    dataset = csv_to_dataset(filename, channel_dim=3)

    n_sample = dataset.shape[0]
    n_axes = min(n_rows * n_cols, n_sample)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_rows, 5 * n_cols))
    for i in range(n_axes):
        i_random = choice(range(0, n_sample - 1, 2))
        freq = dataset[i_random, 0, :]
        dfreq = dataset[i_random, 1, :]

        ax = axes[i // n_rows, i % n_cols]
        ax.plot(freq, "g-", label="Frequence")
        ax.set(xlabel="step", ylabel="value")
        ax_twin = ax.twinx()
        ax_twin.plot(dfreq, "m--", label="∂freq")
        # ax_twin.plot(acc, "r:", label="acceleration")
        # Add both axes's label on same box (stackoverflow.com/a/10129461)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines + lines2, labels + labels2, loc=0)

    fig.show()