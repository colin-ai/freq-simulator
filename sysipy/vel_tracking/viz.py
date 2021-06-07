from random import choice

from matplotlib.pyplot import subplots

from sysipy.common.viz import csv_to_dataset


def subplots_dataset_tracking(filename, n_rows=4, n_cols=4):
    # TODO : adapt x-axis tick

    dataset = csv_to_dataset(filename, channel_dim=3)

    n_sample = dataset.shape[0]
    n_axes = min(n_rows * n_cols, n_sample)

    fig, axes = subplots(n_rows, n_cols, figsize=(5 * n_rows, 5 * n_cols))
    for i in range(n_axes):
        i_random = choice(range(0, n_sample - 1, 2))
        pos = dataset[i_random, 0, :]
        vel = dataset[i_random, 1, :]
        acc = dataset[i_random, 2, :]

        ax = axes[i // n_rows, i % n_cols]
        ax.plot(pos, "g-", label="position")
        ax.set(xlabel="step", ylabel="value")
        ax_twin = ax.twinx()
        ax_twin.plot(vel, "m--", label="velocity")
        ax_twin.plot(acc, "r:", label="acceleration")
        # Add both axes's label on same box (stackoverflow.com/a/10129461)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines + lines2, labels + labels2, loc=0)
    fig.show()
