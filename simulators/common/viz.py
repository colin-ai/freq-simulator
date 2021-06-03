from matplotlib.pyplot import subplots
from numpy import loadtxt, ndarray
from numpy.random import choice


def _is_path(filename):
    return isinstance(filename, str)


def _is_array(inp):
    return isinstance(inp, ndarray)


def csv_to_dataset(filename, channel_dim, delimiter=";"):
    if _is_path(filename):
        dataset = loadtxt(filename, delimiter=delimiter)
        dataset = dataset.reshape((-1, channel_dim, dataset.shape[1]))
    elif _is_array(filename):
        dataset = filename
    else:
        raise TypeError("Input type is not recognize.")
    return dataset


def subplots_dataset_tireslip(filename, n_rows=4, n_cols=4):
    # TODO : adapt x-axis tick
    dataset = csv_to_dataset(filename, channel_dim=2)

    n_sample = dataset.shape[0]
    n_axes = min(n_rows * n_cols, n_sample)

    fig, axes = subplots(n_rows, n_cols, figsize=(5 * n_rows, 5 * n_cols))
    for i in range(n_axes):
        i_random = choice(range(0, n_sample - 1, 2))
        v_vehicle = dataset[i_random, 0]
        v_wheel = dataset[i_random, 1]

        ax = axes[i // n_rows, i % n_cols]
        ax.plot(v_vehicle, "g-", label="v_vehicle")
        ax.plot(v_wheel, "m--", label="v_wheel")
        ax.set(xlabel="t [s]", ylabel="v [km/s]")
        ax_twin = ax.twinx()
        ax_twin.plot(v_vehicle - v_wheel, "r:", label="Δv")
        # Add both axes's label on same box (stackoverflow.com/a/10129461)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines + lines2, labels + labels2, loc=0)
    fig.show()


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

