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


def _subplots_dataset(filename, n_rows=4, n_cols=4):
    dataset = csv_to_dataset(filename, channel_dim=3)

    n_sample = dataset.shape[0]
    n_axes = min(n_rows * n_cols, n_sample)

    fig, axes = subplots(n_rows, n_cols, figsize=(5 * n_rows, 5 * n_cols))
    for i in range(n_axes):
        i_random = choice(range(0, n_sample - 1, 2))

