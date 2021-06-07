import numpy as np


def interpolate(x, y, func_interp):
    dataset = _init_dataset(x)
    for i in range(len(x)):
        dataset[i] = func_interp(x[0], y[0])
    return dataset


def negative_value_thresholder(velocity):
    """
    Force all negative position to zero value.
    """
    velocity[velocity < 0] = 0
    return velocity


def linear_interp(xp, yp):
    x = np.arange(0.0, xp[-1])
    velocities = np.interp(x, xp, yp)
    velocities = negative_value_thresholder(velocities)
    return velocities


def polyn_interp(xp, yp):
    x = np.arange(0.0, xp[-1])
    p = np.polyfit(xp, yp, deg=int(len(yp) * 0.7))
    velocities = np.polyval(p, x)
    velocities = negative_value_thresholder(velocities)
    return velocities

def _init_dataset(input):
    n_seq = input.shape[0]
    x_limit = int(input[0, -1])
    return np.zeros([n_seq, 1, x_limit])

