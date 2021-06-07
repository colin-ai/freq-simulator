import numpy as np


def to_csv(X, filename, **kwargs):
    np.savetxt(filename, X, delimiter=";", **kwargs)


def read_csv(filename):
    return np.loadtxt(filename, delimiter=";")