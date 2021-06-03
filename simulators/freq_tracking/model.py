import numpy as np


def compute_phase(frequencies, dt):
    return 0.5 * np.pi * np.cumsum(frequencies, axis=1) / dt