import sys
import logging


def calc_mode(hist):
    """
    calculate mode from histogram

    :param hist: histogram values for 0-255
    :return: mode
    """
    import numpy as np

    mode = np.argmax(hist)

    return mode
