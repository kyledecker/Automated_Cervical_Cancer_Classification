import os
import sys
sys.path.insert(0, os.path.abspath('./src/'))


def test_calc_mode():
    from feature_extraction import calc_mode
    import numpy as np

    hist = np.zeros(256)
    hist[5] = 0.4
    hist[20] = 0.5
    hist[100] = 0.2

    actual = calc_mode(hist)
    expected = 20

    assert actual == expected