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


def test_calc_median():
    from feature_extraction import calc_median
    import numpy as np

    hist = np.zeros(5)
    hist[0] = 0.4
    hist[1] = 0.5
    hist[2] = 0.2
    hist[3] = 0.9
    hist[4] = 0.1

    actual = calc_median(hist)
    expected = 0

    assert actual == expected
