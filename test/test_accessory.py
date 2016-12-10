import os
import sys
sys.path.insert(0, os.path.abspath('./src/'))


def test_get_iterable():
    from accessory import get_iterable
    import numpy as np

    vec = []
    actual = [ii for ii in get_iterable(vec)]
    expected = []

    assert np.array_equal(actual, expected)

    vec = 4
    actual = [ii for ii in get_iterable(vec)]
    expected = [vec]

    assert np.array_equal(actual, expected)

    vec = (0, 5)
    actual = [ii for ii in get_iterable(vec)]
    expected = vec

    assert np.array_equal(actual, expected)
