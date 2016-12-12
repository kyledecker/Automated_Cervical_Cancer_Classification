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


def test_rgbstring2index():
    from accessory import rgbstring2index
    import numpy as np

    actual = rgbstring2index('rgb')
    expected = [0, 1, 2]

    assert np.array_equal(actual, expected)

    actual = rgbstring2index('rb')
    expected = [0, 2]

    assert np.array_equal(actual, expected)

    actual = rgbstring2index('b')
    expected = [2]

    assert np.array_equal(actual, expected)


def test_color_nans():
    from accessory import color_nans
    import numpy as np

    rgb = np.nan*np.ones((10, 10, 3))
    actual = color_nans(rgb, [0, 0, 0])
    expected = np.zeros((10, 10, 3))

    assert np.array_equal(actual, expected)


def test_percent_color():
    from accessory import percent_color
    import numpy as np

    rgb = np.ones((10, 10, 3))
    rgb[1, 1, :] = [25, 25, 25]
    rgb[1, 2, :] = [0, 0, 0]
    actual = percent_color(rgb, [25, 25, 25])
    expected = 100/99

    assert np.array_equal(actual, expected)


def test_create_dir():
    from accessory import create_dir

    filepath = './test_folder/file.png'
    filedir = './test_folder/'
    create_dir(filepath)
    assert os.path.exists(filedir)
    os.rmdir(filedir)
