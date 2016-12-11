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

    data = [ii for ii in range(0, 3)]

    actual = calc_median(data)
    expected = 1.

    assert actual == expected


def test_calc_variance():
    from feature_extraction import calc_variance
    import numpy as np

    data = [ii for ii in range(0, 4)]

    actual = calc_variance(data, omit=(0,))
    expected = np.std([1, 2, 3])

    assert actual == expected


def test_extract_features():
    from feature_extraction import extract_features
    import numpy as np

    r = 5*np.ones([3, 3])
    r[1, 1] = 0
    g = np.zeros([3, 3])
    b = [[ii for ii in range(0, 3)] for _ in range(0, 3)]

    img = np.zeros([3, 3, 3])
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    actual = extract_features(img, 'gb', 'r', 'rg', pct_yellow=True)
    expected = [np.median(g), np.median(b), np.std(r), 5, 0, 0]

    assert np.array_equal(actual, expected)

    actual = extract_features(img, 'r', 'r', 'r', omit=0)
    expected = [5, 0, 5]

    assert np.array_equal(actual, expected)


def test_calc_pct_yellow():
    from feature_extraction import calc_pct_yellow
    import numpy as np

    rgb = np.ones((10, 10, 3))
    rgb[1, 1, :] = [255, 255, 40]
    rgb[0, 1, :] = [0, 0, 0]
    rgb[0, 0, :] = [np.nan, np.nan, np.nan]

    actual = calc_pct_yellow(rgb)
    expected = 100/98

    assert actual == expected
