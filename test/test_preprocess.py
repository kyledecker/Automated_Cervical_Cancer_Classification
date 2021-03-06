import os
import sys
sys.path.insert(0, os.path.abspath('./src/'))


def make_test_rgb(saveflag=True, filename='./test/test_image.tif'):
    """
    generate known RGB pixel array and output TIFF file

    :param saveflag: flag to save output TIFF file, default True
    :param filename: path and filename of saved TFF test file
    :return: RGB pixel array
    """
    from PIL import Image
    import numpy as np

    w, h = 10, 10
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[5, 5] = (100., 0., 0.)
    rgb[1, 1] = (100., 0., 100.)
    rgb[9, :] = 255.*np.ones((1, w, 3))

    if saveflag:
        img = Image.fromarray(rgb, 'RGB')
        img.save(filename)

    rgb = rgb.astype('float')
    return rgb


def test_read_tiff():
    from preprocess import read_tiff
    import numpy as np

    testfile = './test/test_image.tif'

    # compare rgb values loaded from test file with known pixel values
    expected = make_test_rgb(saveflag=True, filename=testfile)
    actual = read_tiff(filename=testfile)

    assert np.array_equal(expected, actual)

    # remove test TIFF file when finished
    os.remove(testfile)


def test_extract_hist():
    from preprocess import extract_hist
    import numpy as np

    pix_array = np.zeros((2, 2))
    pix_array[1, 1] = 255
    pix_array[0, 1] = 10
    pix_array[0, 0] = 10
    pix_array[1, 0] = np.nan
    actual = extract_hist(pix_array)

    expected = np.zeros(256)
    expected[255] = 1
    expected[10] = 2

    assert np.array_equal(actual, expected)


def test_nan_background():
    from preprocess import nan_background
    import numpy as np

    rgb = np.zeros((2, 2, 3))
    rgb[1, 1, 1] = 255

    actual = nan_background(rgb)
    expected = np.nan*np.empty((2, 2, 3))

    expected[1, 1, :] = (0, 255, 0)
    assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08,
                       equal_nan=True)


def test_nan_upper_bound():
    from preprocess import nan_upper_bound
    import numpy as np

    rgb = 254*np.ones((2, 2, 3))
    rgb[1, 1, :] = (0, 255, 0)

    actual = nan_upper_bound(rgb, (250, 250, 250))
    expected = np.nan*np.empty((2, 2, 3))

    expected[1, 1, :] = (0, 255, 0)
    assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08,
                       equal_nan=True)


def test_nan_yellow_pixels():
    from preprocess import nan_yellow_pixels
    import numpy as np

    rgb = np.ones((10, 10, 3))
    rgb[1, 1, :] = [255, 255, 40]

    rgb = nan_yellow_pixels(rgb)

    assert np.isnan(rgb[1, 1, 1])
    assert np.isfinite(rgb[0, 0, 2])


def test_rgb_histogram():
    from preprocess import rgb_histogram, rgb_preprocess
    import numpy as np

    rgb = make_test_rgb(saveflag=False)
    rgb = rgb_preprocess(rgb, exclude_bg=True, upper_lim=(200, 200, 200))
    rh, gh, bh = rgb_histogram(rgb, process=False)
    expected_rh = np.zeros(256)
    expected_rh[100] = 2
    expected_gh = np.zeros(256)
    expected_gh[0] = 2
    expected_bh = np.zeros(256)
    expected_bh[100] = 1
    expected_bh[0] = 1

    assert np.array_equal(rh, expected_rh)
    assert np.array_equal(gh, expected_gh)
    assert np.array_equal(bh, expected_bh)

    rh, gh, bh = rgb_histogram(rgb, process=True)
    expected_rh = np.zeros(256)
    expected_rh[100] = 1
    expected_gh = np.zeros(256)
    expected_gh[0] = 1
    expected_bh = np.zeros(256)
    expected_bh[100] = 0.5
    expected_bh[0] = 0.5

    assert np.array_equal(rh, expected_rh)
    assert np.array_equal(gh, expected_gh)
    assert np.array_equal(bh, expected_bh)
