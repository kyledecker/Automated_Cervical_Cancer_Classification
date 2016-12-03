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
    pix_array = np.zeros((h, w, 3), dtype=np.uint8)
    pix_array[:, 6:, 1] = 100.*np.ones((h, w-6))
    pix_array[5, 5] = (100., 0., 0.)
    pix_array[1, 1] = (100., 0., 100.)
    pix_array[9, :] = 255.*np.ones((1, w, 3))

    if saveflag:
        img = Image.fromarray(pix_array, 'RGB')
        img.save(filename)

    return pix_array


def test_read_tiff():
    from preprocess import read_tiff
    import numpy as np

    testfile = './test/test_image.tif'

    # compare rgb values loaded from test file with known pixel values
    pix_array = make_test_rgb(saveflag=True, filename=testfile)
    rgb = read_tiff(filename=testfile)
    assert np.array_equal(rgb, pix_array)


def test_extract_hist():
    from preprocess import extract_hist
    import numpy as np

    pix_array = np.zeros((2, 2))
    pix_array[1, 1] = 255
    pix_array[0, 1] = 10
    pix_array[0, 0] = 10
    pix_array[1, 0] = np.nan
    hist = extract_hist(pix_array)

    expected = np.zeros(256)
    expected[255] = 1
    expected[10] = 2

    assert np.array_equal(hist, expected)
