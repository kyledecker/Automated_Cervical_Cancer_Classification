import os
import sys
sys.path.insert(0, os.path.abspath('./src/'))


def make_test_tiff(filename='./test/test_image.tif'):
    """
    generate TIFF image from known RGB pixel array

    :param filename: path and filename of saved TFF test file
    :return: pixel array used to generate TIFF
    """
    from PIL import Image
    import numpy as np

    w, h = 10, 10
    pix_array = np.zeros((h, w, 3), dtype=np.uint8)
    for ii in range(0, h):
        for jj in range(0, w):
            pix_array[ii, jj] = [ii * 10, jj * 10, 90]

    img = Image.fromarray(pix_array, 'RGB')
    img.save(filename)

    return pix_array


def test_read_tiff():
    from preprocess import read_tiff
    import numpy as np

    testfile = './test/test_image.tif'

    # compare rgb values loaded from test file with expected pixel values
    pix_array = make_test_tiff(testfile)
    r, g, b = read_tiff(filename=testfile, verb=0)
    rgb = np.transpose((r, g, b), (1, 2, 0))
    assert np.array_equal(rgb, pix_array)


