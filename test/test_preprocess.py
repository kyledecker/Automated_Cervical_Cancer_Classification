import os
import sys
import logging
sys.path.insert(0, os.path.abspath('./src/'))

def test_read_tiff():
    from preprocess import read_tiff
    from PIL import Image
    import numpy as np

    testfile = './test/test_image.tif'
    w, h = 10, 10
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for ii in range(0, h):
        for jj in range(0, w):
            data[ii, jj] = [ii*10, jj*10, 90]

    img = Image.fromarray(data, 'RGB')
    img.save(testfile)

    r, g, b = read_tiff(filename=testfile, verb=0)

    rgb = np.transpose((r, g, b), (1, 2, 0))
    assert np.array_equal(rgb, data)


