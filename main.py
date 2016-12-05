import os
import sys
import logging
sys.path.insert(0, os.path.abspath('./src/'))


if __name__ == "__main__":
    from preprocess import read_tiff, rgb_histogram, rgb_preprocess
    from feature_extraction import *

    verb = False

    rgb = read_tiff(filename='./test/ExampleAbnormalCervix.tif')
    rgb = rgb_preprocess(rgb, verb=verb, exclude_bg=True, upper_lim=(0,  0,
                                                                     240))
    rh, gh, bh = rgb_histogram(rgb, verb=verb, omit=(0, 255))

    green_otsu = otsu_threshold(rgb[:, :, 1], verb=verb)
    blue_mode = calc_mode(bh)

    msg = "G channel Otsu's threshold = %d" % green_otsu
    logging.info(msg)
    print(msg)

    msg = "B channel mode = %d" % blue_mode
    logging.info(msg)
    print(msg)
