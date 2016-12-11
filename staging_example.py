import os
import sys
import logging
import numpy as np
sys.path.insert(0, os.path.abspath('./src/'))


if __name__ == "__main__":
    from preprocess import *
    from accessory import color_nans

    preproc = True

    data_path = os.getcwd() + '/TrainingData/'
    example_files = os.listdir(data_path)

    for file in example_files:
        rgb = read_tiff(filename=data_path+file)

        if preproc:
            rgb = rgb_preprocess(rgb, exclude_bg=True)
            color_nans(rgb, [0, 0, 0])

            rgb = rgb_preprocess(rgb, exclude_bg=False,
                                 upper_lim=(0, 0, 240))
            color_nans(rgb)

        rgb = nan_yellow_pixels(rgb)
        color_nans(rgb, color=[255, 255, 0], verb=True)
