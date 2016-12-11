import os
import sys
import logging
import numpy as np
sys.path.insert(0, os.path.abspath('./src/'))


if __name__ == "__main__":
    from preprocess import *
    from feature_extraction import calc_pct_yellow
    from accessory import color_nans, percent_color

    preproc = True

    data_path = os.getcwd() + '/TrainingData/'
    example_files = os.listdir(data_path)

    for file in example_files:
        rgb = read_tiff(filename=data_path+file)

        if preproc:
            rgb = rgb_preprocess(rgb, exclude_bg=False,
                                 upper_lim=(0, 0, 240))

        percent = calc_pct_yellow(rgb)
        print(file + ': %.1f ' % percent)

        """color_nans(rgb)

        rgb = nan_yellow_pixels(rgb)
        color_nans(rgb, color=[255, 255, 0], verb=False)

        percent = percent_color(rgb, [255, 255, 0])
        print(file + ': %.1f ' % percent)"""
