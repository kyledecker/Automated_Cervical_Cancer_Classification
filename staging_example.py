import os
import sys
import logging
import numpy as np
sys.path.insert(0, os.path.abspath('./src/'))


if __name__ == "__main__":
    from preprocess import *

    preproc = True

    data_path = os.getcwd() + '/TrainingData/'
    example_files = os.listdir(data_path)

    for file in example_files:
        rgb = read_tiff(filename=data_path+file)

        if preproc:
            rgb = rgb_preprocess(rgb, verb=False, exclude_bg=True,
                                 upper_lim=(0, 0, 100))

        rgb = remove_yellow_pixels(rgb, tol=5, verb=True)
