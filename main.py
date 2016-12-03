import os
import sys
import logging
sys.path.insert(0, os.path.abspath('./src/'))

if __name__ == "__main__":
    from preprocess import read_tiff

    read_tiff(filename='./test/a_image.tif', verb=1)
