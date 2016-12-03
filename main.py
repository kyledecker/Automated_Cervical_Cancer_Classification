import os
import sys
import logging
sys.path.insert(0, os.path.abspath('./src/'))

if __name__ == "__main__":
    from preprocess import read_tiff, extract_hist

    ra, ga, ba = read_tiff(filename='./test/ExampleAbnormalCervix.tif')

    extract_hist(ra, verb=True)
