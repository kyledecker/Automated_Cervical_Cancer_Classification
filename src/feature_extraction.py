import sys
import logging


def calc_mode(hist):
    """
    calculate mode from histogram

    :param hist: histogram values for 0-255
    :return: mode
    """
    import numpy as np

    if np.ndim(hist) > 1:
        msg = 'ERROR [calc_mode] Input must be 1D array of histogram ' \
              'frequency values.'
        logging.error(msg)
        print(msg)
        sys.exit()
    if np.size(hist) > 256:
        msg = 'ERROR [calc_mode] Input histogram data has > 256 bins.'
        logging.error(msg)
        print(msg)
        sys.exit()

    mode = np.argmax(hist)

    return mode


def otsu_threshold(img, verb=False):
    """
    calculate the global otsu's threshold for grayscale image

    :param img: 2D array of pixel values
    :param verb: verbose mode to display threshold image
    :return: threshold (float)
    """
    from skimage.filters import threshold_otsu
    import numpy as np

    if np.ndim(img) != 2:
        msg = 'ERROR [otsu_threshold] Input image must be 2D grayscale (not ' \
              'RGB).'
        logging.error(msg)
        print(msg)
        sys.exit()

    # set omitted pixel values as 0
    img[np.isnan(img)] = 0
    threshold_global_otsu = threshold_otsu(img)

    if verb:
        import matplotlib.pyplot as plt
        global_otsu = img >= threshold_global_otsu
        plt.imshow(global_otsu, cmap=plt.cm.gray)
        plt.show()

    return threshold_global_otsu

def calc_median(hist):
    """
    calculate median from histogram

    :param hist: histogram values for 0-255
    :return: median
    """
    import numpy as np

    if np.ndim(hist) > 1:
        msg = 'ERROR [calc_median] Input must be 1D array of histogram ' \
              'frequency values.'
        logging.error(msg)
        print(msg)
        sys.exit()
    if np.size(hist) > 256:
        msg = 'ERROR [calc_median] Input histogram data has > 256 bins.'
        logging.error(msg)
        print(msg)
        sys.exit()

    median = np.argsort(hist)[len(hist)//2]

    return median
