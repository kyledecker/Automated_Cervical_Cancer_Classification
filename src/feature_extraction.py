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


def calc_variance(hist):
    """
    calculate variance from histogram

    :param hist: histogram values for 0-255
    :return: variance
    """
    import numpy as np

    if np.ndim(hist) > 1:
        msg = 'ERROR [calc_variance] Input must be 1D array of histogram ' \
              'frequency values.'
        logging.error(msg)
        print(msg)
        sys.exit()
    if np.size(hist) > 256:
        msg = 'ERROR [calc_variance] Input histogram data has > 256 bins.'
        logging.error(msg)
        print(msg)
        sys.exit()

    variance = np.std(hist)

    return variance


def extract_features(rgb, median_dims=None, variance_dims=None,
                     mode_dims=None, otsu_dims=None,
                     hist_omit=[], verb=False):
    """
    extract specific image features from rgb pixel array

    :param rgb: RGB pixel array
    :param median_dims: color channel indices to extract median feature
    :param variance_dims: color channel indices to extract variance feature
    :param mode_dims: color channel indices to extract mode feature
    :param otsu_dims: color channel indices to extract Otsu threshold
    :param hist_omit: bins to omit from histogram feature extractions
    :param verb: enable verbose mode to output intermediate figures
    :return: feature array (np.array)
    """
    from preprocess import rgb_histogram
    import numpy as np

    rh, gh, bh = rgb_histogram(rgb, verb=verb, omit=hist_omit)
    hists = (rh, gh, bh)

    try:
        median_feats = [calc_median(hists[ii]) for ii in median_dims]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for median ' \
              'feature out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        median_feats = []
    try:
        variance_feats = [calc_variance(hists[ii]) for ii in variance_dims]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for variance ' \
              'feature out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        variance_feats = []

    try:
        mode_feats = [calc_mode(hists[ii]) for ii in mode_dims]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for mode ' \
              'feature out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        mode_feats = []

    try:
        otsu_feats = [otsu_threshold(rgb[:, :, ii]) for ii in otsu_dims]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for Otsu ' \
              'threshold out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        otsu_feats = []

    features = median_feats
    features = np.append(features, variance_feats)
    features = np.append(features, mode_feats)
    features = np.append(features, otsu_feats)

    return features
