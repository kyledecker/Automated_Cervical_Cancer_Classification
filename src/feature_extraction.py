import sys
import logging


def calc_mode(hist, omit=[]):
    """
    calculate mode from histogram

    :param hist: histogram values for 0-255
    :param omit: pixel values to omit from calculation
    :return: mode
    """
    import numpy as np
    from accessory import get_iterable

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

    for omit_idx in get_iterable(omit):
        hist[omit_idx] = 0

    mode = np.argmax(hist)

    return mode


def otsu_threshold(img, omit=[], verb=False):
    """
    calculate the global otsu's threshold for image

    :param img: 2D array of pixel values
    :param omit: pixel values to omit from calculation
    :param verb: verbose mode to display threshold image
    :return: threshold (float)
    """
    from skimage.filters import threshold_otsu
    import numpy as np
    from accessory import get_iterable

    if np.ndim(img) > 2:
        msg = 'ERROR [otsu_threshold] Input image must be 2D grayscale (not ' \
              'RGB).'
        logging.error(msg)
        print(msg)
        sys.exit()

    # set omitted pixel values as 0
    for omit_idx in get_iterable(omit):
        img[img == omit_idx] = np.nan

    img[np.isnan(img)] = 0
    threshold_global_otsu = threshold_otsu(img)

    if verb:
        import matplotlib.pyplot as plt
        global_otsu = img >= threshold_global_otsu
        plt.imshow(global_otsu, cmap=plt.cm.gray)
        plt.show()

    return threshold_global_otsu


def calc_median(img, omit=[]):
    """
    calculate median pixel value of image

    :param img: 2D array of pixel values
    :param omit: pixel values to omit from calculation
    :return: median
    """
    import numpy as np
    from accessory import get_iterable

    if np.ndim(img) > 2:
        msg = 'ERROR [calc_median] Input image must be 2D grayscale (not ' \
              'RGB).'
        logging.error(msg)
        print(msg)
        sys.exit()

    for omit_idx in get_iterable(omit):
        img[img == omit_idx] = np.nan

    pixels = np.ravel(img)
    pixels = pixels[np.isfinite(pixels)]
    median = np.median(pixels)

    return median


def calc_variance(img, omit=[]):
    """
    calculate variance of pixel values in image

    :param img: 2D array of pixel values
    :param omit: pixel values to omit from calculation
    :return: variance
    """
    import numpy as np
    from accessory import get_iterable

    if np.ndim(img) > 2:
        msg = 'ERROR [calc_variance] Input image must be 2D grayscale (not ' \
              'RGB).'
        logging.error(msg)
        print(msg)
        sys.exit()

    for omit_idx in get_iterable(omit):
        img[img == omit_idx] = np.nan

    pixels = np.ravel(img)
    pixels = pixels[np.isfinite(pixels)]
    variance = np.std(pixels)

    return variance


def calc_pct_yellow(rgb):
    """
    calculate percentage of yellow pixels (excluding NaN and black pixels)

    :param rgb: RGB pixel array
    :return: percent of color in image
    """
    import numpy as np
    from accessory import color_nans, percent_color
    from preprocess import nan_yellow_pixels

    y_label = [255, 255, 0]
    recolored_rgb = np.array(rgb)

    recolored_rgb = color_nans(recolored_rgb, [0, 0, 0])
    recolored_rgb = nan_yellow_pixels(recolored_rgb)
    recolored_rgb = color_nans(recolored_rgb, color=y_label)

    pct = percent_color(recolored_rgb, y_label)

    return pct


def extract_features(rgb, median_ch='', variance_ch='',
                     mode_ch='', otsu_ch='',
                     omit=[], verb=False):
    """
    extract specific image features from rgb pixel array

    :param rgb: RGB pixel array
    :param median_ch: color channels to extract median feature
    :param variance_ch: color channels to extract variance feature
    :param mode_ch: color channels to extract mode feature
    :param otsu_ch: color channels to extract Otsu threshold
    :param omit: pixel values to omit from feature extraction
    :param verb: enable verbose mode to output intermediate figures
    :return: feature array (np.array)
    """
    from preprocess import rgb_histogram
    from accessory import rgbstring2index
    import numpy as np

    rh, gh, bh = rgb_histogram(rgb, verb=verb, omit=omit)
    hists = (rh, gh, bh)

    median_idx = rgbstring2index(median_ch)
    variance_idx = rgbstring2index(variance_ch)
    mode_idx = rgbstring2index(mode_ch)
    otsu_idx = rgbstring2index(otsu_ch)

    try:
        median_feats = [calc_median(rgb[:, :, ii], omit) for ii in median_idx]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for median ' \
              'feature out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        median_feats = []

    try:
        variance_feats = [calc_variance(rgb[:, :, ii], omit) for ii in
                          variance_idx]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for var ' \
              'feature out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        variance_feats = []

    try:
        mode_feats = [calc_mode(hists[ii], omit) for ii in mode_idx]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for mode ' \
              'feature out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        mode_feats = []

    try:
        otsu_feats = [otsu_threshold(rgb[:, :, ii], omit, verb=verb) for
                      ii in otsu_idx]
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
