import os
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

    # find the index of the histogram maximum
    mode = np.argmax(hist)

    return mode


def otsu_threshold(img, omit=[], verb=False, outfile='./otsu_img.png'):
    """
    calculate the global otsu's threshold for image

    :param img: 2D array of pixel values
    :param omit: pixel values to omit from calculation
    :param verb: verbose mode to display threshold image
    :param outfile: file to output otsu image if verbose
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
        from accessory import create_dir
        global_otsu = img >= threshold_global_otsu
        plt.imshow(global_otsu, cmap=plt.cm.gray)

        msg = '[otsu_threshold] Saving Otsu threshold image: %s' % outfile
        logging.info(msg)
        print(msg)

        create_dir(outfile)
        plt.savefig(outfile)
        plt.clf()

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

    # collapse image into array
    pixels = np.ravel(np.array(img))
    pixels = pixels.astype('float')

    # omit specified pixel values from median calculation
    for omit_idx in get_iterable(omit):
        pixels[pixels == omit_idx] = np.nan

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

    # collapse image into array
    pixels = np.ravel(np.array(img))
    pixels = pixels.astype('float')

    # omit specified pixel values from variance calculation
    for omit_idx in get_iterable(omit):
        pixels[pixels == omit_idx] = np.nan

    pixels = pixels[np.isfinite(pixels)]
    variance = np.std(pixels)

    return variance


def calc_pct_yellow(rgb, verb=False, outfile='./yellow.png'):
    """
    calculate percentage of yellow pixels (excluding NaN and black pixels)

    :param rgb: RGB pixel array
    :param verb: verbose mode to display image yellow pixels highlighted
    :param outfile: file to output highlighted image if verbose
    :return: percent of color in image
    """
    import numpy as np
    from accessory import color_nans, percent_color
    from preprocess import nan_yellow_pixels

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [calc_pct_yellow]] Input array dimensions ' + \
              str(rgb.shape) + ' incompatible with expected ' \
                               'N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [calc_pct_yellow] Input RGB array must contain element ' \
              'values between 0 and 255. Actual range: [%.1f, %.1f]' % \
              (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    y_label = [0, 255, 0]
    recolored_rgb = np.array(rgb)

    # assign all image NaNs to black pixels
    recolored_rgb = color_nans(recolored_rgb, [0, 0, 0])

    # assign NaN to all yellow pixels and recolor based on desired label
    recolored_rgb = nan_yellow_pixels(recolored_rgb)
    recolored_rgb = color_nans(recolored_rgb, color=y_label)

    if verb:
        from accessory import save_rgb
        from accessory import create_dir
        msg = '[calc_pct_yellow] Saving yellow labeled image: %s' % outfile
        logging.info(msg)
        print(msg)

        create_dir(outfile)
        save_rgb(recolored_rgb, outfile)

    # calculate the percentage of labeled yellow pixels
    pct = percent_color(recolored_rgb, y_label)

    return pct


def extract_features(rgb, median_ch='', variance_ch='',
                     mode_ch='', otsu_ch='', pct_yellow=False,
                     omit=[], verb=False, outdir='./outputs/'):
    """
    extract specific image features from rgb pixel array

    :param rgb: RGB pixel array
    :param median_ch: color channels to extract median feature
    :param variance_ch: color channels to extract variance feature
    :param mode_ch: color channels to extract mode feature
    :param otsu_ch: color channels to extract Otsu threshold
    :param pct_yellow: use percent yellow pixel feature
    :param omit: pixel values to omit from feature extraction
    :param verb: enable verbose mode to output intermediate figures
    :param outdir: directory to store all outputs if verbose
    :return: feature array (np.array)
    """
    from preprocess import rgb_histogram
    from accessory import rgbstring2index
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [extract_features]] Input array dimensions ' + \
              str(rgb.shape) + ' incompatible with expected ' \
                               'N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [extract_features] Input RGB array must contain ' \
              'element values between 0 and 255. Actual range: ' \
              '[%.1f, %.1f]' % (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    # compute RGB pixel histograms
    outfile = os.path.join(outdir, 'rgb_hist.png')
    rh, gh, bh = rgb_histogram(rgb, verb=verb, omit=omit, outfile=outfile)
    hists = (rh, gh, bh)

    # parse desired RGB features from string inputs
    median_idx = rgbstring2index(median_ch)
    variance_idx = rgbstring2index(variance_ch)
    mode_idx = rgbstring2index(mode_ch)
    otsu_idx = rgbstring2index(otsu_ch)

    # compute desired features
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
        outfiles = {ii: os.path.join(outdir, 'otsu'+str(ii)+'.png')
                    for ii in otsu_idx}
        otsu_feats = [otsu_threshold(rgb[:, :, ii], omit, verb=verb,
                                     outfile=outfiles[ii]) for ii in otsu_idx]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for Otsu ' \
              'threshold out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        otsu_feats = []

    if pct_yellow:
        outfile = os.path.join(outdir, 'ypixels.png')
        ypct_feat = calc_pct_yellow(rgb, verb=verb, outfile=outfile)
    else:
        ypct_feat = []

    # store all features into a single array
    features = median_feats
    features = np.append(features, variance_feats)
    features = np.append(features, mode_feats)
    features = np.append(features, otsu_feats)
    features = np.append(features, ypct_feat)

    return features
