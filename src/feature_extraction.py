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

    otsu_img = np.array(img)
    # set omitted pixel values as 0
    for omit_idx in get_iterable(omit):
        otsu_img[img == omit_idx] = np.nan
        otsu_img[np.isnan(otsu_img)] = 0

    threshold_global_otsu = threshold_otsu(otsu_img)

    if verb:
        import matplotlib.pyplot as plt
        from accessory import create_dir
        global_otsu = otsu_img >= threshold_global_otsu
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
    from accessory import rgbstring2index, write_csv
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
        labels = ['R median', 'G median', 'B median']
        median_labels = [labels[ii] for ii in median_idx]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for median ' \
              'feature out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        median_feats = []
        median_labels = []

    try:
        variance_feats = [calc_variance(rgb[:, :, ii], omit) for ii in
                          variance_idx]
        labels = ['R variance', 'G variance', 'B variance']
        variance_labels = [labels[ii] for ii in variance_idx]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for var ' \
              'feature out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        variance_feats = []
        variance_labels = []

    try:
        mode_feats = [calc_mode(hists[ii], omit) for ii in mode_idx]
        labels = ['R mode', 'G mode', 'B mode']
        mode_labels = [labels[ii] for ii in mode_idx]
    except IndexError:
        msg = 'ERROR [extract_features] Color channel index for mode ' \
              'feature out of bounds (0:R, 1:G, 2:B).'
        logging.error(msg)
        print(msg)
        sys.exit()
    except TypeError:
        mode_feats = []
        mode_labels = []

    try:
        labels = ['R threshold', 'G threshold', 'B threshold']
        otsu_labels = [labels[ii] for ii in otsu_idx]
        outfiles = {ii: os.path.join(outdir, labels[ii][0] + '_otsu.png')
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
        otsu_labels = []

    if pct_yellow:
        outfile = os.path.join(outdir, 'ypixels.png')
        ypct_feat = calc_pct_yellow(rgb, verb=verb, outfile=outfile)
        ypct_label = ['Percent Y']
    else:
        ypct_feat = []
        ypct_label = []

    # store all features into a single array
    features = median_feats
    features = np.append(features, variance_feats)
    features = np.append(features, mode_feats)
    features = np.append(features, otsu_feats)
    features = np.append(features, ypct_feat)

    feature_labels = median_labels
    feature_labels = np.append(feature_labels, variance_labels)
    feature_labels = np.append(feature_labels, mode_labels)
    feature_labels = np.append(feature_labels, otsu_labels)
    feature_labels = np.append(feature_labels, ypct_label)

    if verb:
        outfile = os.path.join(outdir, 'feature_values.csv')
        write_csv(feature_labels, features, outfile)

    return features, feature_labels


def plot_features(features, targets, labels, outfile='features.png'):
    """
    visualize 2D feature space for data set with known targets

    :param features: N x 2 array of features from N data sets
    :param targets: target labels corresponding to each set of features
    :param labels: labels for each feature
    :param outfile: save location of output feature plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from accessory import create_dir

    target_types = np.unique(targets)
    if len(target_types) > 2:
        msg = 'ERROR [plot_features] Function only compatible with 2 targets.'
        logging.error(msg)
        print(msg)
        sys.exit()

    if np.shape(features)[1] > 2:
        msg = 'ERROR [plot_features] Function only compatible with 2 features.'
        logging.error(msg)
        print(msg)
        sys.exit()

    if np.shape(features)[0] != len(targets):
        msg = 'ERROR [plot_features] Mismatch between number of target ' \
              'labels and feature sets.'
        logging.error(msg)
        print(msg)
        sys.exit()

    features0 = features[targets == target_types[0], :]
    features1 = features[targets == target_types[1], :]

    h0 = plt.scatter(features0[:, 0], features0[:, 1], marker='o', c='red',
                     label=target_types[0])
    h1 = plt.scatter(features1[:, 0], features1[:, 1], marker='o',
                     c='blue',
                     label=target_types[1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(handles=[h0, h1], loc=4)
    plt.grid(True)
    plt.axis('tight')

    create_dir(outfile)
    msg = '[plot_features] Feature space plot saved: %s' % outfile
    print(msg)
    plt.savefig(outfile)


def collect_feature_data(filepath, feature_dict,
                         omit, b_cutoff=240,
                         verb=False, outdir='./outputs/'):
    """
    collect feature data from a specified directory

    :param filepath: path to tif file or directory containing tif files
    :param feature_dict: dict of strings specifying color channel for features
    :param omit: pixel values to omit from calculation of features, ex [0, 255]
    :param b_cutoff: blue color channel cutoff for glare removal
    :param verb: verbose mode to save intermediate files and figures
    :param outdir: directory where output files are saved
    :return: feature_array, target_array, feature_labels
    """
    from preprocess import read_tiff, rgb_preprocess
    from feature_extraction import extract_features
    import numpy as np

    msg = 'Data location: %s' % filepath
    logging.info(msg)
    print(msg)

    msg = '\nSELECTED FEATURES:'
    logging.info(msg)
    print(msg)
    msg = 'Color channel median: %s' % feature_dict['med']
    logging.info(msg)
    print(msg)
    msg = 'Color channel variance: %s' % feature_dict['var']
    logging.info(msg)
    print(msg)
    msg = 'Color channel mode: %s' % feature_dict['mode']
    logging.info(msg)
    print(msg)
    msg = 'Color channel Otsu: %s' % feature_dict['otsu']
    logging.info(msg)
    print(msg)
    msg = 'Yellow pixel fraction: %s\n' % feature_dict['ypct']
    logging.info(msg)
    print(msg)

    n_feat = len(feature_dict['med'] + feature_dict['var'] +
                 feature_dict['mode'] + feature_dict['otsu'])
    if feature_dict['ypct']:
        n_feat += 1

    # extract all data files from directory or directly use specified tif
    try:
        all_files = os.listdir(filepath)
        data_files = [f for f in all_files if '.tif' in f]
        data_dir = filepath
    except NotADirectoryError:
        data_dir = os.path.dirname(filepath)
        if data_dir == '.':
            data_dir = ''
        data_files = [os.path.split(filepath)[-1], ]

    n_datasets = len(data_files)

    target_array = np.zeros(n_datasets)
    feature_array = np.zeros((n_datasets, n_feat))

    for i in range(len(data_files)):

        msg = 'Extracting features from ' \
              + data_files[i] + ' (%d/%d)' % (i + 1, len(data_files))
        logging.info(msg)
        print(msg)

        # directory to store outputs for training set
        feat_outdir = os.path.join(outdir, 'feature_data',
                                   os.path.splitext(data_files[i])[0])

        data_file = os.path.join(data_dir, data_files[i])
        rgb = read_tiff(filename=data_file)
        rgb = rgb_preprocess(rgb, exclude_bg=True,
                             upper_lim=(0, 0, b_cutoff))

        features, l = extract_features(rgb,
                                       median_ch=feature_dict['med'],
                                       variance_ch=feature_dict['var'],
                                       mode_ch=feature_dict['mode'],
                                       otsu_ch=feature_dict['otsu'],
                                       pct_yellow=feature_dict['ypct'],
                                       omit=omit,
                                       verb=verb,
                                       outdir=feat_outdir)

        feature_array[i, :] = features

        if 'dys' in data_files[i]:
            target_array[i] = 1
        elif 'heal' in data_files[i]:
            target_array[i] = -1
        else:
            target_array[i] = 0

        msg = 'Target label (1 dysplasia, -1 healthy): %d' % \
              target_array[i]
        logging.debug(msg)

    return feature_array, target_array, l
