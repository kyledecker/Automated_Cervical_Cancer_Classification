import sys
import logging


def read_tiff(filename, verb=False):
    """
    read in TIFF image from file and output pixel RGB values

    :param filename: TIFF path and filename
    :param verb: verbose mode to show TIFF image, default False
    :return: RGB pixel array (np.array)
    """
    from PIL import Image
    import numpy as np

    # read in tiff image
    try:
        img = Image.open(filename)
        msg = '[read_tiff] Image data loaded from ' + filename
        logging.debug(msg)
    except FileNotFoundError as err:
        msg = 'ERROR [read_tiff] %s is not a valid input tif file: {' \
              '0}'.format(err) % filename
        print(msg)
        logging.error(msg)
        sys.exit()

    if verb:
        print(msg)
        img.show()

    # split image data into RGB channels
    img.getdata()
    try:
        r, g, b = img.split()
    except ValueError as err:
        msg = 'ERROR [read_tiff] RGB color channels not ' \
              'available: {0}'.format(err)
        logging.error(msg)
        print(msg)
        sys.exit()

    # assign RGB channels to separate indices of array
    ra = np.array(r)
    ga = np.array(g)
    ba = np.array(b)

    img_shape = np.shape(ra)
    rgb = np.zeros((img_shape[0], img_shape[1], 3))
    rgb[:, :, 0] = ra
    rgb[:, :, 1] = ga
    rgb[:, :, 2] = ba

    return rgb


def extract_hist(pix_array, verb=False):
    """
    generate histogram for 1D or 2D array of single color channel pixel values

    :param pix_array: 1D or 2D array of 0 to 255 pixel values
    :param verb: verbose mode to show histogram, default False
    :return: histogram values for 0-255 (np.array)
    """
    import numpy as np

    if np.max(pix_array) > 255 or np.min(pix_array) < 0:
        msg = 'ERROR [extract_hist] Input pixel array must contain element ' \
              'values between 0 and 255. Actual range: [%.1f, %.1f]' % \
              (np.min(pix_array), np.max(pix_array))
        print(msg)
        logging.error(msg)
        sys.exit()

    pix_array = pix_array[np.isfinite(pix_array)]
    hist, bin_edges = np.histogram(pix_array, bins=256, range=(0, 255))

    if verb:
        import matplotlib.pyplot as plt
        pix_array = np.ravel(pix_array)
        plt.hist(pix_array, bins=256, range=(0, 255))
        plt.axis('tight')
        plt.show()

    return hist


def nan_background(rgb, verb=False):
    """
    identify background pixels (R=0, B=0, and G=0) and convert to NaN

    :param rgb: RGB pixel array with dimensions: height x width x RGB
    :param verb: verbose mode to show excluded pixels in gray, default False
    :return: RGB pixel array (np.array)
    """
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [nan_background] Input array dimensions ' + \
              str(rgb.shape) + \
              ' incompatible with expected N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [nan_background] Input RGB array must contain element ' \
              'values between 0 and 255. Actual range: [%.1f, %.1f]' % \
              (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    rgb[(rgb[:, :, 0] == 0) &
        (rgb[:, :, 1] == 0) &
        (rgb[:, :, 2] == 0), :] = (np.nan, np.nan, np.nan)

    if verb:
        from accessory import show_rgb
        test_img = np.array(rgb)
        test_img[np.isnan(test_img)] = 100
        show_rgb(test_img)

    return rgb


def nan_upper_bound(rgb, lim=(255, 255, 255), verb=False):
    """
    identify bright pixels above RGB threshold and convert to NaN

    :param rgb: RGB pixel array with dimensions: height x width x RGB
    :param lim: RGB pixel value upper limit, default (255, 255, 255)
    :param verb: verbose mode to show excluded pixels in gray, default False
    :return: RGB pixel array (np.array)
    """
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [nan_upper_bound] Input array dimensions ' + \
              str(rgb.shape) + \
              ' incompatible with expected N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [nan_upper_bound] Input RGB array must contain element ' \
              'values between 0 and 255. Actual range: [%.1f, %.1f]' % \
              (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    rgb[(rgb[:, :, 0] > lim[0]) &
        (rgb[:, :, 1] > lim[1]) &
        (rgb[:, :, 2] > lim[2]), :] = (np.nan, np.nan, np.nan)
    if verb:
        from accessory import show_rgb
        test_img = np.array(rgb)
        test_img[np.isnan(test_img)] = 100
        show_rgb(test_img)

    return rgb


def nan_yellow_pixels(rgb, rlims=[200, 255], glims=[150, 255], blims=[0, 150],
                      gb_delta=30, verb=False):
    """
    set pixel values in range to NaN with restriction that B < G

    :param rgb: RGB pixel array with dimensions: height x width x RGB
    :param rlims: minimum and maximum of R pixel values
    :param glims: maximum and maximum of G pixel values
    :param blims: maximum and maximum of B pixel values
    :param gb_delta: minimum difference between G and B pixel values
    :param verb: verbose mode to show excluded pixels in gray, default False
    :return: RGB pixel array (np.array)
    """
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [nan_yellow_pixels] Input array dimensions ' + \
              str(rgb.shape) + \
              ' incompatible with expected N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [nan_yellow_pixels] Input RGB array must contain ' \
              'element values between 0 and 255. Actual range: ' \
              '[%.1f, %.1f]' % (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    rgb[(rgb[:, :, 0] >= rlims[0]) &
        (rgb[:, :, 0] <= rlims[1]) &
        (rgb[:, :, 1] >= glims[0]) &
        (rgb[:, :, 1] <= glims[1]) &
        (rgb[:, :, 2] >= blims[0]) &
        (rgb[:, :, 2] <= blims[1]) &
        (rgb[:, :, 1] <= rgb[:, :, 0]) &
        (rgb[:, :, 2] < rgb[:, :, 1]-gb_delta), :] = (np.nan, np.nan, np.nan)

    if verb:
        from accessory import show_rgb
        test_img = np.array(rgb)
        test_img[np.isnan(test_img)] = 100
        show_rgb(test_img)

    return rgb


def process_rgb_histogram(hist, omit=[]):
    """
    omit unwanted bins and normalize histogram by dividing out total # pixels

    :param hist: color channel histogram with 256 bins
    :param omit: pixel value bins to omit, default []
    :return: normalized histogram (np.array)
    """
    import numpy as np
    from accessory import get_iterable

    if len(hist) != 256:
        msg = 'ERROR [process_rgb_histogram] Input histogram must contain ' \
              '256 bins (1 per pixel). Actual bin number = %d. ' % len(hist)
        print(msg)
        logging.error(msg)
        sys.exit()

    hist = np.array(hist).astype('float')

    for ii in get_iterable(omit):
        hist[ii] = 0

    tot_pix = sum(hist)
    print(tot_pix)
    try:
        norm_hist = np.divide(hist, tot_pix)
    except RuntimeWarning:
        msg = 'Trying to normalize histogram by dividing by 0 ' \
              'Setting norm_hist to 0 in result.'
        logging.debug(msg)
        norm_hist = np.zeros(len(hist))

    return norm_hist


def rgb_preprocess(rgb, verb=False, exclude_bg=True, upper_lim=[255, 255,
                                                                255]):
    """
    pre-process rgb pixel array to label background and bright (glare) pixels

    :param rgb: RGB pixel array
    :param verb: verbose mode to show excluded pixels, default False
    :param exclude_bg: exclude background (0,0,0) pixels, default True
    :param upper_lim: exclude RGB above upper limit, default (255, 255, 255)
    :return: RGB pixel array (np.array)
    """
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [rgb_preprocess] Input array dimensions ' + \
              str(rgb.shape) + \
              ' incompatible with expected N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [rgb_preprocess] Input RGB array must contain ' \
              'element values between 0 and 255. Actual range: ' \
              '[%.1f, %.1f]' % (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    max_RGB = [255, 255, 255]

    if exclude_bg:
        rgb = nan_background(rgb)
    if np.sum(upper_lim) < np.sum(max_RGB):
        rgb = nan_upper_bound(rgb, upper_lim)

    msg = '[rgb_preprocess] Pre-processing RGB pixel array.'
    logging.debug(msg)
    if verb:
        from accessory import show_rgb
        print(msg)
        test_img = np.array(rgb)
        test_img[np.isnan(test_img)] = 100
        show_rgb(test_img)

    return rgb


def rgb_histogram(rgb, verb=False, process=True, omit=[],
                  outfile='./hist.png'):
    """
    generate histograms for each color channel of input RGB pixel array

    :param rgb: RGB pixel array
    :param verb: verbose mode to show histograms, default False
    :param process: normalize histograms and omit pixel bins, default True
    :param omit: pixel value bins to omit, default []
    :param outfile: file to output histogram figure if verbose
    :return: histogram values for 0-255 for each color channel (np.arrays)
    """
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [rgb_histogram] Input array dimensions ' + \
              str(rgb.shape) + \
              ' incompatible with expected N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [rgb_histogram] Input RGB array must contain ' \
              'element values between 0 and 255. Actual range: ' \
              '[%.1f, %.1f]' % (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    # compute histograms for each color channel
    rh = extract_hist(rgb[:, :, 0])
    gh = extract_hist(rgb[:, :, 1])
    bh = extract_hist(rgb[:, :, 2])

    # omit bins and normalize histograms
    if process:
        rh = process_rgb_histogram(rh, omit)
        gh = process_rgb_histogram(gh, omit)
        bh = process_rgb_histogram(bh, omit)

    msg = '[rgb_histogram] Extracting RGB histograms from pixel array.'
    logging.debug(msg)
    if verb:
        print(msg)
        import matplotlib.pyplot as plt
        from accessory import create_dir
        bins = [ii for ii in range(0, 256)]

        # plot RGB histograms in subplots with shared x axis
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(bins, rh)
        axarr[0].axis('tight')
        axarr[0].set_xlabel('R Pixel Value')
        axarr[0].set_ylabel('Frequency')

        axarr[1].plot(bins, gh)
        axarr[1].axis('tight')
        axarr[1].set_xlabel('G Pixel Value')
        axarr[1].set_ylabel('Frequency')

        axarr[2].plot(bins, bh)
        axarr[2].axis('tight')
        axarr[2].set_xlabel('B Pixel Value')
        axarr[2].set_ylabel('Frequency')

        msg = '[rgb_histogram] Saving RGB histogram figure: %s' % outfile
        logging.info(msg)
        print(msg)

        create_dir(outfile)
        plt.savefig(outfile)
        plt.clf()

    return rh, gh, bh
