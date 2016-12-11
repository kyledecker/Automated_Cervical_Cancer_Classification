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

    img = Image.open(filename)
    msg = '[read_tiff] Image data loaded from ' + filename
    logging.debug(msg)

    if verb:
        print(msg)
        img.show()

    img.getdata()
    try:
        r, g, b = img.split()
    except ValueError as err:
        msg = 'ERROR [read_tiff] RGB color channels not ' \
              'available: {0}'.format(err)
        logging.error(msg)
        print(msg)
        sys.exit()

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
        msg = 'ERROR [extract_hist] Pixel array value out of bounds (min=0, ' \
              'max=255). Actual bounds (min=%d, max=%d).' % (np.min(
                pix_array), np.max(pix_array))
        logging.error(msg)
        print(msg)
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


def remove_background(rgb, verb=False):
    """
    identify background pixels (R=0, B=0, and G=0) and convert to NaN

    :param rgb: RGB pixel array with dimensions: height x width x RGB
    :param verb: verbose mode to show excluded pixels in gray, default False
    :return: RGB pixel array (np.array)
    """
    import numpy as np

    img_shape = np.shape(rgb)
    if img_shape[2] != 3:
        msg = 'ERROR [remove_background] Dimensions of input RGB pixel ' \
              'array incorrect. Expected dimensions are height x width x RGB.'
        logging.error(msg)
        print(msg)
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


def limit_upper_bound(rgb, lim=(255, 255, 255), verb=False):
    """
    identify bright pixels above RGB threshold and convert to NaN

    :param rgb: RGB pixel array with dimensions: height x width x RGB
    :param lim: RGB pixel value upper limit, default (255, 255, 255)
    :param verb: verbose mode to show excluded pixels in gray, default False
    :return: RGB pixel array (np.array)
    """
    import numpy as np

    img_shape = np.shape(rgb)
    if img_shape[2] != 3:
        msg = 'ERROR [limit_upper_bound] Dimensions of input RGB pixel ' \
              'array incorrect. Expected dimensions are height x width x RGB.'
        logging.error(msg)
        print(msg)
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


def remove_yellow_pixels(rgb, tol=10, verb=False):
    """
    identify yellow pixels and convert to NaN

    :param rgb: RGB pixel array with dimensions: height x width x RGB
    :param tol: tolerance of R pixel value from max 255
    :param verb: verbose mode to show excluded pixels in gray, default False
    :return: RGB pixel array (np.array)
    """
    import numpy as np

    img_shape = np.shape(rgb)
    if img_shape[2] != 3:
        msg = 'ERROR [limit_upper_bound] Dimensions of input RGB pixel ' \
              'array incorrect. Expected dimensions are height x width x RGB.'
        logging.error(msg)
        print(msg)
        sys.exit()

    rgb[(rgb[:, :, 0] > 255-tol) &
        (rgb[:, :, 2] < rgb[:, :, 1]), :] = (np.nan, np.nan, np.nan)

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

    hist = np.array(hist).astype('float')

    for ii in get_iterable(omit):
        hist[ii] = 0

    tot_pix = sum(hist)
    norm_hist = np.divide(hist, tot_pix)

    return norm_hist


def rgb_preprocess(rgb, verb=False, exclude_bg=True, upper_lim=(255, 255,
                                                                255)):
    """
    pre-process rgb pixel array to label background and bright (glare) pixels

    :param rgb: RGB pixel array
    :param verb: verbose mode to show excluded pixels, default False
    :param exclude_bg: exclude background (0,0,0) pixels, default True
    :param upper_lim: exclude RGB above upper limit, default (255, 255, 255)
    :return: RGB pixel array (np.array)
    """
    import numpy as np

    max_RGB = (255, 255, 255)

    if exclude_bg:
        rgb = remove_background(rgb)
    if np.sum(upper_lim) < np.sum(max_RGB):
        rgb = limit_upper_bound(rgb, upper_lim)

    msg = '[rgb_preprocess] Pre-processing RGB pixel array.'
    logging.debug(msg)
    if verb:
        from accessory import show_rgb
        print(msg)
        test_img = np.array(rgb)
        test_img[np.isnan(test_img)] = 100
        show_rgb(test_img)

    return rgb


def rgb_histogram(rgb, verb=False, process=True, omit=[]):
    """
    generate histograms for each color channel of input RGB pixel array

    :param rgb: RGB pixel array
    :param verb: verbose mode to show histograms, default False
    :param process: normalize histograms and omit pixel bins, default True
    :param omit: pixel value bins to omit, default []
    :return: histogram values for 0-255 for each color channel (np.arrays)
    """
    import numpy as np

    img_shape = np.shape(rgb)
    if img_shape[2] != 3:
        msg = 'ERROR [rgb_histogram] Dimensions of input RGB pixel array ' \
              'incorrect. Expected dimensions are height x width x RGB.'
        logging.error(msg)
        print(msg)
        sys.exit()

    rh = extract_hist(np.squeeze(rgb[:, :, 0]))
    gh = extract_hist(np.squeeze(rgb[:, :, 1]))
    bh = extract_hist(np.squeeze(rgb[:, :, 2]))

    if process:
        rh = process_rgb_histogram(rh, omit)
        gh = process_rgb_histogram(gh, omit)
        bh = process_rgb_histogram(bh, omit)

    msg = '[rgb_histogram] Extracting RGB histograms from pixel array.'
    logging.debug(msg)
    if verb:
        print(msg)
        import matplotlib.pyplot as plt
        bins = [ii for ii in range(0, 256)]

        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].bar(bins, rh)
        axarr[0].axis('tight')
        axarr[0].set_xlabel('R Pixel Value')
        axarr[0].set_ylabel('Frequency')

        axarr[1].bar(bins, gh)
        axarr[1].axis('tight')
        axarr[1].set_xlabel('G Pixel Value')
        axarr[1].set_ylabel('Frequency')

        axarr[2].bar(bins, bh)
        axarr[2].axis('tight')
        axarr[2].set_xlabel('B Pixel Value')
        axarr[2].set_ylabel('Frequency')

        plt.show()

    return rh, gh, bh
