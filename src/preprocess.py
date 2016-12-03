import sys
import logging


def read_tiff(filename, verb=False):
    """
    read in TIFF image from file and output pixel RGB values

    :param filename: TIFF path and filename
    :param verb: verbosity (set True to show TIFF image), default False
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
    :param verb: verbosity (set True to show histogram), default False
    :return: histogram frequencies for 0-255 (np.array)
    """
    import numpy as np

    if np.max(pix_array) > 255 or np.min(pix_array) < 0:
        msg = 'ERROR [extract_hist] Pixel array value out of bounds (min=0, ' \
              'max=255). Actual bounds (min=%d, max=%d).' % (np.min(
                pix_array), np.max(pix_array))
        logging.error(msg)
        print(msg)
        sys.exit()

    pix_array = np.ravel(pix_array)
    pix_array = pix_array[np.isfinite(pix_array)]
    hist, bin_edges = np.histogram(pix_array, bins=256, range=(0, 255))

    if verb:
        import matplotlib.pyplot as plt
        plt.hist(pix_array, bins=256, range=(0, 255))
        plt.axis('tight')
        plt.show()

    return hist


def remove_background(rgb):
    """
    identify background pixels (R=0, B=0, and G=0) and convert to NaN

    :param rgb: RGB pixel array with dimensions: height x width x RGB
    :return: RGB pixel array (np.array)
    """
    import numpy as np

    img_shape = np.shape(rgb)
    if img_shape[2] != 3:
        msg = 'ERROR [remove_background] Dimensions of input RGB pixel array ' \
              'incorrect. Expected dimensions are height x width x RGB.'
        logging.error(msg)
        print(msg)
        sys.exit()

    for ii in range(0, img_shape[0]):
        for jj in range(0, img_shape[1]):
            if np.sum(rgb[ii, jj]) == 0:
                rgb[ii, jj, :] = (np.nan, np.nan, np.nan)

    return rgb


def rgb_histogram(rgb, verb=False, exclude_bg=True):
    """
    generate histograms for each color channel of input RGB pixel array

    :param rgb: RGB pixel array
    :param verb: verbosity (set True to show histograms), default False
    :param exclude_bg: flag to exclude background (0,0,0) pixels, default True
    :return: histogram frequencies for 0-255 for each color channel (np.arrays)
    """
    import numpy as np

    if exclude_bg:
        rgb = remove_background(rgb)

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

    msg = '[rgb_histogram] Extracting RGB histograms from pixel array.'
    logging.debug(msg)
    if verb:
        print(msg)
        import matplotlib.pyplot as plt
        bins = [ii for ii in range(0, 256)]

        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(bins, rh)
        axarr[1].plot(bins, gh)
        axarr[2].plot(bins, bh)
        plt.show()

    return rh, gh, bh
