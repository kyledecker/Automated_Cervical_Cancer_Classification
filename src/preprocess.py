import sys
import logging


def read_tiff(filename, verb=False):
    """
    read in TIFF image from file and output pixel RGB values

    :param filename: TIFF path and filename
    :param verb: verbosity (set True to show TIFF image), default False
    :return: R, G, B (np.arrays)
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

    return ra, ga, ba


def extract_hist(pix_array, verb=False):
    """
    generate histogram for 1D or 2D array of single color channel pixel values

    :param pix_array: 1D or 2D array of 0 to 255 pixel values
    :param verb: verbosity (set True to show histogram), default False
    :return: histogram frequencies for 0 to 255 (np.array)
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
