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

