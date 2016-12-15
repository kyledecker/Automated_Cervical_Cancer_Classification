import sys
import os
import logging


def show_rgb(rgb):
    """
    display RGB image from pixel array

    :param rgb: N x M x 3 RGB pixel array
    """
    from PIL import Image
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [show_rgb] Input array dimensions ' + str(rgb.shape) + \
              ' incompatible with expected N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [show_rgb] Input RGB array must contain element ' \
              'values between 0 and 255. Actual range: [%.1f, %.1f]' % \
              (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    rgb = rgb.astype(dtype=np.uint8)
    img = Image.fromarray(rgb, 'RGB')
    img.show()


def save_rgb(rgb, filename='./rgb.png'):
    """
    save RGB image from pixel array

    :param rgb: N x M x 3 RGB pixel array
    :param filename: file where rgb image is saved
    """
    from PIL import Image
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [save_rgb] Input array dimensions ' + str(rgb.shape) + \
              ' incompatible with expected N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [save_rgb] Input RGB array must contain element ' \
              'values between 0 and 255. Actual range: [%.1f, %.1f]' % \
              (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    rgb = rgb.astype(dtype=np.uint8)
    img = Image.fromarray(rgb, 'RGB')

    create_dir(filename)
    img.save(filename)


def create_dir(filepath):
    """
    create new folder if directory in file path does not exist

    :param filepath: file path and name
    """
    out_dir = os.path.dirname(filepath)
    if not os.path.exists(out_dir) and out_dir != '':
        try:
            msg = '[create_dir] Creating new directory: ' + out_dir
            print(msg)
            logging.info(msg)
            os.makedirs(out_dir)
        except:
            msg = 'ERROR [create_dir] Invalid output path ' + out_dir + \
                  '. Exiting script...'
            print(msg)
            logging.error(msg)
            sys.exit()


def get_iterable(x):
    """
    convert int value into iterable array

    :param x: parameter to be iterated on
    :return: iterable array
    """
    import collections
    if isinstance(x, collections.Iterable):
        return x
    else:
        return x,


def color_nans(rgb, color=[0, 0, 255], verb=False):
    """
    assign color to NaN pixels in image

    :param rgb: RGB pixel array with dimensions: height x width x RGB
    :param color: RGB color used in place of NaN pixel
    :param verb: verbose mode to show colored image, default False
    :return: RGB pixel array (np.array)
    """
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [color_nans] Input array dimensions ' + str(rgb.shape) + \
              ' incompatible with expected N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [color_nans] Input RGB array must contain element ' \
              'values between 0 and 255. Actual range: [%.1f, %.1f]' % \
              (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(color) > 255 or np.min(color) < 0:
        msg = 'ERROR [color_nans] Input color must contain element values ' \
              'between 0 and 255'
        print(msg)
        logging.error(msg)
        sys.exit()

    try:
        rgb[np.isnan(rgb[:, :, 0]), :] = color
    except ValueError as err:
        msg = 'ERROR [color_nans] Input color must be an array of 3 RGB ' \
              'elements: {0}'.format(err)
        print(msg)
        logging.error(msg)
        sys.exit()

    if verb:
        from accessory import show_rgb
        show_rgb(rgb)

    return rgb


def rgbstring2index(rgbstring):
    """
    converts RGB string into array containing corresponding channel indices

    :param rgbstring: string of letters specifying desired color channels
    :return: array of rgb channel index (np.array)
    """
    import numpy as np
    import re
    from accessory import get_iterable

    if len(re.findall('[rgb]', rgbstring.lower())) != len(rgbstring):
        msg = 'ERROR [rgbstring2index] Input string ' + \
              rgbstring + 'must contain only r, g, or b characters.'
        print(msg)
        logging.error(msg)
        sys.exit()

    idx = np.array([])
    if 'r' in rgbstring.lower():
        idx = np.append(idx, 0)

    if 'g' in rgbstring.lower():
        idx = np.append(idx, 1)

    if 'b' in rgbstring.lower():
        idx = np.append(idx, 2)

    idx = idx.astype('int')

    return get_iterable(idx)


def percent_color(rgb, color):
    """
    calculate percent of RGB color in image (black pixels excluded)

    :param rgb: RGB pixel array with dimensions: height x width x RGB
    :param color: RGB color to calculate percentage of
    :return: percent of color in image
    """
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        msg = 'ERROR [percent_color] Input array dimensions ' + \
              str(rgb.shape) + \
              ' incompatible with expected N x M x 3 RGB input.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(rgb) > 255 or np.min(rgb) < 0:
        msg = 'ERROR [percent_color] Input RGB array must contain element ' \
              'values between 0 and 255. Actual range: [%.1f, %.1f]' % \
              (np.min(rgb), np.max(rgb))
        print(msg)
        logging.error(msg)
        sys.exit()

    if np.max(color) > 255 or np.min(color) < 0:
        msg = 'ERROR [percent_color] Input color must contain element ' \
              'values between 0 and 255'
        print(msg)
        logging.error(msg)
        sys.exit()

    nCol = np.sum((rgb[:, :, 0] == color[0]) & (rgb[:, :, 1] == color[1]) &
                  (rgb[:, :, 2] == color[2]))

    nTot = np.sum(~((rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 0) &
                    (rgb[:, :, 2] == 0)))

    percent = 100*nCol/nTot

    return percent


def write_csv(labels, values, outfile='./outputs/'):
    """
    write labeled values to a csv file

    :param labels: labels for each value
    :param values: values to write to file
    :param outfile: name and path of saved .csv
    :return:
    """
    if not ('.csv' in outfile):
        msg = 'ERROR [write_csv] Output filename does not have .csv extension.'
        print(msg)
        logging.error(msg)
        sys.exit()

    if len(labels) != len(values):
        msg = 'ERROR [write_csv] Dimension mismatch between number of ' \
              'labels and number of values to write to csv. %d labels and ' \
              '!= %d values' % (len(labels), len(values))
        print(msg)
        logging.error(msg)
        sys.exit()

    create_dir(outfile)
    with open(outfile, 'w') as f:
        [f.write('{0}, {1}\n'.format(labels[ii], values[ii]))
         for ii in range(0, len(values))]

    msg = '[write_csv] Saving csv file to: %s' % outfile
    logging.info(msg)
    print(msg)
