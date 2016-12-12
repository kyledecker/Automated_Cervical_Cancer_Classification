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

    rgb = rgb.astype(dtype=np.uint8)
    img = Image.fromarray(rgb, 'RGB')
    img.show()


def save_rgb(rgb, filename='./rgb.png'):
    """
    save RGB image from pixel array

    :param rgb: N x M x 3 RGB pixel array
    """
    from PIL import Image
    import numpy as np

    rgb = rgb.astype(dtype=np.uint8)
    img = Image.fromarray(rgb, 'RGB')
    img.save(filename)


def create_dir(filepath):
    """
    create new folder if directory in file path does not exist

    :param filepath: file path and name
    """
    out_dir = os.path.dirname(filepath)
    if not os.path.exists(out_dir):
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

    :param x:
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

    rgb[np.isnan(rgb[:, :, 0]), :] = color

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
    from accessory import get_iterable

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

    nCol = np.sum((rgb[:, :, 0] == color[0]) & (rgb[:, :, 1] == color[1]) &
                  (rgb[:, :, 2] == color[2]))

    nTot = np.sum(~((rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 0) &
                    (rgb[:, :, 2] == 0)))

    percent = 100*nCol/nTot

    return percent
