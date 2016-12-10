def show_rgb(rgb):
    from PIL import Image
    import numpy as np

    rgb = rgb.astype(dtype=np.uint8)
    img = Image.fromarray(rgb, 'RGB')
    img.show()


def get_iterable(x):
    import collections
    if isinstance(x, collections.Iterable):
        return x
    else:
        return x,
