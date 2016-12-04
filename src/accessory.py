def show_rgb(rgb):
    from PIL import Image
    import numpy as np

    rgb = rgb.astype(dtype=np.uint8)
    img = Image.fromarray(rgb, 'RGB')
    img.show()
