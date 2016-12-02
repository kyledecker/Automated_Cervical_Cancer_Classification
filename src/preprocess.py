def read_tiff(filename, verb=0):
    from PIL import Image
    import numpy as np

    img = Image.open(filename)

    if verb:
        img.show()

    img.getdata()
    r, g, b = img.split()

    ra = np.array(r)
    ga = np.array(g)
    ba = np.array(b)

    return ra, ga, ba