import numpy as np
from PIL import Image

def crop(image, threshold = 0):

    coords = np.argwhere(image > threshold)
    
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)

    return image[ymin: ymax + 1, xmin: xmax + 1]

def scale_to_MNIST(cropped, target_size = 20):

    h, w = cropped.shape

    scale = target_size/max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    img_pil = Image.fromarray(cropped)
    resize = img_pil.resize((new_w, new_h))
    scaled_digit = np.array(resize)

    canvas = np.zeros((28, 28), dtype=scaled_digit.dtype)

    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    canvas[y_offset: y_offset + new_h, x_offset: x_offset + new_w] = scaled_digit

    return canvas/255

