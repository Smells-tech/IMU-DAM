import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize

def kp_invert(kp, frame):
    return kp['value'] * frame.shape + frame.shape / 2

def kp_bbox(kp):
    y = kp['value'][..., 0]
    x = kp['value'][..., 1]
    left, top = min(x), min(y)
    h, w = max(x) - left, max(y) - top
    return left, top, h, w

def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    kp = fa.estimate_kp(frame)
    if len(kp) == 0:
        return []
    kp = kp_invert(kp, frame)
    return np.array(kp_bbox(kp)) * scale_factor

def compute_bbox(tube_bbox, frame_shape, increase_area=0.1, y_offset=0, x_offset=0, square=True):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    # Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    h, w = bot - top, right - left
    if square:
        mn = min(h, w)
        h, w = mn, mn

    y_offset = int(y_offset * frame_shape[0])
    top = max(top + y_offset, 0)
    x_offset = int(x_offset * frame_shape[1])
    left = max(left + x_offset, 0)

    underflow = min(top, left, 0)
    top -= underflow
    left -= underflow

    overflow = min(frame_shape[0] - (top + h), frame_shape[1] - (left + w), 0)
    h += overflow
    w += overflow

    return left, top, w, h
