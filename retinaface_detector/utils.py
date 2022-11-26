from itertools import product as product
from math import ceil
import numpy as np


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 2:4] *
                             variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 4:6] *
                             variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 6:8] *
                             variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 8:10] *
                             variances[0] * priors[:, 2:],
                             ), axis=1)
    return landms


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def priorbox(cfg, image_size):
    # Configs
    min_sizes = cfg['min_sizes']
    steps = cfg['steps']
    clip = cfg['clip']
    image_size = image_size
    feature_maps = [
        [
            ceil(image_size[0]/step),
            ceil(image_size[1]/step)
        ] for step in steps
    ]

    # Forward
    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1]
                            for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0]
                            for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    output = np.array(anchors).reshape(-1, 4)
    if clip:
        np.clip(output, 0, 1)
    return output


def nms(dets, thresh):
    """
    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------
    """
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def square_box(box, mode):
    """
    Modify the bounding box so that it is squared

    `box`: x1 y1 x2 y2 tuple
    `mode`: how to get the square size
    - `max`: max of box width, height
    - `min`: min of box width, height
    - `avg`: average of box width, height
    """
    # If the first argument is a box
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1
    if mode == "max":
        sz = max(w, h) // 2
    elif mode == "min":
        sz = min(w, h) // 2
    elif mode == "avg":
        sz = (w + h) // 4
    else:
        raise ValueError(
            f"Unsupported mode {mode}, valid values are 'max', 'min' and 'avg"
        )

    # New coordinate
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    x1 = max(cx - sz, 0)
    x2 = cx + sz
    y1 = max(cy - sz, 0)
    y2 = cy + sz
    return x1, y1, x2, y2
