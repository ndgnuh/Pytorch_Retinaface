import numpy as np
import cv2
from .config import cfg_mnet, cfg_re50
from .utils import priorbox, nms, decode, decode_landm
try:
    from onnxruntime import InferenceSession
except Exception:
    print(
        "Install this package with [cpu] or [gpu] option, or install onnxruntime manually")


def prepare_input(image: np.ndarray):
    # Resize to model size
    image = cv2.resize(image, (640, 640))

    # convert to float32
    # for some reasons, they did not normalize the image to 0..1
    image = image - np.array([104, 117, 123])
    image = image.astype('float32')

    # To c, h, w
    image = image.transpose((2, 0, 1))

    # Add batch dim
    image = image[None, ...]

    return image


def detect(model: InferenceSession,
           image: np.ndarray,
           confidence_threshold: float = 0.02,
           top_k: int = 5000,
           keep_top_k: int = 750,
           nms_threshold: float = 0.4):
    im_height, im_width = image.shape[:2]
    scale = np.array([im_width, im_height, im_width, im_height])

    # Forward
    input_image = prepare_input(image)
    locations, confidences, landmarks = model.run(
        None,
        {
            model.get_inputs()[0].name: input_image
        }
    )

    # TODO: cfg
    cfg = cfg_mnet
    prior_data = priorbox(cfg, image_size=(640, 640))
    boxes = decode(locations[0], prior_data, cfg['variance'])

    # Rescale bounding boxes
    boxes = boxes * scale
    scores = confidences[0, :, 1]

    # Rescale landmarks
    landmarks = decode_landm(landmarks[0], prior_data, cfg['variance'])
    scale1 = np.array([
        input_image.shape[3],
        input_image.shape[2],
        input_image.shape[3],
        input_image.shape[2],
        input_image.shape[3],
        input_image.shape[2],
        input_image.shape[3],
        input_image.shape[2],
        input_image.shape[3],
        input_image.shape[2]])
    landmarks = landmarks * scale1

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    # do NMS
    detections = np.hstack(
        [boxes, scores[:, np.newaxis]]
    ).astype(np.float32, copy=False)
    keep = nms(detections, nms_threshold)
    detections = detections[keep, :]
    landmarks = landmarks[keep]

    # keep top-K faster NMS
    detections = detections[:keep_top_k, :]
    landmarks = landmarks[:keep_top_k, :]

    # boxes
    boxes = detections[:, :4].round().astype(int)
    scores = detections[:, 4]

    return boxes, scores, landmarks
