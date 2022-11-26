import numpy as np
import cv2
from typing import Dict, Collection
from onnxruntime import InferenceSession, get_available_providers
from . import configs
from . import download
from .utils import priorbox, nms, decode, decode_landm


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
           config: Dict,
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

    prior_data = priorbox(config, image_size=(640, 640))
    boxes = decode(locations[0], prior_data, config['variance'])

    # Rescale bounding boxes
    boxes = boxes * scale
    scores = confidences[0, :, 1]

    # Rescale landmarks
    landmarks = decode_landm(landmarks[0], prior_data, config['variance'])
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


def get_model_path(url):
    filename = path.basename(url)


class FaceDetector:
    def __init__(self,
                 model: str = 'mobilenet',
                 onnx_providers: Collection[str] = get_available_providers(),
                 config: dict = None,
                 confidence_threshold: float = 0.02,
                 top_k: int = 5000,
                 keep_top_k: int = 750,
                 nms_threshold: float = 0.4):

        if config is None:
            config = getattr(configs, model)

        self.config = config
        self.model = InferenceSession(
            download.download_model(config['url']),
            providers=onnx_providers
        )
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold

    def __call__(self, image: np.ndarray):
        return detect(self.model,
                      self.config,
                      image=image,
                      top_k=self.top_k,
                      keep_top_k=self.keep_top_k,
                      nms_threshold=self.nms_threshold,
                      confidence_threshold=self.confidence_threshold)
