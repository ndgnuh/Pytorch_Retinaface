# RetinaFace Detector

I just want to use RetinaFace conveniently. Most if not all of the codes in this repository are adapted from https://github.com/biubug6/Pytorch_Retinaface.

### Contents
- [Installation](#installation)
- [Inference](#inference)
- [References](#references)

## Installation

```shell
pip install git+https://github.com/ndgnuh/retinaface_detector
```

Install ONNX runtime:
```shell
pip install onnxruntime-gpu
# or
pip install onnxruntime
```

## Inference

Quick start:
```python
from retinaface_detector import FaceDetector
import cv2

detector = FaceDetector()
cap = cv2.VideoCapture(0)
_, frame = cap.read()
boxes, scores, landmarks = detector(frame)
```

Full settings:
```python
FaceDetector(
  model: str = 'mobilenet', # model name, either 'mobilenet' or 'resnet50', anything in the `retinaface_detector/configs.py`
  onnx_providers: Collection[str] = onnxruntime.get_available_providers(),
  config: dict = None, # default to getattr(configs, model)
  confidence_threshold: float = 0.02, # refer to https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
  top_k: int = 5000, # refer to https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
  keep_top_k: int = 750, # refer to https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
  nms_threshold: float = 0.4, # refer to https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
  square_box: Optional[str] = None) # does nothing by default, see below for details
```

### Square box

This options modify the bounding box to a square:
```python
x1 = max(cx - sz, 0)
x2 = cx + sz
y1 = max(cy - sz, 0)
y2 = cy + sz
```
where `cx`, `cy` are the center coordinates of the box, `sz` is calculated based on the `square_box` argument:
- `"max"`: `max(w, h) // 2`
- `"min"`: `min(w, h) // 2`
- `"avg"`: `(w + h) // 4`

If `square_box` is `None`, do nothing.

## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- [Retinaface (torch)](https://github.com/biubug6/Pytorch_Retinaface)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
