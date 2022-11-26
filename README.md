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

```python
from retinaface_detector import FaceDetector
import cv2

detector = FaceDetector()
cap = cv2.VideoCapture(0)
_, frame = cap.read()
boxes, scores, landmarks = detector(frame)
```

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
