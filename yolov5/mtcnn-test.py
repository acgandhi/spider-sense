import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import facenet_pytorch
from facenet_pytorch import MTCNN, InceptionResnetV1
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, bbox_iou
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from icecream import ic

elDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(elDevice))
mtcnn = MTCNN(factor=0.25, keep_all=True, device=elDevice)

img = cv2.cvtColor(cv2.imread("/content/spider-sense/ivan_test_pics/white_ivan.png"), cv2.COLOR_BGR2RGB)

mtcnn = MTCNN(factor=0.25, keep_all=True, device=device)

# Get cropped and prewhitened image tensor
t0 = time.time()
boxes = mtcnn.detect(img)
t1 = time.time() - t0
im = Image.fromarray(img)
draw = ImageDraw.Draw(im)
try:
    for box in boxes[0].tolist():
        draw.rectangle(box, outline=(255, 0, 0), width=1)
except:
    pass
im.save("/content/result.jpg")
print(img.shape)
print(t1, boxes)