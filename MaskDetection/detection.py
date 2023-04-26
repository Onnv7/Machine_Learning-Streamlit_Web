import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import itertools
import os
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='./MaskDetection/mask.pt')


def process(frame):
    results = model(frame)
    img = np.squeeze(results.render())
    return img
