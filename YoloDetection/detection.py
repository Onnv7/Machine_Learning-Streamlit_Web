import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import itertools
import os


model = None
mask_model = torch.hub.load('ultralytics/yolov5', 'custom',
                            path="./YoloDetection/models/mask.pt")
smoke_model = torch.hub.load('ultralytics/yolov5', 'custom',
                             path="./YoloDetection/models/smoke.pt")


def load_model(mode):
    if(mode == "smoke"):
        mode = smoke_model
    elif(mode == "mask"):
        mode = mask_model
    print(model)


def process(frame, mode):
    if(mode == "Smoke"):
        results = smoke_model(frame)
    elif(mode == "Mask or No Mask"):
        results = mask_model(frame)
    img = np.squeeze(results.render())
    return img
