
import numpy as np
import pandas as pd
import sys
import os
import skvideo.io
from openCV import cv2
folder = r'C:\Users\ptrkm\PycharmProjects\20180419_5_6mbar_500fps_D204_D204_0173.mp4'

SAVE_PATH = r''
def read_and_substract(path):
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True
    images = []
    while (fc < frameCount and ret):
        ret, frame = cap.read()
        buf[fc] = frame
        images.append(frame)
        fc += 1
    cap.release()
    median_image = np.median(buf, axis=2)
    images = [img - median_image for img in images]
    height, width, layers = images[-1].shape
    new_path = os.path.join(SAVE_PATH,os.path.basename(path))
    out = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (height, width))
    for i in range(len(images)):
        out.write(images[i])
    out.release

    print("Video created for {}".format(os.path.basename(path)))

read_and_substract(folder)






