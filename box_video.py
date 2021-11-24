# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:20:08 2021

@author: Nicklas
"""

import pandas as pd
import cv2
import glob
import os
import sys
import numpy as np

import matplotlib.patches as patches
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing, binary_opening
from skimage import measure
from scipy.ndimage import gaussian_filter

path = "/scratch/s183993/placenta/raw_data/datadump/20180307_5_6mbar_500fps_D130.csv"
df = pd.read_csv (path)
        
df1 = df.groupby(by=['FrameName']).agg(list)
    
folder_path = "/scratch/s183993/placenta/raw_data/datadump/20180307_5_6mbar_500fps_D130"

image_paths = glob.glob(os.path.join(folder_path, df.FrameName[0]))
#outout = (os.path.join(folder_path, 'out5.png'))


def draw_bboxes(img, bboxes,text, color=(0, 0, 255), thickness=1):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100,50)
    fontScale              = 1
    fontColor              = (255,0,0)
    lineType               = 2
    for bbox in bboxes:
        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[-2:]), color, thickness)
        cv2.putText(image,str(text), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
        img_array.append(image)
        

img_array = []
for i in range(len(df)):    
    image_paths = glob.glob(os.path.join(folder_path, df.FrameName[i]))
    text = df.FrameName[i]
    image = cv2.imread(image_paths[0])
    height, width, layers = image.shape
    size = (width,height)
    bbox = [np.array([df1.x1[i], df1.y1[i], df1.x2[i], df1.y2[i]])]
    draw_bboxes(image,bbox,text)

out = cv2.VideoWriter('project130.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

#cv2.imshow("OpenCV/Numpy normal", image)
print("saving")
#cv2.imwrite(outout,image)