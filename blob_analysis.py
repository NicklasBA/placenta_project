
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
import cv2
from numba import jit
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening
from skimage import measure
from scipy.ndimage import gaussian_filter
import glob

# image_folder = sys.argv[1]
# txt_folder = sys.argv[1]
# n_points_in_median = sys.argv[2]

n_points_in_median = 100
image_folder = r'C:\Users\ptrkm\Action Classification\Images'
txt_folder = r'C:\Users\ptrkm\Action Classification\Images'

def calculate_median(image_files):

    for idx, img in enumerate(image_files):
        image = np.asarray(Image.open(img))

        if idx == 0:
            total_array = np.zeros(image.shape+(len(image_files, )))
        total_array[:,:,idx] = image

    median = np.median(total_array)
    return median


def random_subset(folder):
    image_files = os.listdir(folder)
    image_files = np.random.choice(image_files, n_points_in_median,replace=False).tolist()
    image_files = [os.path.join(folder, img) for img in image_files]
    return image_files


class BlobAnalysis:
    def __init__(self, image_folder = image_folder, txt_folder = txt_folder, n_points= n_points_in_median):

        self.image_folder = image_folder
        self.txt_folder = txt_folder
        self.n_points = n_points

        self.background_files = self.random_subset(image_folder)
        self.median = self.calculate_median(self.background_files)


        self.inlet = 50
        self.outlet = 740


    def run_through_sequence(self, path_seq):
        all_blobs = {}
        for idx, path in enumerate(path_seq):
            img = self.sub_median_and_bin(path)
            img = self.open_and_close(img)
            blobs = self.find_blobs(img)
            all_blobs[path] = blobs

        list_of_blobs = list(all_blobs.values())
        left, center, right = self.count_in_sequence(list_of_blobs)
        discard = self.choose_sequences(left, center, right)
        if discard is False:
            return {key: [v.bbox for v in val] for key, val in all_blobs.items()}
        else:
            return discard

    @staticmethod
    def find_blobs(binary_image):
        blob_labels = measure.label(binary_image)
        blob_features = measure.regionprops(blob_labels)

        if blob_features:
            # blob_area = sorted([blob.area for blob in blob_features], reverse=True)
            blobs = [blob for blob in blob_features if blob.area >=50]
            return blobs
        else:
            return None


    @staticmethod
    def choose_sequences(left, center, right):
        discard = False
        for i in range(1,len(center)-1):
            if center[i] > center[i-1]:
                if left[i] >= left[i-1]:
                    discard = True
            elif center[i] < center[i-1]:
                if right[i] <= right[i-1]:
                    discard = True

        return discard

    def count_in_sequence(self,list_of_blobs):

        left = []
        center = []
        right = []
        for blobs in list_of_blobs:
            if not isinstance(blobs, type(None)):
                l, c, r = self.count_in(blobs)
                left.append(l)
                center.append(c)
                right.append(r)

        return left, center, right

    def count_in(self, blobs):
        count_center = 0
        count_left = 0
        count_right = 0
        for b in blobs:
            if self.outlet >= b.centroid[1]>= self.inlet:
                count_center += 1
            elif b.centroid[1] < self.inlet:
                count_left += 1
            elif b.centroid[1] > self.outlet:
                count_right += 1

        return count_left, count_center, count_right

    @staticmethod
    def open_and_close(binary_image):
        img = binary_opening(binary_image)
        img = binary_closing(img)
        return img


    def sub_median_and_bin(self, image_path):
        image = np.asarray(Image.open(image_path)).mean(axis=2)
        image = gaussian_filter(image, sigma = 2)
        image = image - self.median
        # threshold = threshold_otsu(image)
        binary_image = image > 2
        binary_image += image < -2

        return binary_image

    @staticmethod
    def calculate_median(image_files):

        for idx, img in enumerate(image_files):
            image = np.asarray(Image.open(img)).mean(axis = 2)

            if idx == 0:
                total_array = np.zeros(image.shape + (len(image_files), ))
            total_array[:, :, idx] = gaussian_filter(image, sigma=2)

        median = np.median(total_array, axis=2)
        return median

    @staticmethod
    def random_subset(folder):
        np.random.seed(42)
        image_files = os.listdir(folder)
        image_files = np.random.choice(image_files, n_points_in_median, replace=False).tolist()
        image_files = [os.path.join(folder, img) for img in image_files]
        return image_files


test = BlobAnalysis(image_folder,txt_folder,n_points_in_median)
imgs = [glob.glob(image_folder + "\*" + "D234_380"+ str(i)+"*") for i in range(19,68-19)]
imgs = [img[0] for img in imgs]

bboxes = test.run_through_sequence(imgs)
path = list(bboxes.keys())[-1]
im = Image.open(path)

# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(im)

# Create a Rectangle patch
b1 = bboxes[path][0]
b2 = bboxes[path][1]
rect = patches.Rectangle((b1[1],b1[0]), b1[3]-b1[1], b1[2]-b1[0], linewidth=1, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((b2[1],b2[0]), b2[3]-b2[1], b2[2]-b2[0], linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)
ax.add_patch(rect2)
plt.show()

breakpoint()

image_files = test.random_subset(folder=image_folder)
median = test.calculate_median(image_files)
tim = plt.imshow(np.asarray(Image.open(image_files[99])))
plt.show()
breakpoint()

test_img = test.sub_median_and_bin(median = median,image_path=image_files[99])
test_img = test.open_and_close(test_img)

plt.imshow(test_img)
plt.show()
breakpoint()

#
#
# area = []
# for idx, img in enumerate(imgs):
#     x = test.sub_median_and_bin(median = median, image_path=img[0])
#     x = test.open_and_close(x)
#     try:
#         area += test.find_blobs(x)
#     except:
#         breakpoint()
