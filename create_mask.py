


import numpy as np
import pandas as pd
import os
import shutil
import pickle
from PIL import Image
import cv2
import glob

PADDED_PIXELS = 10


def rotate_im(image, angle = 10):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occudpied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
    return image

def add_mask(image, bbox):
    mask = np.zeros_like(image)
    bbox[0] -= PADDED_PIXELS
    bbox[1] -= PADDED_PIXELS
    bbox[3] += PADDED_PIXELS
    bbox[2] += PADDED_PIXELS

    mask[bbox[0]:bbox[3],bbox[1]:bbox[4],:] = 1


    return mask

def combine_image_and_bbox(image, all_bbox):
    """

    :param image: np.ndarray of image
    :param all_bbox: list of all bounding boxes
    :return: the masked image
    """

    mask = np.zeros_like(image)
    for bbox in all_bbox:
        mask += add_mask(image, bbox, noise)

    new_image = image * mask

    return new_image

def find_frames(path_to_csv):

    frames = glob.glob(path_to_csv + "*.csv")
    return frames


def collect_frames(path_to_frame):
    """

    :param path_to_frame: path to the frame returned by the preproccessing containing bounding boxes
    :return: list of frames and [list[list]] of associated bounding boxes
    """

    csv = pd.read_csv(path_to_frame)
    frames = list(pd.unique(csv['FrameName']))
    bb_dict = {}

    for frame in frames:
        temp = csv[csv['FrameName'] == frame]
        bbox = [list(i) for i in temp[["x1", "y1", "x2", "y2"]].values]
        bb_dict[frame] = bbox

    return bb_dict

def collect_path_dict(ground_path):

    all_folders = [os.path.join(ground_path,file) for file in os.listdir(ground_path) if len(file) > 28]

    path_to_im = {}
    for folder in all_folders:
        files = os.listdir(folder)
        for file in files:
            path_to_im[os.path.join(folder, file)] = file

    return path_to_im, all_folders


def save_video(paths, OUTDIR, video_name, path_to_im, bb_dict):
    img_array = []
    for filename in paths:
        try:
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img = combine_image_and_bbox(img, bb_dict[path_to_im[filename]])
            img_array.append(img)
        except:
            return None
    # print(f"\tFound and loaded {len(img_array)} images.")
    out = cv2.VideoWriter(f'{OUTDIR}{video_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    # print(f"\tWriting to {OUTDIR}{video_name}.mp4")
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    # print("Thank you, next")

if __name__ == '__main__':

    ground_path = r'/scratch/s183993/placenta/raw_data/frames'
    OUTDIR = r'/scratch/s183993/placenta/raw_data/videos/videos_blackened'
    path_to_csv = ground_path
    paths_to_csv = find_frames(path_to_csv)

    bb_dict = {}
    for path in paths_to_csv:
        bb_dict.update(collect_frames(path))

    path_to_im, all_folders = collect_path_dict(ground_path)
    path_list = [glob.glob(os.path.join(folder,"") + "*.png") for folder in all_folders]
    video_names = [folder.split(os.sep)[-1] for folder in all_folders]

    for paths, name in list(zip(path_list, video_names)):
        save_video(paths, OUTDIR, name, path_to_im, bb_dict)
        print("Succesfully printed for " + name)


