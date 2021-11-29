


import numpy as np
import pandas as pd
import os
import shutil
import pickle
from PIL import Image
import cv2
import glob

PADDED_PIXELS = 50
IMAGE_SIZE = (250, 250)
NOICE_STD = 9
PADDING = 50



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

def get_bbox(bbox):
    if bbox[0] > bbox[2]:
        topy = bbox[2]
        bottomy = bbox[0]
    else:
        topy = bbox[0]
        bottomy = bbox[2]
    if bbox[1] < bbox[3]:
        leftx = bbox[1]
        rightx = bbox[3]
    else:
        leftx = bbox[3]
        rightx = bbox[1]

    assert topy < bottomy
    assert leftx < rightx

    coordinates = (leftx, rightx, topy, bottomy)
    return coordinates

def add_mask(sizes, bbox):
    mask = np.zeros(sizes)
    n,m = sizes[0], sizes[1]

    if bbox[0] > bbox[2]:
        topy = bbox[2]
        bottomy = bbox[0]
    else:
        topy = bbox[0]
        bottomy = bbox[2]
    if bbox[1] < bbox[3]:
        leftx = bbox[1]
        rightx = bbox[3]
    else:
        leftx = bbox[3]
        rightx = bbox[1]

    assert topy < bottomy
    assert leftx < rightx

    coordinates_inner = (leftx, rightx, topy, bottomy)
    center = centroid(coordinates_inner)

    leftx = np.max([0, center[0]-PADDED_PIXELS])
    rightx = np.min([n, center[0]+PADDED_PIXELS])
    topy = np.max([0, center[1]-PADDED_PIXELS])
    bottomy = np.min([m, center[1] + PADDED_PIXELS])

    coordinates = (leftx, rightx, topy, bottomy)
    mask[leftx:rightx,topy:bottomy,:] = 1

    if np.sum(mask[:,:,0]) ==0:
        breakpoint()

    return mask, coordinates, coordinates_inner

def centroid(bbox):
    row = (bbox[0]+bbox[1])//2
    column = (bbox[2]+bbox[3])//2
    return [row, column]


def get_area(bbox):
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

def combine_image_and_bbox_into_rcnn_struct(image, all_bbox):
    sizes = image.shape

    new_all = []
    for bbox in all_bbox:
        new_all.append(get_bbox(bbox))

    return sizes, new_all

def combine_image_and_bbox(image, all_bbox):
    """
    :param image: np.ndarray of image
    :param all_bbox: list of all bounding boxes
    :return: the masked image
    """
    sizes = image.shape
    for bbox in all_bbox:
        mask, coordinates, coordinates_inner = add_mask(sizes, bbox)

    temp = np.copy(image[coordinates_inner[0]:coordinates_inner[1], coordinates_inner[2]:coordinates_inner[3],:])
    row, col = get_padding(coordinates, sizes)
    breakpoint()
    temp = np.pad(temp, (row, col,[0,0]), mode='constant')
    breakpoint()
    # temp = add_noise(temp, coordinates, coordinates_inner)
    return temp

def get_padding(coordinates, sizes):
    ty = coordinates[0]
    by = coordinates[1]
    lx = coordinates[2]
    rx = coordinates[3]
    row = [0,0]
    col = [0,0]
    if abs(ty -by) == PADDING and abs(lx - rx) == PADDING:
        return row, col
    else:
        if ty == 0:
            row[0] = abs(PADDING-by)
        if lx == 0:
            col[0] = abs(PADDING-rx)
        if by == sizes[0]:
            row[1] = abs(PADDING-abs(ty-by))
        if rx == sizes[1]:
            col[1] = abs(PADDING-abs(lx-rx))

    return row, col

def add_noise(img, coordinates, coordinates_inner):
    dty = abs(coordinates[0]-coordinates_inner[0])
    dby = abs(coordinates[2]-coordinates_inner[2])
    dlx = abs(coordinates[1]-coordinates_inner[1])
    drx = abs(coordinates[3]-coordinates_inner[3])

    mask = np.zeros_like(img)
    # top
    mask[0:dty-1,:] = 1
    # left
    mask[:,0:dlx-1] = 1
    # bottom
    mask[-(dby-1):,:] = 1
    # right
    mask[:,-(drx-1):] = 1
    noice = np.random.normal(0, NOICE_STD, img.shape)
    img = img.astype(np.int32) + noice*mask
    img[img < 0] = 0

    return img.astype(np.uint8)


def new_array(coordinates):
    n = abs(coordinates[0]-coordinates[1])
    m = abs(coordinates[2]-coordinates[3])
    return np.zeros((n,m,3))


def find_frames(path_to_csv):

    frames = glob.glob(os.path.join(path_to_csv,"") + "*.csv")
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
        bb_dict[frame] = [bbox[0]]

    return bb_dict

def collect_path_dict(ground_path):

    all_folders = []
    for file in os.listdir(ground_path):
        if len(file) >= 33 and os.path.isdir(os.path.join(ground_path,file)):
            all_folders.append(os.path.join(ground_path,file))


    path_to_im = {}
    for folder in all_folders:
        files = glob.glob(os.path.join(folder,"") + "*.png")
        for file in files:
            path_to_im[os.path.join(folder, file)] = os.path.basename(file)

    return path_to_im, all_folders

def save_files(outdir, img_array):
    for idx, arr in enumerate(img_array):
        img = Image.fromarray(arr)
        img.save(os.path.join(outdir, str(idx)+".png"))

def save_video(paths, OUTDIR, video_name, path_to_im, bb_dict):
    img_array = []
    if len(paths) < 20:
        return None
    else:
        for filename in paths:
            img = cv2.imread(filename)

            img_n = combine_image_and_bbox(img, bb_dict[path_to_im[filename]])
            height, width, layers = img_n.shape
            size = (width, height)
            img_array.append(img_n)
        # print(f"\tFound and loaded {len(img_array)} images.")
        out = cv2.VideoWriter(f'{OUTDIR}{video_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
        # print(f"\tWriting to {OUTDIR}{video_name}.mp4")
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        # print("Thank you, next")

def save_structure(paths,path_to_im, bb_dict, collected_dict):
    if len(paths) < 20:
        return None
    else:
        mode_num = np.random.choice([1,2], prob = (0.8,0.2))
        if mode_num == 1:
            mode = 'train'
        else:
            mode = 'val'

        for filename in paths:
            collected_dict[mode][filename] = {}
            img = cv2.imread(filename)
            sizes, bbox = combine_image_and_bbox_into_rcnn_struct(img, bb_dict[path_to_im[filename]])
            collected_dict[mode][filename]['height'] = sizes[0]
            collected_dict[mode][filename]['width'] = sizes[1]
            collected_dict[mode][filename]['bbox'] = bbox

        return collected_dict


if __name__ == '__main__':

    ground_path = r'/scratch/s183993/placenta/raw_data/frames'
    OUTDIR = r'/scratch/s183993/placenta/raw_data/videos/videos_blackened_bbox'
    path_to_csv = ground_path
    paths_to_csv = find_frames(path_to_csv)

    bb_dict = {}
    for path in paths_to_csv:
        temp = collect_frames(path)
        bb_dict.update(collect_frames(path))

    path_to_im, all_folders = collect_path_dict(ground_path)
    path_list = [glob.glob(os.path.join(folder,"") + "*.png") for folder in all_folders]
    video_names = [folder.split(os.sep)[-1] for folder in all_folders]

    # collected_dict = {'train': {}, "val": {}}
    #
    # for idx, (paths, name) in enumerate(list(zip(path_list, video_names))):
    #     collected_dict = save_structure(paths, path_to_im, bb_dict,collected_dict)


    for idx, (paths, name) in enumerate(list(zip(path_list, video_names))):
        save_video(paths, OUTDIR, name, path_to_im, bb_dict)
        print("Succesfully printed for " + name)



