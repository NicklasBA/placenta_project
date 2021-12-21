
import pandas as pd
import os
import numpy as np
import pickle
import glob
import shutil

SAVE_NAME = r"/home/s183993/placenta_project/Mask_RCNN/mask_rcnn_p2.pkl"



def extract_ground_path(pstr):
    """

    :param pstr: Path string to get ground path from
    :return: ground path
    """

    name = "_".join(pstr.split(".")[0].split("/")[-1].split("_")[:-1])
    ground_path = os.sep.join(pstr.split("/")[:-2])
    ending = pstr.split(os.sep)[-1]

    return os.path.join(os.path.join(ground_path, name), ending)

def ensure_present(path_new, path_pcl):
    """

    :param path_new: Path to check if is in old files
    :param path_pcl: dict of old files
    :return:
    """

    present = path_pcl.get(path_new, False)
    return present

def run_through_pkl(pcl, path_pcl):

    new_pcl = {}
    non_present = 0
    for key in pcl.keys():
        new_pcl[key] = {}
        npcl = pcl[key]
        for key, val in npcl.items():
            ground_path = extract_ground_path(key)
            present = ensure_present(ground_path, path_pcl)
            if present:
                new_pcl[ground_path] = val
            else:
                non_present += 1

        print(f"{non_present} images were not found")

    with open(SAVE_NAME, "wb") as handle:
        pickle.dump(new_pcl, handle, protocol=4)


def create_path_pcl(parentdir):

    folders = [os.path.join(parentdir, i) for i in os.listdir(parentdir)]
    path_pcl = {}

    for folder in folders:
        files = glob.glob(os.path.join(folder, "*.png"))
        for file in files:
            path_pcl[file] = True

    return path_pcl

if __name__ == '__main__':

    parentdir = r'/scratch/s183993/placenta/raw_data/frames/'

    path_pcl = create_path_pcl(parentdir)
    pcl = pickle.load(open(r'/home/s183993/placenta_project/Mask_RCNN/mask_rcnn.pkl','rb'))
    run_through_pkl(pcl,path_pcl)