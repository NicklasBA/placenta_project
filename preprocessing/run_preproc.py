import glob
import os
import sys
from blob_analysis import BlobAnalysis
import argparse
import pandas as pd
import anno_rcnn_helper as ah
import matplotlib.pyplot as plt
import cv2
from PIL.Image import Image
import shutil
import annotate_images as anno_im
CATEGORY_D = 0
CATEGORY_NS = 1
import combine_front_and_back as fb

def make_csv(cfg):
    """
    Creates the necessary csv file for forward compatibility with the Facebook's setup
    :param cfg: the configuration file (should be self evident by now what it is)
    :return: puts a csv file in the data folder, containing names and labels for testing
    """
    files = [os.path.join(cfg.DATA.PATH_TO_DATA_DIR, file) for file in os.listdir(cfg.DATA.PATH_TO_DATA_DIR)]

    if cfg.DATA.KNOWN_LABELS:
        labels = [CATEGORY_D if "D" in os.path.basename(file) else CATEGORY_NS for file in files]
    else:
        labels = [0 for _ in files]

    file = pd.DataFrame()
    file['path'] = files
    file['labels'] = labels

    file.to_csv(os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'test.csv'), header=None, index=None, sep=" ")
    print("Created the csv file for evaluation")


def run_preprocessing(cfg):
    """
    Wrapper function for running preprocessing, taking only the configuration file
    :param cfg: config file containing all relevant information
    :return: Saves the videos created by the preprocessing at the location cfg.DATA.PATH_TO_DATA_DIR
    """

    anno_im.finds_seqs(folder_path=cfg.DATA.IMAGE_DIR, OUTDIR=cfg.DATA.PATH_TO_DATA_DIR)
    make_csv(cfg)


def make_full_preproc(args):

    class PseudoCfg:
        def __init__(self, new = True):
            if new:
                self.DATA = PseudoCfg(new=False)

        def get(self,item, default = None):
            # Something to mimick the way that cfg notes work
            if item in self.__dir__():
                return self.__getattribute__(item)
            else:
                return default

    cfg = PseudoCfg()
    cfg.DATA.PATH_TO_DATA_DIR = args['path_to_data_dir']
    videos = [os.path.join(args['path_to_videos'], video) for video in os.listdir(args['path_to_videos'])]

    for video in videos:
        cfg = fb.split_video(video,cfg)
        anno_im.finds_seqs(folder_path=cfg.DATA.IMAGE_DIR, OUTDIR=cfg.DATA.PATH_TO_DATA_DIR)
        shutil.rmtree(cfg.DATA.IMAGE_DIR)



if __name__ == '__main__':
    """
    To run preprocessing for all videos in folder, all resulting videos are put in the same folder.
    """
    parser = argparse.ArgumentParser(description='Find sequences in videos')
    parser.add_argument('--path_to_data_dir', required=True)
    parser.add_argument('--path_to_videos', required=True)
    args = vars(parser.parse_args())
    args2 = parser.parse_args()
    breakpoint()
    make_full_preproc(args)
    argparse.Namespace




