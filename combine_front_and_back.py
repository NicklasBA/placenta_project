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
import preprocessing.run_preproc as preproc
import shutil
sys.path.append(r"C:\Users\ptrkm\PycharmProjects\placenta_project\SlowFast")
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job, write_results
from slowfast.utils.parser import load_config, parse_args
CATEGORY_D = 0
CATEGORY_NS = 1


def create_config_from_args(args):
    """
    Function to update the config file with the user defined arguments
    :param args: user arguments, should default contain path to default configuration file
    :return: config object as they are used in facebook's
    """

    if isinstance(args, dict):
        nargs = {'cfg',args['cfg']}
        nargs = argparse.Namespace(**nargs)
    elif isinstance(args, argparse.Namespace):
        nargs = {'cfg', args.cfg}
        args = vars(args)
    else:
        raise NotImplementedError(f"Not implemented yet for arguments not of type {type(args)}")

    cfg = load_config(nargs)
    cfg = assert_and_infer_cfg(cfg)
    cfg = insert_items(cfg, args)

    return cfg

def insert_items(cfg, args):
    """
     Method of inserting keys and values into the larger cfg
    :param cfg:
    :param args:
    :return:
    """

    for key, val in args.items():
        if cfg.get(key, None) is not None:
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, dict):
                        raise NotImplementedError("Not implemented yet for two levels of inserting,"
                                                  " should be changed to recursive function then")
                    else:
                        cfg[key][k] = v
            else:
                cfg[key] = val

    return cfg


def create_name(base, idx):
    """

    :param base: basename of the video
    :param idx: count of the frame
    :return: a name for the frame (This function should be replaced with a oneliner, i was just on a plane when i
    wrote this code, so i couldn't look up how to do it)
    """

    num = "".join([str(0) for _ in range(6 - len(str(idx)))]) + str(idx)
    name = base + "_" + num + ".png"
    return name

def split_video(path_to_video, cfg):
    """

    :param path_to_video: From the drag and drop functionality in the front end, passing a path to the video location
    :param cfg: the config file for further evaluation
    :return: Creates a folder in cfg.PATH_TO_DATADIR containing all images
    """


    if cfg.DATA.get('PATH_TO_DATA_DIR', None) is None:
        raise ValueError("There should be a specified path to data directory in the configuration file")
    else:
        image_dir = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'temp')
        os.mkdir(image_dir)
        cfg.DATA.IMAGE_DIR = image_dir
        video_name = os.path.basename(path_to_video)

        cap = cv2.VideoCapture(path_to_video)
        if cap.isOpened()==False:
            print("Something went wrong in the reading of the video")

        count = 0
        while True:
            ret, frame = cap.read()
            if ret:
                save_name = create_name(video_name, count)
                cv2.imwrite(os.path.join(image_dir, save_name), frame)
            else:
                break

        cap.release()

    return cfg


def prepare_and_delete_data(cfg):
    """
    Semi wrapper function for making the preproccessing and subsequently deleting the image data
    :param cfg:
    :return:
    """

    preproc.run_preprocessing(cfg)
    shutil.rmtree(cfg.DATA.IMAGE_DIR)








