"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 mask_rcnn.py train --dataset=/path/to/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 mask_rcnn.py train --dataset=/path/to/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 mask_rcnn.py train --dataset=/path/to/dataset --weights=imagenet
"""

import os
import sys
import random
import math
import re
import time
import numpy as np
import skimage
import datetime
import json
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import pickle
# Root directory of the project
ROOT_DIR = r'/home/s183993/placenta_project/Mask_RCNN/'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class PlacentaConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "placenta"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    #Must be devisibile by 2, 6 or mores times, hence we use default for now
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1024

    # Number of classes (including background)
    NUM_CLASSES = 2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 32

############################################################
#  Dataset
############################################################

class PlacentaDataset(utils.Dataset):
    """
    Overrides the Mask R-CNN implementation of dataset to include
    the format present for the placenta dataset
    """
    def load_rbc(self, dataset_dir, subset):
        """Load a subset of dataset.
                dataset_dir: Root directory of the dataset.
                subset: Subset to load: train, val or train
                """
        assert subset in ["train", "val", "test"]
        # Test actually not implemented, as we are only interested in val performance
        self.add_class("placenta", 1, "placenta")
        # Possible that the first rbc needs to be changed into proper source IDK :)

        annotations = pickle.load(open(os.path.join(dataset_dir, "mask_rcnn.pkl"),'rb'))
        annotations = annotations[subset]
        for file in annotations:
            image_path = file
            width = annotations[file]['width']
            height = annotations[file]['height']
            bbox = annotations[file]['bbox']
            self.add_image(
                "placenta",
                image_id=image_path,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                bbox=bbox,
                count=len(bbox)
            )
        

    def load_image(self, image_id):
        """
        Loads image
        """
        info = self.image_info[image_id]
        path = info['path']
        image = skimage.io.imread(path)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "placenta":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def fill_bbox(self, bbox, shape):
        m = np.zeros((shape[0],shape[1]))
        y1 = bbox[0]
        y2 = bbox[2]
        x1 = bbox[1]
        x2 = bbox[3]
        # assert y1 < y2
        # assert x1 < x2
        m[y1:y2, x1:x2] = 1
        return m.astype(np.uint8)


    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], info['count']], dtype = np.uint8)
        for i, bbox in enumerate(info['bbox']):
            mask[:, :, i] += self.fill_bbox(bbox, shape = (info['height'], info['width']))

        class_ids = np.array([1 for _ in range(info['count'])])
        return mask.astype(bool), class_ids.astype(np.int32)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PlacentaDataset()
    dataset_train.load_rbc(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PlacentaDataset()
    dataset_val.load_rbc(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument("--gpu",
                        help="GPU to run on",
                        default=0,
                        type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PlacentaConfig()
    else:
        class InferenceConfig(PlacentaConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

