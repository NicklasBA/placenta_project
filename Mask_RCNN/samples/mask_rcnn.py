import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.model import log
import pickle

"""
Er begyndt ud fra train_shapes.py, men opdagede balloon.py allerede har gjort det 
meste af arbejdet for os, så man kunne evt starte fra den i stedet.
Funktionerne load_image, load_mask, image reference og load_balloon (måske load_rbc) skal bare ændres, og 
så skal der sættes et par hyperparametre
"""



class PlacentaConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "placenta"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 150
    IMAGE_MAX_DIM = 800

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class InferenceConfig(PlacentaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

class PlacentaDataset(utils.Dataset):
    """
    Overrides the Mask R-CNN implementation of dataset to include
    the format present for the placenta dataset
    """
    def load_rbc(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
                dataset_dir: Root directory of the dataset.
                subset: Subset to load: train, val or train
                """
        assert subset in ["train", "val", "test"]
        # Test actually not implemented, as we are only interested in val performance
        self.add_class("rbc", 1, "rbc")
        # Possible that the first rbc needs to be changed into proper source IDK :)

        annotations = pickle.load(open(os.path.join(dataset_dir, "mask_rcnn.pkl"),'rb'))
        annotations = annotations[subset]

        for file in annotations:
            image_path = file
            width = file['width']
            height = file['height']
            bbox = file['masks']

            self.add_image(
                "balloon",
                image_id=image_path,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                bbox=bbox,
                count=len(bbox)

            )

    def load_image(self, filename):
        """
        Loads image
        """
        image = cv2.imread(filename)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "rbc":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def fill_bbox(self, bbox, shape):
        m = np.zeros((shape[0],shape[1]))
        y1 = bbox[0]
        y2 = bbox[2]
        x1 = bbox[1]
        x2 = bbox[3]
        assert y1 < y2
        assert x1 < x2
        m[y1:y2, x1:x2] = 1
        return m

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], info['count']], dtype = np.uint8)
        for i, bbox in enumerate(info['bbox']):
            mask[:, :, i] += self.fill_bbox(bbox, shape = (info['height'], info['width']))

        class_ids = np.array([1 for _ in range(info['count'])])
        """
        Quite certain that we dont need the stuff below, ours is far more simple
        """
        # shapes = info['shapes']
        # count = len(shapes)
        # mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        # for i, (shape, _, dims) in enumerate(info['shapes']):
        #     mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
        #                                         shape, dims, 1)
        # # Handle occlusions
        # occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        # for i in range(count-2, -1, -1):
        #     mask[:, :, i] = mask[:, :, i] * occlusion
        #     occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # # Map class names to class IDs.
        # class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)




def detection(MODEL_DIR, dataset_train, dataset_val):
    inference_config = InferenceConfig()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))

    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax())


def main():

    # Root directory of the project
    ROOT_DIR = r'/home/s183993/placenta_project/Mask_RCNN/'

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    config = PlacentaConfig()
    config.display()

    # Training dataset
    dataset_train = PlacentaDataset()
    dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PlacentaDataset()
    dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()

    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2,
                layers="all")

    #detection(MODEL_DIR, dataset_train, dataset_val)

if __name__ == "main":
    main()
