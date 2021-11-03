import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing, binary_opening
from skimage import measure
from scipy.ndimage import gaussian_filter
import glob
from numpy.linalg import norm

# image_folder = sys.argv[1]
# txt_folder = sys.argv[1]
# n_points_in_median = sys.argv[2]

# image_folder = r'C:\Users\ptrkm\Action Classification\Images'
# txt_folder = r'C:\Users\ptrkm\Action Classification\Images'

INLET_POS = 50  # Horizontal position for inlet
OUTLET_POS = 740  # Horizontal position for outlet
MIN_BLOB_SIZE = 50  # Minimum number of pixels in an accepted blob
N_IMAGES_MEDIAN = 100  # Number of image files used for the median
SIGMA = 4

class Blob:
    def __init__(self, bbox, centroid):
        self.bbox = bbox
        self.centroid = np.array(centroid)
        self.keep = True
        self.get_radius()

    def get_radius(self):
        self.radius = abs(self.bbox[0] - self.bbox[2])

    def __sub__(self, blob):
        return norm(self.centroid - blob.centroid)

    def update_bbox(self, blob):
        min_row = np.min((self.bbox[0], blob.bbox[0]))
        min_col = np.min((self.bbox[1], blob.bbox[1]))
        max_row = np.min((self.bbox[2], blob.bbox[2]))
        max_col = np.min((self.bbox[3], blob.bbox[3]))
        self.bbox = (min_row, min_col,max_row, max_col)
        self.get_radius()
        blob.keep = False

def calculate_median(image_paths: list) -> np.ndarray:
    for idx, img_path in enumerate(image_paths):
        image = np.asarray(Image.open(img_path)).mean(axis=2)
        if idx == 0:
            total_array = np.zeros(image.shape + (len(image_paths),))
        total_array[:, :, idx] = gaussian_filter(image, sigma=SIGMA)

    median_image = np.median(total_array, axis=2)
    return median_image


def random_subset(folder: str, n_images_median=N_IMAGES_MEDIAN) -> list:
    rand_image_paths = os.listdir(folder)
    rand_image_paths = np.random.choice(rand_image_paths, n_images_median, replace=False).tolist()
    rand_image_paths = [os.path.join(folder, p) for p in rand_image_paths]
    return rand_image_paths


class BlobAnalysis:
    def __init__(self,
                 image_folder: str,
                 min_blob_size: int = MIN_BLOB_SIZE,
                 n_images_median: int = N_IMAGES_MEDIAN,
                 inlet: int = INLET_POS,
                 outlet: int = OUTLET_POS):

        self.image_folder = image_folder
        self.n_images_median = n_images_median
        self.min_blob_size = min_blob_size

        self.inlet = inlet
        self.outlet = outlet

        self.background_files = self.random_subset(image_folder)
        self.median = self.calculate_median(self.background_files)

    def get_blobs_in_files(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        blob_list = []
        for _, p in enumerate(paths):
            img = self.sub_median_and_bin(p)
            img = self.open_and_close(img)
            blobs = self.find_blobs(img)
            blob_list.append(blobs)
        return blob_list

    def ava_coordinate_change(self, path, bbox):
        img = np.asarray(Image.open(path))
        shapes = img.shape
        new_bboxes = []

        for b in bbox:
            bb = [0,0,0,0]
            bb[0] = b[0]/shapes[0]
            bb[1] = b[1]/shapes[1]
            bb[2] = b[2]/shapes[0]
            bb[3] = b[3]/shapes[1]
            new_bboxes.append(tuple(bb))

        return new_bboxes

    def get_bbox_if_valid_blob_seq(self, blob_seq, names=None):
        """Return bounding boxes for valid sequence"""
        if names:
            pass
        else:
            names = range(len(blob_seq))

        left, center, right = self.count_in_sequence(blob_seq)
        is_valid = True
        if is_valid:
            return {key: [v.bbox for v in val] for key, val in zip(names, blob_seq)}
        else:
            return None

    @staticmethod
    def check_rotated(blobs):
        """
        :param blobs: list of skimage measure objects
        :return: New bounding boxes
        """
        blobs = [Blob(bbox=blob.bbox, centroid=blob.centroid) for blob in blobs]

        for idx1, c1 in enumerate(blobs[:-1]):
            for idx2, c2 in enumerate(blobs[idx1+1:]):
                distance =  c1 - c2
                if distance < (c1.radius + c2.radius)/2 and c2.keep:
                    c1.update_bbox(c2)
                    break

        blobs = [blob for blob in blobs if blob.keep]
        return blobs

    def find_blobs(self, binary_image):
        blob_labels = measure.label(binary_image)
        blob_features = measure.regionprops(blob_labels)

        if blob_features:
            # blob_area = sorted([blob.area for blob in blob_features], reverse=True)

            blobs = [blob for blob in blob_features if blob.area >= self.min_blob_size]
            blobs = self.check_rotated(blobs)
            return blobs
        else:
            return None

    @staticmethod
    def valid_sequences(left, center, right):
        """Return true is sequence is valid"""
        valid = True
        for i in range(1, len(center) - 1):
            if center[i] > center[i - 1]:
                if left[i] >= left[i - 1]:
                    valid = False
            elif center[i] < center[i - 1]:
                if right[i] <= right[i - 1]:
                    valid = False

        return valid

    def count_in_sequence(self, list_of_blobs):
        left = []
        center = []
        right = []
        for blobs in list_of_blobs:
            l, c, r = self.count_in(blobs)
            left.append(l)
            center.append(c)
            right.append(r)

        return left, center, right

    def count_in(self, blobs):
        count_center = 0
        count_left = 0
        count_right = 0
        if not isinstance(blobs, type(None)):
            for b in blobs:
                if self.outlet >= b.centroid[1] >= self.inlet:
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
        image = gaussian_filter(image, sigma=SIGMA)
        image = image - self.median
        # threshold = threshold_otsu(image)
        binary_image = image > 2
        binary_image += image < -2

        return binary_image

    @staticmethod
    def calculate_median(image_paths: list):
        return calculate_median(image_paths)

    @staticmethod
    def random_subset(folder: str):
        return random_subset(folder)


if __name__ == '__main__':
    test = BlobAnalysis(image_folder, txt_folder, N_IMAGES_MEDIAN)
    imgs = [glob.glob(image_folder + "\*" + "D234_380" + str(i) + "*") for i in range(19, 68 - 19)]
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
    rect = patches.Rectangle((b1[1], b1[0]), b1[3] - b1[1], b1[2] - b1[0], linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((b2[1], b2[0]), b2[3] - b2[1], b2[2] - b2[0], linewidth=1, edgecolor='r',
                              facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.add_patch(rect2)
    plt.show()

    image_files = test.random_subset(folder=image_folder)
    median = test.calculate_median(image_files)
    tim = plt.imshow(np.asarray(Image.open(image_files[99])))
    plt.show()

    test_img = test.sub_median_and_bin(median=median, image_path=image_files[99])
    test_img = test.open_and_close(test_img)

    plt.imshow(test_img)
    plt.show()
