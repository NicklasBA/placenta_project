import cv2
import glob
import os
import create_mask as cm

class BoundingBox:
    def __init__(self, bbox):
        self.identifier = None
        self.bbox = bbox

def save_video(paths, bboxs, OUTDIR, video_name):
    """

    :param paths: (list) of paths to image files
    :param bboxs: (list) of bounding boxes
    :param OUTDIR: (path) to where the videos should be saved
    :param video_name: (str) and self evident
    :return: None but saves a video with video_name
    """
    img_array = []
    if len(paths) < 20:
        return None
    else:
        for bbox, filename in list(zip(bboxs, paths)):
            img = cv2.imread(filename)

            img_n = cm.combine_image_and_bbox(img, bbox)
            height, width, layers = img_n.shape
            size = (width, height)
            img_array.append(img_n)
        # print(f"\tFound and loaded {len(img_array)} images.")
        out = cv2.VideoWriter(f'{OUTDIR}{video_name}.avi', cv2.VideoWriter_fourcc(*'HFYU'), 15, size)
        print(f"\tWriting to {OUTDIR}{video_name}.avi")
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        # print("Thank you, next")

def run_through_folder(folder, evals, OUTDIR):
    """

    :param folder: Path to images
    :param evals: (dict) evaluation of every image in shape eval = {"path/to/im/": {"bbox": [list of bbox],
                                                                                    "count": len([list_of_bbox])}}
    :param OUTDIR: Path to were videos should be put
    :return:
    """
    collected, hashes, hash_to_file = create_identifiers_for_folder(folder, evals)

    for h in hashes:
        files = hash_to_file[h]
        files, endings = extract_files_and_endings(files)
        bboxs = []
        for file in files:
            for bbox in collected[file]:
                if bbox.identifier == h:
                    bboxs.append(bbox.bbox)

        assert len(files) == len(bboxs)
        video_name = str(h)
        save_video(files,bboxs, OUTDIR, video_name)


def create_identifiers_for_folder(folder, evals):
    """

    :param folder: Path to images
    :param evals: (dict) evaluation of every image in shape eval = {"path/to/im/": {"bbox": [list of bbox],
                                                                                    "count": len([list_of_bbox])}}
    :return: (dict) containing BoundingBox objects in every image {"path/to/im/": [BoundingBox,...]
            (list) all integer identifiers
            (dict) identifier to file {identifier: ["path/to/im", ...]
    """

    files = glob.glob(os.path.join(folder, '*.png'))
    files, endings = extract_files_and_endings(files)

    collected = {}
    hash_to_file = {}
    all_hashes = [-1]


    for key, val in evals.items():
        collected[key] = [BoundingBox(i) for i in val["bbox"]]

    for idx, file in enumerate(files):
        bboxes = collected[file]
        subs = []
        new_hashes = []
        for i in range(len(bboxes)):
            if collected[file][i].identifier is None:
                nh = all_hashes[-1] + i + 1
                new_hashes.append(nh)
                collected[file][i].identifier = nh
                subs.append(i)

        for nh in new_hashes:
            hash_to_file[nh] = []
            hash_to_file[nh].append(file)

        all_hashes += new_hashes
        bboxes = [bboxes[i] for i in subs]

        for i, f in enumerate(files[idx+1:]):
            new_bboxes = collected[f]

            subset = []
            for j in range(len(bboxes)):
                index = identify_bbox(bboxes[j],new_bboxes)
                if index != -1:
                    collected[f][index].identifier = collected[file][j].identifier
                    hash_to_file[collected[f][index].identifier].append(f)
                    subset.append(index)

            bboxes = [new_bboxes[i] for i in subset]
            if len(bboxes) == 0:
                break

    return collected, all_hashes, hash_to_file

def extract_files_and_endings(files):
    """

    :param files: List of image files
    :return: files sorted by endings along with the endings.
    """
    endings = []
    for file in files:
        ending = file.split(".")[0][-6:]
        endings.append(int(ending))

    files = [file for _, file in sorted(zip(endings, files))]
    endings = sorted(endings)

    return files, endings

def identify_bbox(cbox, new_boxes):
    """

    :param cbox: Bounding box of current image (list or tuple)
    :param new_boxes: bounding boxes in next image (list of lists) or (list of tuples)
    :return: the index in new_boxes that the current bbox is identified with, -1 if isn't identified with any
    """

    for idx, nbox in enumerate(new_boxes):
        ins = inside(cbox, new_boxes)
        if ins:
            return idx

    return -1

def center(bbox):
    """
    :param bbox: (list or tuple) of bbox coordinates on the form (y1, x1, y2, x2)
    :return: center coordinates of bbox (y_c, x_c)
    """

    assert bbox[0] < bbox[2] and bbox[1] < bbox[3]

    row_c = int((bbox[0] + bbox[2])//2)
    col_c = int((bbox[1] + bbox[3])//2)

    return (row_c, col_c)

def inside(bbox1, bbox2):
    """

    :param bbox1: Current bounding box
    :param bbox2: Box to check if inside bbox1
    :return: (bool) True if bbox2 is inside bbox1 otherwise False
    """

    c2 = center(bbox2)

    if bbox1[0] <= c2[0] <= bbox1[2] and bbox1[1] <= c2[1] <= bbox1[3]:
        return True
    else:
        return False
