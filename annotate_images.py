import glob
import os
import sys
from blob_analysis import BlobAnalysis
import argparse
import pandas as pd

import matplotlib.pyplot as plt

CATEGORY_D = 0
CATEGORY_NS = 1


def find_sequences(folder_path: str):
    analyser = BlobAnalysis(folder_path)
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    print(f"A total of {len(image_paths)} images to process.")
    print("Finding blobs...")
    blobs = analyser.get_blobs_in_files(image_paths)
    print("Finding sequences...")
    _, c, _ = analyser.count_in_sequence(blobs)
    n_seqs = 0
    sequences = []
    i = 0
    while i < len(c):
        if c[i] == 0:
            i += 1
        else:
            sequences.append([])
            try:
                while c[i] > 0 and i < len(c):
                    sequences[n_seqs].append(i)
                    i += 1
            except IndexError:
                pass
            n_seqs += 1

    print("Splitting sequences and checking each of them...")
    all_anno_seq = []
    for seq in sequences:
        seq_blobs = [blobs[i] for i in seq]
        seq_names = [image_paths[i].rsplit(os.path.sep, 1)[-1] for i in seq]
        seq_bbox = analyser.get_bbox_if_valid_blob_seq(seq_blobs, names=seq_names)
        if seq_bbox:
            all_anno_seq.append(seq_bbox)

    return all_anno_seq, sequences

def create_annotations(folder_path: str):
    meas_id, category = find_id_and_class(folder_path)
    all_seq, sequences = find_sequences(folder_path)

    annotations = pd.DataFrame(columns=["VideoID", "SequenceID", "FrameName", "FrameNumber", "BoundingBox", "Category"])

    print("Creating pretty formated annotations...")
    c = 0
    for anno, seq in zip(all_seq, sequences):
        c += 1
        for items, frame in zip(anno.items(), seq):
            name, bboxes = items
            for bb in bboxes:
                entry = {"VideoID": meas_id,
                         "SequenceID": f"{meas_id}_{c:04d}",
                         "FrameName": name,
                         "FrameNumber": frame,
                         "BoundingBox": bb,
                         "Category": category}
                annotations = annotations.append(entry, ignore_index = True)

    return annotations

def find_id_and_class(name):
    name = name.split("_")
    meas_id, category = None, None
    for part in name:
        if "D" == part[0]:
            meas_id = part
            category = CATEGORY_D
        if "NS" == part[:2]:
            meas_id = part
            category = CATEGORY_NS

    if meas_id[-1] == os.path.sep:
        meas_id = meas_id[:-1]
    if meas_id[0] == os.path.sep:
        meas_id = meas_id[1:]

    return meas_id, category



def create(folder_path):
    pass


if __name__ == '__main__':
    # Stupid simple argument parser
    parser = argparse.ArgumentParser(description='Find sequences and annotate folders containing the image files')
    parser.add_argument('folder', help='Folder containing images to be annotated')
    parser.add_argument('-b', '--batch',
                        help='Description for bar argument',
                        action='store_true',
                        required=False)
    parser.add_argument('-o', '--output',
                        help='Output folder for annotation files (Default: Parent of input)',
                        required=False)
    args = vars(parser.parse_args())

    # Little input checking
    if not os.path.isdir(args["folder"]):
        raise NotADirectoryError(f"Input path {args['folder']} is not a directory")
    if args["output"]:
        if not os.path.isdir(args["output"]):
            raise NotADirectoryError(f"Output path {args['folder']} is not a directory")

    # Get all folders to process if it is a batch job
    if args["batch"]:
        in_paths = glob.glob(os.path.join(args["folder"], "*", ""))
        new_folders = []
        for i in reversed(range(len(in_paths))):
            sub_folders = glob.glob(os.path.join(in_paths[i], "*", ""))
            if sub_folders:
                in_paths.pop(i)
            new_folders += sub_folders
        in_paths += new_folders
        outdir = os.path.join(args["folder"], "")
    else:
        in_paths = [args["folder"]]
        outdir = os.path.join(args["folder"].rsplit(os.path.sep, 1)[0], "")

    processed_files = {}
    count = 1
    for path in in_paths:
        print(f"Starting analysis for {os.path.basename(os.path.abspath(path))} ({count}/{len(in_paths)})")
        processed_files[path] = create_annotations(path)
        count += 1

    print("Saving to csv...")
    for key, df in processed_files.items():
        df.to_csv(outdir+os.path.basename(os.path.abspath(key))+".csv")

    print(f"Created {len(processed_files)} csv files. Wrote to {outdir}")