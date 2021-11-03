import glob
import os
import sys
from blob_analysis import BlobAnalysis
import argparse
import pandas as pd

import matplotlib.pyplot as plt

CATEGORY_D = 0
CATEGORY_NS = 1

def rename_files(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        file_ending = file.split(".")[0].split("_")[-1]
        if len(file_ending) < 6:
            file_end_new = "".join([str(0) for i in range(6-len(file_ending))]) + file_ending
            new_file = file.split(".")[0][:-len(file_ending)]+file_end_new + ".png"
            os.rename(os.path.join(folder_path,file), os.path.join(folder_path,new_file))

    print("File names were changed")

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
    new_sequences = []
    sequences = [seq for seq in sequences if len(seq)>= 10]
    endings = [[int(image_paths[i].split(".")[0].split("_")[-1]) for i in seq] for seq in sequences]

    combine = []
    i = 0

    while i < len(endings)-1:
        comb = sequences[i]
        for idx in range(i, len(endings)-1):
            if endings[idx][-1]+1 == endings[idx+1][0]:
                comb += sequences[idx+1]
                i+=1
            else:
                i+=1
                break
        new_sequences.append(comb)
    sequences = new_sequences

    for seq in sequences:
        seq_paths = [image_paths[i] for i in seq]
        seq_blobs = analyser.get_blobs_in_files(seq_paths)
        seq_names = [image_paths[i].rsplit(os.path.sep, 1)[-1] for i in seq]
        seq_bbox = analyser.get_bbox_if_valid_blob_seq(seq_blobs, names=seq_names)
        if seq_bbox:
            all_anno_seq.append(seq_bbox)

    all_anno_seq_ava = []
    for idx, seq in enumerate(all_anno_seq):
        all_anno_seq_ava.append({})
        for key in seq.keys():
            path = os.path.join(folder_path, key)
            bbox_list = analyser.ava_coordinate_change(path, seq[key])
            all_anno_seq_ava[idx][key] = bbox_list
    all_anno_seq = ensure_continuity(all_anno_seq)
    all_anno_seq_ava = ensure_continuity(all_anno_seq_ava)

    return all_anno_seq, sequences, all_anno_seq_ava

def ensure_continuity(all_anno_seq):
    new_all_anno = []
    for seq in all_anno_seq:
        files = list(seq.keys())
        endings = [int(i.split(".")[0].split("_")[-1]) for i in files]
        files = [x for _,x in sorted(zip(endings,files))]
        endings = sorted(endings)
        new_lists = [[]]
        count = 0
        for idx in range(len(endings)-1):
            new_lists[count].append(files[idx])
            if endings[idx] + 2 < endings[idx+1]:
                count+= 1
                new_lists.append([])

        new_lists = [sek for sek in new_lists if len(seq) > 10]
        for idx in range(len(new_lists)):
            dc = {name: seq[name] for name in new_lists[idx]}
            new_all_anno.append(dc)

    return new_all_anno


def create_annotations(folder_path: str):
    meas_id, category = find_id_and_class(folder_path)
    all_seq, sequences, ava_sequences = find_sequences(folder_path)

    annotations = pd.DataFrame(columns=["VideoID", "SequenceID", "FrameName", "FrameNumber", "BoundingBox","BoundingBoxAva",
                                        "x1","y1","x2","y2","x1A","y1A","x2A","y2A","Category"])

    print("Creating pretty formated annotations...")
    c = 0
    for anno, seq, ava_seq in zip(all_seq, sequences, ava_sequences):
        c += 1
        for items, frame in zip(anno.items(), seq):
            name, bboxes = items
            try:
                bboxes_ava = ava_seq[name]
            except:
                breakpoint()
            for bb, bb_ava in zip(bboxes, bboxes_ava):
                entry = {"VideoID": meas_id,
                         "SequenceID": f"{meas_id}_{c:04d}",
                         "FrameName": name,
                         "FrameNumber": frame,
                         "BoundingBox": bb,
                         "BoundingBoxAva": bb_ava,
                         "x1": bb[1],
                         "y1": bb[0],
                         "x2": bb[3],
                         "y2": bb[2],
                         "x1A": bb_ava[1],
                         "y1A": bb_ava[0],
                         "x2A": bb_ava[3],
                         "y2A": bb_ava[2],
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
        rename_files(folder_path=path)
        print(f"Starting analysis for {os.path.basename(os.path.abspath(path))} ({count}/{len(in_paths)})")
        processed_files[path] = create_annotations(path)
        count += 1

    print("Saving to csv...")
    for key, df in processed_files.items():
        df.to_csv(outdir+os.path.basename(os.path.abspath(key))+".csv")

    print(f"Created {len(processed_files)} csv files. Wrote to {outdir}")