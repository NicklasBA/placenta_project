

import sys
import os
import pandas as pd
import numpy as np
import pickle
from numpy.linalg import norm



def find_csv_files(path, train, val, test):
    csv_files = {file.rsplit(os.sep)[-1].split(".")[0]:
                     os.path.join(path, file) for file in os.listdir(path)
                 if '.csv' in file}

    train_dict = {}
    val_dict = {}
    test_dict = {}
    for doner in train:
        for folder, path in csv_files.items():
            if doner in folder:
                train_dict[folder] = path
    for doner in val:
        for folder, path in csv_files.items():
            if doner in folder:
                val_dict[folder] = path

    for doner in test:
        for folder, path in csv_files.items():
            if doner in folder:
                test_dict[folder] = path

    return train_dict,val_dict,test_dict


def create_annotations_and_frames(csv_dictionary):

    count = 0
    for key, path in csv_dictionary.items():

        current_csv = pd.read_csv(path)
        if 'Unnamed' in current_csv.columns[0]:
            current_csv = current_csv.drop(current_csv.columns[0],axis=1)
        sequences = list(pd.unique(current_csv['SequenceID']))

        for seq in sequences:
            csv = current_csv[current_csv['SequenceID'] == seq]
            csv = csv.reset_index()
            if count == 0:
                frame_list = pd.DataFrame()
                frame_list['original_video_id'] = [key + "_"+seq.split("_")[-1] for _ in range(len(csv))]
                frame_list['video_id'] = count
                frame_list['frame_id'] = csv['FrameNumber']
                frame_list['path'] = csv['FrameName'].apply(lambda x: key + "_" + seq.split("_")[-1] + os.sep + x)
                frame_list['labels'] = ['""' for _ in range(len(csv))]
                count+=1

                annotations = pd.DataFrame()
                annotations['original_video_id'] = [key + "_" + seq.split("_")[-1] for _ in range(len(csv))]
                annotations['middle_frame_count'] = [i for i in range(len(csv))]
                annotations['x1'] = csv['x1A']
                annotations['y1'] = csv['y1A']
                annotations['x2'] = csv['x2A']
                annotations['y2'] = csv['y2A']
                annotations['score'] = [0.0 for _ in range(len(csv))]
            else:
                new_frame_list = create_frame_dataframe(csv,seq,key,count)
                frame_list = pd.concat([frame_list, new_frame_list], ignore_index=True)

                new_annotations = create_annotations_dataframe(csv,key,seq)
                annotations = pd.concat([annotations, new_annotations], ignore_index = True)
                count+=1

    return frame_list, annotations


def create_frame_dataframe(csv, seq, key,count):
    frame_list = pd.DataFrame()
    frame_list['original_video_id'] = [key + "_" + seq.split("_")[-1] for _ in range(len(csv))]
    frame_list['video_id'] = count
    frame_list['frame_id'] = csv['FrameNumber']
    frame_list['path'] = csv['FrameName'].apply(lambda x: key + "_" + seq.split("_")[-1] + os.sep + x)
    frame_list['labels'] = ['""' for _ in range(len(csv))]

    return frame_list

def create_annotations_dataframe(csv, key, seq):
    annotations = pd.DataFrame()
    annotations['original_video_id'] = [key + "_" + seq.split("_")[-1] for _ in range(len(csv))]
    annotations['middle_frame_count'] = [i for i in range(len(csv))]
    annotations['x1'] = csv['x1A']
    annotations['y1'] = csv['y1A']
    annotations['x2'] = csv['x2A']
    annotations['y2'] = csv['y2A']
    annotations['score'] = [0.0 for _ in range(len(csv))]

    return annotations




path = r'/scratch/s183993/placenta/raw_data/frames'

train = ['D128', 'D131', 'NS72','NS110','D204','NS112', 'D202']
val = ['NS73', 'D203']
test = ['D130','NS74','NS111']

train, val, test = find_csv_files(path, train, val, test)

train_frame, train_annot = create_annotations_and_frames(train)

val_frame, val_annot = create_annotations_and_frames(train)
test_frame, test_annot = create_annotations_and_frames(train)

save_path = os.path.dirname(path)

train_frame.to_csv(os.path.join(save_path, 'train_frame.csv'), index = None, sep = " ")
val_frame.to_csv(os.path.join(save_path, 'val_frame.csv'), index = None, sep = " ")
test_frame.to_csv(os.path.join(save_path, 'test_frame.csv'), index = None, sep = " ")


train_annot.to_csv(os.path.join(save_path, 'annotations_train.csv'),index = None, header = False, sep = " ")
val_annot.to_csv(os.path.join(save_path, 'annotations_val.csv'),index = None, header = False, sep = " ")
test_annot.to_csv(os.path.join(save_path, 'annotations_test.csv'), index = None,header = False, sep = " ")


