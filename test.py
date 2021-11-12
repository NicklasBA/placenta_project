


import pandas as pd
import os
import sys
import pickle
import numpy as np
import shutil


data_path = r'/scratch/s183993/placenta/raw_data/videos'

def create_csv(path, mode):
    if mode == 'train':
        files = ['D128', 'D131', 'NS72','NS110','D204','NS112', 'D202']
    elif mode == 'val':
        files = ['NS73', 'D203']
    else:
        files = ['D130','NS74','NS111']

    doner_path = os.path.join(path, 'classD')
    navle_path = os.path.join(path, 'classNS')

    video_doner = []
    video_navle = []

    for file in files:
        for video in os.listdir(doner_path):
            if file in video:
                video_doner.append(video)

        for video in os.listdir(navle_path):
            if file in video:
                video_navle.append(video)
    csv = pd.DataFrame()

    paths = [os.path.abspath(file) for file in video_doner]
    paths += [os.path.abspath(file) for file in video_navle]

    labels = [0 for _ in range(len(video_doner))]
    labels += [1 for _ in range(len(video_navle))]

    csv['paths'] = paths
    csv['labels'] = labels

    csv.to_csv(os.path.join(path, "{}.csv".format(mode)), header = False, index = False, sep = " ")

create_csv(data_path, 'train')
create_csv(data_path, 'val')
create_csv(data_path, 'test')


#
#
# def correct_csv(csv_path,data_path, anno = False):
#     csv = pd.read_csv(os.path.join(data_path,csv_path))
#     if 'Unnamed' in csv.columns[0]:
#         csv = csv.drop(csv.columns[0], axis = 1)
#
#     if anno is False:
#         csv.to_csv(os.path.join(data_path,csv_path), sep =" ", index = False)
#     else:
#         csv.to_csv(os.path.join(data_path,csv_path), index = False, header = False)
#
# data_path  = r'/scratch/s183993/placenta/raw_data/'
# train_frame = r'train.csv'
# train_anno = r'train_annotations.csv'
# val_frame = r'val.csv'
# val_anno = r'val_annotations.csv'
# test_frame = r'test.csv'
# test_anno = r'test_annotations.csv'
#
# correct_csv(train_frame, data_path)
# correct_csv(val_frame, data_path)
# correct_csv(test_frame, data_path)
# correct_csv(train_anno, data_path,True)
# correct_csv(val_anno, data_path, True)
# correct_csv(test_anno, data_path, True)
# # breakpoint()
# # def ensure_names(names, path):
# #     total_there =0
# #     for name in names:
# #         if os.path.isfile(os.path.join(path, name)):
# #             total_there += 1
# #         else:
# #             breakpoint()
# #
# #     print(len(names)-total_there)
# #
# # def move_files(old_csv, ground_path, count, total_frame_list, total_annotations,  train = False, test = False, val= False):
# #
# #     g_name = old_csv.FrameName[0].split(".")[0][:-7]
# #     sequences = list(pd.unique(old_csv['SequenceID']))
# #     frame_list = pd.DataFrame()
# #
# #     for seq in sequences:
# #         try:
# #             csv = old_csv[old_csv['SequenceID'] == seq]
# #             csv = csv.reset_index()
# #             new_path = g_name + "_" + seq
# #             frame_list = pd.DataFrame()
# #             frame_list['original_video_id'] = [new_path for _ in range(len(csv))]
# #             frame_list['video_id'] = [count for _ in range(len(csv))]
# #             frame_list['frame_id'] = csv['FrameNumber']
# #             frame_list['path'] = csv['FrameName'].apply(lambda x: os.path.join(new_path, x))
# #             ensure_names(list(frame_list['path']), ground_path)
# #             frame_list['labels'] = ['""' for _ in range(len(csv))]
# #
# #             total_frame_list = pd.concat([total_frame_list,frame_list], ignore_index= True)
# #             count += 1
# #
# #             annotations = pd.DataFrame()
# #             annotations['original_video_id'] = [new_path for _ in range(len(csv))]
# #             annotations['middle_frame_count'] = [i for i in range(len(csv))]
# #             annotations['x1'] = csv['x1A']
# #             annotations['y1'] = csv['y1A']
# #             annotations['x2'] = csv['x2A']
# #             annotations['y2'] = csv['y2A']
# #             annotations['score'] = [0.0 for _ in range(len(csv))]
# #
# #             total_annotations = pd.concat([total_annotations, annotations], ignore_index=True)
# #         except:
# #             breakpoint()
# #
# #     return total_frame_list, total_annotations, count
# #
# #
# # ground_path = r'/scratch/s183993/placenta/raw_data/frames'
# # csv_files = [file for file in os.listdir(ground_path) if '.csv' in file]
# #
# # train = ['D128', 'D131', 'NS72','NS110','D204','NS112', 'D202']
# # val = ['NS73', 'D203']
# # test = ['D130','NS74','NS111']
# #
# # train_frame_list = pd.DataFrame()
# # val_frame_list = pd.DataFrame()
# # test_frame_list = pd.DataFrame()
# # train_anno = pd.DataFrame()
# # val_anno = pd.DataFrame()
# # test_anno = pd.DataFrame()
# # count = 0
# # for csv_file in csv_files:
# #     csv = pd.read_csv(os.path.join(ground_path,csv_file))
# #     if csv_file.split(".")[0][-4:] in train:
# #         train_frame_list, train_anno, count = move_files(csv, ground_path, count, train_frame_list, train_anno, train = True)
# #     if csv_file.split(".")[0][-4:] in test:
# #         test_frame_list, test_anno, count = move_files(csv, ground_path, count, test_frame_list, test_anno, test = True)
# #     if csv_file.split(".")[0][-4:] in val:
# #         val_frame_list, val_anno, count = move_files(csv, ground_path, count, val_frame_list, val_anno, val = True)
# #
# # save_path = r'/scratch/s183993/placenta/raw_data/'
# #
# # train_frame_list.to_csv(os.path.join(save_path, 'train.csv'))
# # val_frame_list.to_csv(os.path.join(save_path, 'val.csv'))
# # test_frame_list.to_csv(os.path.join(save_path, 'test.csv'))
# #
# # train_anno.to_csv(os.path.join(save_path, 'train_annotations.csv'))
# # val_anno.to_csv(os.path.join(save_path, 'val_annotations.csv'))
# # test_anno.to_csv(os.path.join(save_path, 'test_annotations.csv'))
# # #
# # #
# # #
# # #
# #
# #
#
#
#
#