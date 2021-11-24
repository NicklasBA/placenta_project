


import pandas as pd
import os
import sys
import pickle
import numpy as np
import shutil
import glob
import cv2
#
# def alter_csv(csv):
#     csv[csv.columns[-1]] = [0 for _ in range(len(csv))]
#     return csv
#
# test_path = r'C:\Users\ptrkm\PycharmProjects\placenta_project\data\placenta\annotations\test_annotations.csv'
# train_path = r'C:\Users\ptrkm\PycharmProjects\placenta_project\data\placenta\annotations\train_annotations.csv'
# val_path = r'C:\Users\ptrkm\PycharmProjects\placenta_project\data\placenta\annotations\val_annotations.csv'
#
#
#
# train = pd.read_csv(train_path, header = None)
# val = pd.read_csv(val_path, header = None)
#
# # test = alter_csv(test)
# train = alter_csv(train)
# val = alter_csv(val)
# train.to_csv(train_path, header = False, index = False)
# val.to_csv(val_path, header = False, index = False)
# #
# test.to_csv(test_path, header = False, index = False)
# train.to_csv(train_path, header = False, index = False)
# val.to_csv(val_path, header = False, index = False)
#
# breakpoint()

# csv = pd.read_csv(r'C:\Users\ptrkm\PycharmProjects\20180404_5_6mbar_500fps_D167.csv')
# breakpoint()







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





def correct_csv(csv_path,data_path, anno = False):
    csv = pd.read_csv(os.path.join(data_path,csv_path))
    if 'Unnamed' in csv.columns[0]:
        csv = csv.drop(csv.columns[0], axis = 1)

    if anno is False:
        csv.to_csv(os.path.join(data_path,csv_path), sep =" ", index = False)
    else:
        csv.to_csv(os.path.join(data_path,csv_path), index = False, header = False)

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
# breakpoint()





def ensure_names(names, path):
    total_there =0
    for name in names:
        if os.path.isfile(os.path.join(path, name)):
            total_there += 1
        else:
            breakpoint()

    print(len(names)-total_there)

def move_files(old_csv, ground_path, count, total_frame_list, total_annotations,  train = False, test = False, val= False):

    g_name = old_csv.FrameName[0].split(".")[0][:-7]
    sequences = list(pd.unique(old_csv['SequenceID']))
    frame_list = pd.DataFrame()

    for seq in sequences:
        try:
            csv = old_csv[old_csv['SequenceID'] == seq]
            csv = csv.reset_index()
            new_path = g_name + "_" + seq
            frame_list = pd.DataFrame()
            frame_list['original_video_id'] = [new_path for _ in range(len(csv))]
            frame_list['video_id'] = [count for _ in range(len(csv))]
            frame_list['frame_id'] = csv['FrameNumber']
            frame_list['path'] = csv['FrameName'].apply(lambda x: os.path.join(new_path, x))
            ensure_names(list(frame_list['path']), ground_path)
            frame_list['labels'] = ['""' for _ in range(len(csv))]

            total_frame_list = pd.concat([total_frame_list,frame_list], ignore_index= True)
            count += 1

            annotations = pd.DataFrame()
            annotations['original_video_id'] = [new_path for _ in range(len(csv))]
            annotations['middle_frame_count'] = [i for i in range(len(csv))]
            annotations['x1'] = csv['x1A']
            annotations['y1'] = csv['y1A']
            annotations['x2'] = csv['x2A']
            annotations['y2'] = csv['y2A']
            annotations['label'] = [0 if 'D' in seq else 1 for _ in range(len(csv))]
            annotations['score'] = [0.0 for _ in range(len(csv))]

            total_annotations = pd.concat([total_annotations, annotations], ignore_index=True)
        except:
            breakpoint()

    return total_frame_list, total_annotations, count


def move_files_p2(csv_file, ground_path):
    sequences = list(pd.unique(csv_file['SequenceID']))
    moved = []
    for seq in sequences:
        csv = csv_file[csv_file['SequenceID'] == seq]
        csv = csv.reset_index()
        new_path = ground_path + "_" + seq
        if os.path.exists(new_path) is False:
            os.mkdir(new_path)


        paths = [os.path.join(new_path, x) for x in csv['FrameName']]
        old_path = [os.path.join(ground_path, x) for x in csv['FrameName']]
        moved.append(len(paths))
        for idx, p in enumerate(paths):
            src = old_path[idx]
            shutil.copy(src, p)

        print("Images for " + ground_path + seq + " Was moved succesfully")

    print("images for " + ground_path + " Was succesfully moved")
    print("A total of {} images were moved".format(np.sum(moved)))

def find_csv_and_ground_path(path_to_files):

    csv_files = [os.path.join(path_to_files, file) for file in os.listdir(path_to_files) if 'csv' in file]
    folders = [file.split(".")[0] for file in csv_files]
    ground_path = []
    for idx, file in enumerate(csv_files):
        ground_path.append(folders[idx])

    return ground_path

def find_files(data_dir, folder):
    all_folders = glob.glob(os.path.join(data_dir,folder)+"_*")
    files = []
    for folder in all_folders:
        files.append(glob.glob(os.path.join(folder, "*.png")))

    files = [i for file in files for i in file]
    endings = [int(i.split(".")[0][-6:]) for i in files]

    files = [x for _, x in sorted(zip(endings, files))]
    endings = sorted(endings)

    all_seq = []
    while i < len(endings) - 1:
        comb = []
        for idx in range(i, len(endings) - 1):
            if endings[idx] + 1 == endings[idx + 1]:
                comb.append(files[idx])
                i += 1
            else:
                i += 1
                break
        if len(comb) > 10:
            all_seq.append(comb)

    return all_seq

def save_video(paths, OUTDIR, video_name):
    img_array = []
    for filename in paths:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    print(f"\tFound and loaded {len(img_array)} images.")
    out = cv2.VideoWriter(f'{OUTDIR}{video_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    print(f"\tWriting to {OUTDIR}{video_name}.mp4")
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    # print("Thank you, next")

if __name__ == '__main__':
    path_to_files = r'/scratch/s183993/placenta/raw_data/datadump/'
    OUTDIR = r'/scratch/s183993/placenta/raw_data/videos/'
    folders = find_csv_and_ground_path(path_to_files)
    pcl = {}
    for folder in folders:
        base_name = os.path.basename(folder)
        all_seq = find_files(path_to_files, folder)
        pcl[folder]['lengths'] = []
        for idx, seq in enumerate(all_seq):
            pcl[folder]['lengths'].append(len(seq))
            save_video(seq,OUTDIR, base_name+f"_{idx:06}:")

        print(f"minimum length was {np.min(pcl[folder]['lengths'])}")
        print(f"maximum length was {np.max(pcl[folder]['lengths'])}")
        print(f"maximum length was {np.mean(pcl[folder]['lengths'])}")
        print("Done for " + base_name)

    with open(os.path.join(OUTDIR, 'lengths.pkl','wb')) as handle:
        pickle.dump(pcl, handle, protocol=pickle.HIGHEST_PROTOCOL)



#
# org_path = r'/scratch/s183993/placenta/raw_data/datadump/20180307_5_6mbar_500fps_D130'
# ground_path = r'/scratch/s183993/placenta/raw_data/datadump/'
# move_files_back(org_path, ground_path)

# path_to_files= r'/scratch/s183993/placenta/raw_data/datadump/'
#
# find_csv_and_ground_path(path_to_files)








#
# ground_path = r'/scratch/s183993/placenta/raw_data/frames'
# csv_files = [file for file in os.listdir(ground_path) if '.csv' in file]
#
# train = ['D128', 'D131', 'NS72','NS110','D204','NS112', 'D202']
# val = ['NS73', 'D203']
# test = ['D130','NS74','NS111']
#
# train_frame_list = pd.DataFrame()
# val_frame_list = pd.DataFrame()
# test_frame_list = pd.DataFrame()
# train_anno = pd.DataFrame()
# val_anno = pd.DataFrame()
# test_anno = pd.DataFrame()
# count = 0
# for csv_file in csv_files:
#     csv = pd.read_csv(os.path.join(ground_path,csv_file))
#     if csv_file.split(".")[0][-4:] in train:
#         train_frame_list, train_anno, count = move_files(csv, ground_path, count, train_frame_list, train_anno, train = True)
#     if csv_file.split(".")[0][-4:] in test:
#         test_frame_list, test_anno, count = move_files(csv, ground_path, count, test_frame_list, test_anno, test = True)
#     if csv_file.split(".")[0][-4:] in val:
#         val_frame_list, val_anno, count = move_files(csv, ground_path, count, val_frame_list, val_anno, val = True)
#
# save_path = r'/scratch/s183993/placenta/raw_data/'
#
# train_frame_list.to_csv(os.path.join(save_path, 'train.csv'))
# val_frame_list.to_csv(os.path.join(save_path, 'val.csv'))
# test_frame_list.to_csv(os.path.join(save_path, 'test.csv'))
#
# train_anno.to_csv(os.path.join(save_path, 'train_annotations.csv'))
# val_anno.to_csv(os.path.join(save_path, 'val_annotations.csv'))
# test_anno.to_csv(os.path.join(save_path, 'test_annotations.csv'))
# #
# #
# #
# # #
# #
# #
#
#
#
# #
