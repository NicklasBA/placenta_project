
import os
import pandas as pd
import shutil
import numpy as np
import pickle

def create_split(path):
    files = os.listdir(path)

    train = []
    train_lab = []

    val = []
    val_lab = []

    test_lab =[]
    test = []

    for file in files:
        mode = np.random.choice([1, 2, 3], p=(0.7, 0.2, 0.1))
        if 'NS' in file:
            if mode == 1:
                train.append(os.path.join(path, file))
                train_lab.append(1)
            elif mode == 2:
                val.append(os.path.join(path,file))
                val_lab.append(1)
            else:
                test.append(os.path.join(path,file))
                test_lab.append(1)
        else:
            if mode == 1:
                train.append(os.path.join(path, file))
                train_lab.append(0)
            elif mode == 2:
                val.append(os.path.join(path,file))
                val_lab.append(0)
            else:
                test.append(os.path.join(path,file))
                test_lab.append(0)

    train_pd = pd.DataFrame()
    val_pd = pd.DataFrame()
    test_pd = pd.DataFrame()

    train_pd = insert_into_frame(train_pd, train, train_lab)
    val_pd = insert_into_frame(val_pd, val, val_lab)
    test_pd = insert_into_frame(test_pd, test, test_lab)

    num_ns = np.sum(train_lab)
    print(f"{num_ns} out of {len(train)} was NS in train")
    num_ns = np.sum(val_lab)
    print(f"{num_ns} out of {len(val)} was NS in val ")
    num_ns = np.sum(test_lab)
    print(f"{num_ns} out of {len(test)} was NS in test")


    return train_pd, val_pd, test_pd


def insert_into_frame(frame, names, lab):
    frame['names'] = names
    frame['labels'] = lab
    return frame

if __name__ == '__main__':
    path = r'/scratch/s183993/placenta/raw_data/videos_blackened_org_bbox_full'

    t_org = pd.read_csv(r'/scratch/s183993/placenta/raw_data/videos_blackened_org_bbox/train.csv', header = None, names = ['names','labels'], sep = " ")
    val_org = pd.read_csv(r'/scratch/s183993/placenta/raw_data/videos_blackened_org_bbox/val.csv', header = None, names = ['names','labels'],sep = " ")
    test_org = pd.read_csv(r'/scratch/s183993/placenta/raw_data/videos_blackened_org_bbox/test.csv', header = None, names = ['names','labels'],sep = " ")



    t, v, te = create_split(path)

    t_org = pd.concat((t_org, t), axis = 0)
    val_org = pd.concat((val_org, v), axis =0)
    test_org = pd.concat((test_org, te), axis = 0)

    t_org.to_csv(r'/scratch/s183993/placenta/raw_data/videos_blackened_org_bbox/train.csv', header = False,index = False, sep = " ")
    val_org.to_csv(r'/scratch/s183993/placenta/raw_data/videos_blackened_org_bbox/val.csv', header = False,index = False, sep = " ")
    test_org.to_csv(r'/scratch/s183993/placenta/raw_data/videos_blackened_org_bbox/train.csv', header=False, index= False,
                 sep=" ")
