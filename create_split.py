import os
import pandas as pd
import shutil
import numpy as np
import pickle
import random
import re
import shutil
def create_split(path):
    files = os.listdir(path)
    files = [file for file in files if '.avi' in file or '.mp4' in file]
    random.shuffle(files)
    train = []
    train_lab = []

    val = []
    val_lab = []

    test_lab = []
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


def rename_files(parentdir,outdir):

    if os.path.isdir(outdir) is False:
        os.mkdir(outdir)

    folders = [os.path.join(parentdir, i) for i in os.listdir(parentdir) if os.path.isdir(os.path.join(parentdir, i))]


    doner = {}
    fetal = {}

    for folder in folders:

        if "p2" not in folder and ".zip" not in folder:
            if "NS" in folder:
                for file in os.listdir(folder):
                    try:
                        name = re.search(r'(?<=NS)\w+', folder)
                        fetal[os.path.join(folder, file)] = "NS" + name.group(0)
                    except:
                        breakpoint()
            else:
                for file in os.listdir(folder):
                    try:
                        name = re.search(r'(?<=D)\w+', folder)
                        doner[os.path.join(folder, file)] = "D" + name.group(0)
                    except:
                        breakpoint()

    for idx, file in enumerate(list(doner.keys())):
        doner[file] = os.path.join(outdir, doner[file] + "_" + str(idx) + ".avi")
    for idx, file in enumerate(list(fetal.keys())):
        fetal[file] = os.path.join(outdir, fetal[file] + "_" + str(idx) + ".avi")

    move_files_in_dict(doner)
    move_files_in_dict(fetal)

def move_files_in_dict(dict):
    count = 0
    for key, val in dict.items():
        shutil.copy(key, val)
        count+=1

    print(f"Copied files in dictionary, {count} files were placed in new folder")



if __name__ == '__main__':
    parentdir = r'/scratch/s183993/videos/'
    outdir = r'/scratch/s183993/videos_all/'
    rename_files(parentdir, outdir)

    t, v, te = create_split(outdir)

    t.to_csv(r'/scratch/s183993/train.csv', header = False,index = False, sep = " ")
    v.to_csv(r'/scratch/s183993/val.csv', header = False,index = False, sep = " ")
    te.to_csv(r'/scratch/s183993/test.csv', header=False, index= False, sep=" ")
