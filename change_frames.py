
import pandas as pd
import numpy as np
import os
import glob
import shutil


def change_csv(csv):

    files = list(csv[csv.columns[0]])
    diag = list(csv[csv.columns[1]])
    new_files = []
    new_diag = []
    for diag,  file in zip(diag, files):
        if os.path.exists(file):
            new_files.append(file)
            new_diag.append(diag)

    new_csv = pd.DataFrame()
    new_csv['path'] = new_files
    new_csv['diag'] = new_diag

    return new_csv

def save_csv(csv,save_path):
    csv.to_csv(save_path, header=False, index=False, sep = " ")
    print("Saved " + save_path + " succesfully")

if __name__ == '__main__':
    ground_path = r'/scratch/s183993/placenta/raw_data/videos'

    train = pd.read_csv(os.path.join(ground_path, 'train.csv'), header = None, sep = " ")
    val = pd.read_csv(os.path.join(ground_path,'val.csv'), header = None, sep = " ")
    test = pd.read_csv(os.path.join(ground_path,'test.csv'), header = None, sep = " ")

    train = change_csv(train)
    val = change_csv(val)
    test = change_csv(test)

    save_csv(csv = train, save_path=os.path.join(ground_path, 'train.csv'))
    save_csv(csv = val, save_path=os.path.join(ground_path, 'val.csv'))
    save_csv(csv=test, save_path=os.path.join(ground_path, 'test.csv'))