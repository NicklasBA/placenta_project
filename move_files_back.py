


import os
import numpy as np
import pandas
import shutil
import glob

def find_root_folders(parentdir):

    all_folders = [i for i in os.listdir(parentdir) if os.path.isdir(i)]
    mean_length = np.mean([len(i) for i in all_folders])

    old_folders = []
    new_folders = []
    for folder in all_folders:
        if len(folder) < mean_length:
            old_folders.append(folder)
        elif len(folder) > mean_length:
            new_folders.append(folder)

    folder_dict = {}
    for folder in old_folders:
        folder_dict[os.path.join(parentdir,folder)] = []
        for fold in new_folders:
            if folder in fold:
                folder_dict[folder].append(os.path.join(parentdir, fold))

    return folder_dict


def move_back(folder_dict):

    for key, item in folder_dict.items():
        for folder in item:
            files = glob.glob(os.path.join(folder,"*.png"))

            for file in files:
                dst = os.path.join(key, os.path.basename(file))
                shutil.move(file, dst)

        print("Succesfully moved all files for " + key)



if __name__ == '__main__':
    parentdir = r'/scratch/s183993/placenta/raw_data/frames'

    folder_dict = find_root_folders(parentdir)
    move_back(folder_dict)



