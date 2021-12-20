


import os
import numpy as np
import pandas
import shutil
import glob

def find_root_folders(parentdir):

    all_folders = [i for i in os.listdir(parentdir)]
    mean_length = np.mean([len(i) for i in all_folders])

    old_folders = []
    new_folders = []
    for folder in all_folders:
        if len(folder) in [27,28, 29, 30, 31]:
            old_folders.append(folder)
        else:
            new_folders.append(folder)

    folder_dict = {}
    for folder in old_folders:
        folder_dict[os.path.join(parentdir,folder)] = []
        for fold in new_folders:
            if folder in fold:
                folder_dict[os.path.join(parentdir,folder)].append(os.path.join(parentdir, fold))

    return folder_dict


def move_back(folder_dict):

    for key, item in folder_dict.items():
        for folder in item:
            files = glob.glob(os.path.join(folder,"*.png"))

            for file in files:
                dst = os.path.join(key, os.path.basename(file))
                shutil.move(file, dst)

        print("Succesfully moved all files for " + key)


def ensure_moved(parentdir):

    folders = [os.path.join(parentdir, i) for i in os.listdir(parentdir)]
    mean_len = np.mean([len(i) for i in folders])

    folders_with_something_left = []

    for folder in folders:
        if len(folder) in [27,28,29,30,31]:
            print(f"Original folder containing {len(os.listdir(folder))}")
        else:
            files = os.listdir(folder)
            if len(files) > 0:
                folders_with_something_left.append(folder)

    print(f"There were {len(folders_with_something_left)} folders for which the files were not moved")
    print(f"These were {folders_with_something_left}")


if __name__ == '__main__':
    parentdir = r'/scratch/s183993/placenta/raw_data/frames'
    #
    folder_dict = find_root_folders(parentdir)
    move_back(folder_dict)
    ensure_moved(parentdir)



