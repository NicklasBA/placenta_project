import numpy as np
import pandas as pd
import os
import sys
import argparse


HIT = np.array([0.7, 0.2, 0.1])



def get_folders_and_split(ground_path, designation, iter = 100):
    """

    :param ground_path: Directory containing all subdirectories containing the videos
    :param designation: Whether fetal (NS) or Doner (D)
    :return: indices for the three splits
    """

    sub_dirs = [os.path.join(ground_path, dir) for dir in os.listdir(ground_path)]

    diff = 100

    for i in range(iter):
        train = []
        test = []
        val = []
        for idx, file in enumerate(sub_dirs):
            if designation in file:
                mode = np.random.choice([1, 2, 3], p=(0.7, 0.2, 0.1))
                if mode == 1:
                    train.append(idx)
                elif mode == 2:
                    val.append(idx)
                elif mode == 3:
                    test.append(idx)

        percentages = calculate_percentage_per_split(sub_dirs, train, test, val)
        tmp = np.sqrt(np.sum((percentages - HIT)**2))
        if tmp < diff:
            train_best = train.copy()
            val_best = val.copy()
            test_best = test.copy()

    print(f"final weighting {calculate_percentage_per_split(sub_dirs, train_best, test_best, val_best)}")

    return train_best, val_best, test_best, sub_dirs

def make_csv_files(sub_dirs, split):

    videos = []
    labels = []


    for idx in split:
        videos += [os.path.join(sub_dirs[idx], file) for file in os.listdir(sub_dirs[idx]) if '.avi' in file]

    for vid in videos:
        if 'NS' in os.path.basename(vid):
            labels.append(1)
        else:
            labels.append(0)

    frame = pd.DataFrame()
    frame['path'] = videos
    frame['labels'] = labels

    return frame



def calculate_percentage_per_split(sub_dirs, train, test, val):
    """

    :param ground_dir: directory containing all videos
    :param train: index of folders put in train
    :param test: index of folder put in test
    :param val: index of folder put in val
    :return: returns the percentage per split
    """

    num_train = get_num_files(sub_dirs, train)
    num_val = get_num_files(sub_dirs, val)
    num_test = get_num_files(sub_dirs, test)

    all_files = num_train+num_val+num_test
    percentages = np.array([num_train/all_files, num_val/all_files, num_test/all_files])

    return percentages

def get_num_files(sub_dirs, split):

    num_files = 0
    for idx in split:
        num_files += len(os.listdir(sub_dirs[idx]))

    return num_files

def sanity_check(sub1, sub2):

    for file1, file2 in zip(sub2, sub1):
        if file1 != file2:
            print("They are not the same")
            return False

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Makes splits')
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--outdir', required = False)
    args = vars(parser.parse_args())

    if args['outdir'] is None:
        outdir = args['datadir']
    else:
        outdir = args['outdir']

    datadir = args['datadir']

    train_D, val_D, test_D, sup_dirs1 = get_folders_and_split(datadir, 'D')
    train_NS, val_NS, test_NS, sup_dirs2 = get_folders_and_split(datadir, 'NS')

    check = sanity_check(sup_dirs2, sup_dirs1)
    if check:
        train = train_D + train_NS
        val = val_NS + val_D
        test = test_D+test_NS

        train_csv = make_csv_files(sup_dirs2, train)
        val_csv = make_csv_files(sup_dirs2, val)
        test_csv = make_csv_files(sup_dirs2, test)

        train_csv.to_csv(os.path.join(outdir, 'train.csv'), header = False, index=False, sep = " ")
        val_csv.to_csv(os.path.join(outdir, 'val.csv'), header=False, index=False, sep=" ")
        test_csv.to_csv(os.path.join(outdir, 'test.csv'), header=False, index=False, sep=" ")
    else:
        raise ValueError("This did not go well")




