

import numpy as np
import pandas as pd
import pickle
import os
import shutil
import glob
import anno_rcnn_helper as help
import argparse
import multiprocessing as mp


def combine_eval_files_and_folders(path_to_eval_files, path_to_folders):
    """

    :param path_to_eval_files: directory of PKL files containing evals from RCNN
    :param path_to_folders: directory of the folders containing the images
    :return: (list) with elements (folder, eval_file)
    """

    eval_files = glob.glob(os.path.join(path_to_eval_files, "*.pkl"))
    folders = [file for file in glob.glob(path_to_folders,"") if os.path.isdir(file)]

    combination = []
    for folder in folders:
        name = folder.split("_")[-1]
        for file in eval_files:
            if name in file:
                combination.append((folder, file))
                break

    return combination

def run_through_combination(combination, outdir):
    for folder, file in combination:
        pcl = pickle.load(open(file,'rb'))
        help.run_through_folder(folder = folder, evals = pcl, OUTDIR=outdir)

def run_through_comb_multi(folder_file, outdir):

    pcl = pickle.load(open(folder_file[1],'rb'))
    help.run_through_folder(folder = folder_file[0], evals = pcl, OUTDIR=outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find sequences and annotate folders containing the image files')

    parser.add_argument('--folder_dir', help='directory containing folders of images')
    parser.add_argument('--pkl_dir', help = 'directory containing evaluation files')


    parser.add_argument('--output',
                        help='Output folder for annotation files (Default: Parent of input)',
                        required=False)
    parser.add_argument('--multip',
                        help = 'if multiprocessing should be utilized',
                        default='f',
                        required=False)

    args = parser.parse_args()

    combination = combine_eval_files_and_folders(args.pkl_dir, args.folder_dir)
    if args.multip == "f":
        run_through_combination(combination,args.output)
    else:
        pool = mp.pool(mp.cpu_count()-1)
        results = [pool.apply(run_through_comb_multi(),args = (f, args.output)) for f in combination]
        pool.close()








