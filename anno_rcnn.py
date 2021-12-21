import pickle
import os
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
    all_folders = glob.glob(os.path.join(path_to_folders,"*",""))

    combination = []
    for folder in all_folders:
        name = os.path.normpath(folder.rsplit("_", 1)[-1])
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

    parser.add_argument('-f', '--folder_dir', 
                        help='directory containing folders of images')
    parser.add_argument('-p', '--pkl_dir', 
                        help = 'directory containing evaluation files')
    parser.add_argument('-o', '--output',
                        help='Output folder for annotation files (Default: Parent of folder_dir)',
                        required=False)
    parser.add_argument('-m','--multip',
                        help = 'if multiprocessing should be utilized',
                        default='f',
                        required=False)

    args = parser.parse_args()

    if args.folder_dir is None or args.pkl_dir is None:
        print(f"WARNING: Either folder_dir ({args.folder_dir}) or pkl_dir ({args.pkl_dir}) is not set, this run is probably going to fail...")

    # Set output folder, and/or make sure it exist
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.folder_dir[:-1]), "anno_rcnn_output", "")
        print(f"Setting output folder to: {args.output}")
    try:
        os.mkdir(args.output)
    except FileExistsError:
        pass

    combination = combine_eval_files_and_folders(args.pkl_dir, args.folder_dir)

    if args.multip == "f":
        run_through_combination(combination,args.output)
    else:
        try: 
            pool_count = int(args.multip)
        except ValueError:
            pool_count = mp.cpu_count()-1
            print(f"Could not determine CPU count from --multip {args.multip}. Maxing out. Using {pool_count} CPUs.")
        pool = mp.pool(pool_count)
        results = [pool.apply(run_through_comb_multi(),args = (f, args.output)) for f in combination]
        pool.close()
