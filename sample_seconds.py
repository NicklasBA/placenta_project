import numpy as np
import pandas as pd
import pickle
import os
import argparse
import cv2

def read_and_save_video(path_to_video, save_path):

    cap = cv2.VideoCapture(path_to_video)

    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    frame_height = int(cap.get(3))
    frame_width = int(cap.get(4))
    size = (frame_height, frame_width)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'HFYU'), 15, size)
    turn = True
    while True:
        ret, frame = cap.read()

        if ret:
            if turn:
                out.write(frame)
                turn = False
            else:
                turn = True

        else:
            break

    cap.release()
    out.release()

    return None


def change_frame(dataframe, new_path):
    """

    :param dataframe: Dataframe containing paths to videos in the split
    :param new_path: ground path to where the new videos should lie, basenames remain unchanged
    :return: dataframe of same structure as input containing paths to new videos
    """

    videos = []

    for idx, path in enumerate(dataframe[dataframe.columns[0]]):
        new_name = os.path.join(new_path, os.path.basename(path))
        videos.append(new_name)
        read_and_save_video(path, new_name)

    new_frame = pd.DataFrame()
    new_frame['videos'] = videos
    new_frame['diags'] = dataframe[dataframe.columns[1]]
    return new_frame


def change_csv_files(path_to_csv, new_path, outdir):
    """

    :param path_to_csv: Path to the csv files containing the splits and full paths
    :return: new csv files, containing the paths of the new (and downsampled videos)
    """

    csv_files = [os.path.join(path_to_csv, i) for i in os.listdir(path_to_csv) if '.csv' in i]

    for file in csv_files:
        frame = pd.read_csv(file)
        new_frame = change_frame(frame, new_path)
        name = os.path.basename(file)
        new_frame.to_csv(os.path.join(outdir, name))

        print(f"Saved {name} succesfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample every second frame from videos')
    parser.add_argument('--csv_path',desc = 'Path to directory containing csv files of splits', required=True)
    parser.add_argument('--new_path',desc = 'Directory to where the new videos should be saved', required=True)
    parser.add_argument('--outdir', desc = 'Directory to save new csv files' ,required=True)
    args = vars(parser.parse_args())

    change_csv_files(path_to_csv=args['csv_path'],new_path=args['new_path'],outdir=args['outdir'])
    