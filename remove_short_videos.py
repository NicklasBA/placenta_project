import os
import sys
import pickle
import cv2


def remove_files(parentdir):

    folders = [os.path.join(parentdir, file) for file in
               os.listdir(parentdir) if os.path.isdir(os.path.join(parentdir, file))]

    for folder in folders:
        files = [os.path.join(folder, file) for file in os.listdir(folder) if '.avi' in file]
        for file in files:
            count = read_video_length(file)
            if count < 30:
                os.remove(file)


def read_video_length(path):
    cap = cv2.VideoCapture(path)
    if cap.isOpened() == False:
        print("Something went wrong in the reading of the video")

    count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            count += 1
        else:
            break
    return count

if __name__ == '__main__':
    path = r'/scratch/s183993/videos_col/videos'
    remove_files(path)

