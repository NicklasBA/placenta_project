
import pandas as pd
import os
import pickle
import shutil
import numpy as np



def change_csv(csv_path):

    csv = pd.read_csv(csv_path, header = None)

    labels = [0 if 'D' in i else 1 for i in csv[csv.columns[0]]]
    csv.insert(6, 'label', labels)
    csv.to_csv(csv_path, header = False, index = False)

csv_path = '/home/s183993/placenta_project/data/placenta/annotations/train_annotations.csv'
change_csv(csv_path)
csv_path = '/home/s183993/placenta_project/data/placenta/annotations/val_annotations.csv'
change_csv(csv_path)
csv_path = '/home/s183993/placenta_project/data/placenta/annotations/test_annotations.csv'
