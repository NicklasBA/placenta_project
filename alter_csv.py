
import pandas as pd
import os
import pickle
import shutil
import numpy as np
import sys


def change_csv(csv_path, new_path):

    csv = pd.read_csv(csv_path, header = None)
    csv[csv.columns[0]] = [os.path.join(new_path, os.path.basename(p)) for p in csv[csv.columns[0]]]
    csv.to_csv(csv_path, header = False, index = False)


new_path = r'/scratch/s183993/placenta/raw_data/videos_blackened_noice'
csv_files = [os.path.join(new_path,i) for i in os.listdir(new_path) if ".csv" in i]
for c in csv_files:
    change_csv(c, new_path)

print("Alteration succesfull")