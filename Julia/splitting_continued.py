# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 14:29:00 2021

@author: Nicklas
"""

import glob
import pandas as pd
import os
import argparse
import random

print( "Creating CSV from folders")
DEFAULT_FILES = '/scratch/s183993/placenta/raw_data/videos_blackened_org_bbox_full/*'
# DEFAULT_FILES =  '/Users/joachimsecher/Downloads/placenta_images/*'

parser = argparse.ArgumentParser(description="Create CSVs for JULIA Training split creator")
parser.add_argument("-i", '--input',
                    default='',
                    help=f"Directory with D_out.csv and NS_out.csv. Default is CWD")
parser.add_argument("-vi", '--video_dir',
                    default='',
                    help=f"Directory with videos. Default is CWD")
parser.add_argument("-o", '--output',
                    default='',
                    help=f"Output directory. Default is CWD")

args = parser.parse_args()

# Split col
SCID = "x1"
TRAINID = 1
VALID = 2
TESTID = 3
DONORID = 0
FETALID = 1


df_D = pd.read_csv(os.path.join(args.input, "D_out.csv"), )
df_NS = pd.read_csv(os.path.join(args.input, "NS_out.csv"), )
df_D[SCID] = df_D[SCID].astype(int)
df_NS[SCID] = df_NS[SCID].astype(int)

# df = pd.concat([df_D, df_NS])

output_dict_train = []
output_dict_val = []
output_dict_test = []

# Create test for donor
for df, id in zip([df_D, df_NS], [DONORID, FETALID]):
    print(f"Splitting for id {id}")
    for code, split in zip(df["DonorCode"], df[SCID]):
        files = glob.glob(os.path.join(args.video_dir, "*"+code+"*.mp4"))
        for file in files:
            if split == TRAINID:
                output_dict_train.append([file, id])
            elif split == VALID:
                output_dict_val.append([file, id])
            elif split == TESTID:
                output_dict_test.append([file, id])
            else:
                raise RuntimeError(f"Unknown split value of {split}, for donor {code}")

random.shuffle(output_dict_train)
random.shuffle(output_dict_val)
random.shuffle(output_dict_test)

df_train = pd.DataFrame(output_dict_train, columns=["Paths", "ID"])
df_val = pd.DataFrame(output_dict_val, columns=["Paths", "ID"])
df_test = pd.DataFrame(output_dict_test, columns=["Paths", "ID"])

df_train.to_csv(os.path.join(args.output, "test.csv"), index=False, sep=" ", header=False)
df_val.to_csv(os.path.join(args.output, "train.csv"), index=False, sep=" ", header=False)
df_test.to_csv(os.path.join(args.output, "test.csv"), index=False, sep=" ", header=False)

print("Done, new splits saved to", os.path.join(args.output, "test.csv"), os.path.join(args.output, "train.csv"), os.path.join(args.output, "test.csv"))