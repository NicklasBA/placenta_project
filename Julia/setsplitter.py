# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:22:32 2021

@author: Nicklas
"""

import glob
import pandas as pd
import re
import os
import argparse

print( "Creating CSV from folders")
DEFAULT_FILES = '/scratch/s183993/placenta/raw_data/videos_blackened_org_bbox_full/*'
# DEFAULT_FILES =  '/Users/joachimsecher/Downloads/placenta_images/*'

parser = argparse.ArgumentParser(description="Create CSVs for JULIA Training split creator")
parser.add_argument("-i", '--input',
                    default='/Users/joachimsecher/Downloads/placenta_images/*',
                    help=f"Directory with found single cell videos. Default: {DEFAULT_FILES}")
parser.add_argument("-o", '--output',
                    default='',
                    help=f"Output directory. Default is CWD")

args = parser.parse_args()
FILES = args.input
output_dir = args.output

paths = glob.glob(FILES)
df = pd.DataFrame({"paths":paths})

def donor_code(path):
    code = re.search("(D)[0-9]{1,4}|(NS)[0-9]{1,4}", path)
    try:
        code = code[0]
    except TypeError:
        code = None
    return code

def donor_type(path):
    dtype = donor_code(path)
    if dtype is None:
        pass
    else:
        if "D" in dtype.upper():
            dtype = "D"
        elif "NS" in dtype.upper():
            dtype = "NS"

    return dtype

def donor_id(path):
    code = re.search("[0-9]{1,4}", donor_code(path))
    try:
        code = int(code[0])
    except TypeError:
        code = None
    return code


df['DonorCode'] = df.paths.apply(donor_code)
df['DonorType'] = df.paths.apply(donor_type)
df.drop('paths', inplace=True, axis=1)
# df[] = df.paths.apply(donor_id)
df_D = df[df.DonorType == "D"]
df_NS = df[df.DonorType == "NS"]
num_d = len(df_D)
num_NS = len(df_NS)
df_D["Num"] = df_D['Num'] = df.groupby('DonorCode')['DonorCode'].transform('count')
df_NS["Num"] = df_NS.groupby(["DonorCode"])['DonorCode'].transform('nunique')
#breakpoint()
df_D.to_csv(os.path.join(output_dir, "D.csv"))
df_NS.to_csv(os.path.join(output_dir, "NS.csv"))

print("Ignore the wranings above...")
print("Saved data to 'D.csv' and 'NS.csv', now go run the julia script")