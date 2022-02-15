# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:22:32 2021

@author: Nicklas
"""

import glob
import pandas as pd


print( "Creating videos from images")
#FILES='/scratch/s183993/placenta/raw_data/videos_blackened_org_bbox_full/*_6mbar_500fps_*_*'
FILES='/scratch/s183993/placenta/raw_data/Placenta_package/*_6mbar_500fps_*_*'
# OUTDIR = "/home/s183993/placenta_project/outputs"
paths = glob.glob(FILES)
#breakpoint()
df = pd.DataFrame({"paths":paths})
df['DonorType'] = df.paths.str.slice(-13,-12)
df['DonorCode'] = df.paths.str.slice(-14,-9)
df['Num'] = df.paths.str.slice(-8,-4)
df2 = df[df.DonorType == "D"]
df3 = df[df.DonorType == "N"]
num_d = sum(df.DonorType == 'D')
num_NS = sum(df.DonorType == 'N')
df2 = df2.groupby(df.DonorCode).count()
df3 = df3.groupby(df.DonorCode).count()
#breakpoint()
df2.to_csv("D.csv")
df3.to_csv("NS.csv")
