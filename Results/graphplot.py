# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:24:28 2021

@author: Nicklas
"""

import pandas as pd
import matplotlib.pyplot as plt
#Graphing

train = pd.read_csv("run-SLOWFAST_4x16_R50_runs-mydata-tag-Train_Top1_err.csv")
val = pd.read_csv("run-SLOWFAST_4x16_R50_runs-mydata-tag-Val_Top1_err.csv")

steppage = train.Step.values[-1]/val.Step.values[-1]
plt.style.use('ggplot')
epoch = 196

plt.figure()
plt.plot(train.Step/epoch,train.Value)
plt.plot((val.Step*steppage)/epoch,val.Value)
plt.title('Top-1 error')
plt.ylabel('loss')
plt.xlabel('step')
plt.legend(['train','val'], loc = 'upper left')
plt.show()