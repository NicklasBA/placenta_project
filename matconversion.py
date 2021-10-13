# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:45:11 2021

@author: Nicklas
"""
import os
import glob

import scipy.io
import pandas as pd

def flatten(t):
    return [item for sublist in t for item in sublist]

def mat_conversion(path):
    #This function take a path to a .mat file in the specied format of Freia H. bachelor
    assert isinstance(path, str)  #Input path must be string
    in_list = [[]] * 5
    outRbc = [[]] * 16

    
    matfiles = []
    
    matacc = [] #0
    matdis = [] #1
    mataccframeno = [] #2
    matdisframeno = [] #3
    matRBC = [] #4
    
    mat = scipy.io.loadmat(path)
    matfiles.append(mat)
    
    matacc = flatten(flatten(mat['acc']))
    matdis = flatten(flatten(mat['dis']))
    mataccframeno = flatten(mat['frameno_acc'])
    matdisframeno = flatten(mat['frameno_dis'])
    matRBC = mat['RBCs']
 
    colnames = ['frame', 'centroid', 'box', 'pixellist', 'circularity', 'symmetry', 'gradient', 'eccentricity', 'majoraxis', 'orientation', 'area', 'line']
    
    mat['RBCs']['inlet'] = flatten(flatten(flatten(mat['RBCs']['inlet'])))
    mat['RBCs']['outlet'] = flatten(flatten(flatten(mat['RBCs']['outlet'])))
    mat['RBCs']['yref'] = flatten(flatten(flatten(mat['RBCs']['yref'])))
    mat['RBCs']['label'] = flatten(flatten(mat['RBCs']['label']))
    mat['RBCs']['frame'] = (flatten(flatten(mat['RBCs']['frame'])))
    mat['RBCs']['centroid'] =  ([flatten(arr) for arr in (flatten(flatten(mat['RBCs']['centroid'])))])
    
    
    #lengths = [len(flatten(flatten(mat['RBCs']['frame']))[i]) for i in range(len(flatten(flatten(mat['RBCs']['frame']))))]
    
    dfObj = pd.DataFrame(mat['RBCs'][0])
    dfObj = dfObj.explode(['frame','centroid'])
    print("Hello from a function")
    return dfObj
    

basepath = 'C:/Users/Nicklas/OneDrive - Danmarks Tekniske Universitet/Undevisning/FetalMaternal_project/Placenta_package'

path = os.path.join(basepath)
testfiles = []
for entry in os.listdir(path):
    if os.path.isfile(os.path.join(path, entry)):
        testfiles.append(os.path.join(path, entry))
        
dfObj = mat_conversion(testfiles[8])

# =============================================================================
# path = os.path.join(basepath)
# for entry in os.listdir(path):
#     if os.path.isfile(os.path.join(path, entry)):
#         mat_conversion(os.path.join(path, entry))
# =============================================================================
