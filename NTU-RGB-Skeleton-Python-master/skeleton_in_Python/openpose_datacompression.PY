#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:15:46 2018

@author: pratyusha
"""
from multiprocessing import Process
import numpy as np
import scipy.io
import array
# import pandas as pd
import os
import json
from skel_compression import *

sourcepath="/Users/HuongNguyen/Downloads/workspace/open_pose/" #change the data path to openpose
path, folder1, files = next(os.walk(sourcepath))
elapsed=[]
no_joints=18
no_coord=2
for i in range(0,1):#len(folder1)):
    folder2_path=sourcepath+folder1[i]+'/'
    path, folder2, files = next(os.walk(folder2_path))
    posex=[]
    posey=[]
    skel_data=[]
    no_frames=len(files)


    for j in range(0,len(files)):
        filepath=folder2_path+files[j]
        with open(filepath) as data_file:
            data = json.load(data_file)
            posex.append(data['people'][0]['pose_keypoints'][0::3])  #x
            posey.append(data['people'][0]['pose_keypoints'][1::3])   #y

    posex = np.array(posex)
    posey = np.array(posey)
    skel_data.append(posex)
    skel_data.append(posey)


    skel_data=np.array(skel_data)
    skel_data=np.reshape(skel_data, (no_frames,no_joints,no_coord))
    # print(skel_data.shape)

    skel_comp(skel_data, no_frames, no_joints, no_coord, device='openpose')

