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
from skel_compression_1 import *
# from scratch import *

# sourcepath="/Users/HuongNguyen/Desktop/posenet/pose_net/" #change the data path to openpose
# path, folder1, files = next(os.walk(sourcepath))
# elapsed=[]
no_joints=17
no_coord=2
window_size = 30

# for i in range(0,1):#len(folder1)):
#     folder2_path=sourcepath+folder1[i]+'/'
#     path, folder2, files = next(os.walk(folder2_path))
posex = []
posey = []
skel_data = []
#
#     for j in range(0,len(files)):
#         filepath=folder2_path+files[j]
def pose_net(x):
    with open(x, 'r') as data_file:
        global posex
        global posey
        global skel_data
        posex = []
        posey = []
        skel_data = []
        no_joints = 17
        no_coord = 2
        window_size = 30
        ppp = 0

    # with open('/Users/HuongNguyen/Downloads/brushteeth.json', 'r') as data_file:
        data = json.load(data_file)
        posex.append(data['Posex'])  # x
        posey.append(data['Posey'])  # y

        posex = np.array(posex)
        posey = np.array(posey)
        skel_data.append(np.reshape(posex, (-1, 17)))
        skel_data.append(np.reshape(posey, (-1, 17)))
        # xdata = np.reshape(posex, (-1, 17))
        # ydata = np.reshape(posey, (-1, 17))
        # xdata = np.array(xdata)
        # ydata = np.array(ydata)


        # cf = 1
        # if np.count_nonzero(xdata) == 0 and np.shape(xdata)[1] > 5:
        #     if np.any(xdata):
        #         ppp += 1

            # print(ppp)
            #now = decoded_mt/window_size
            #for k in range (1, now):
            #    for i in range (1, no_joints):
            #       kk += 1
            #       skel_comp(xdata[i, :], ydata[i, :], vj_xdata[i, :], vj_ydata[i, :], cf)

        skel_data = np.array(skel_data)
        no_coord, no_frames, no_joints = skel_data.shape


        # print(skel_data.shape)
        skel_data = np.reshape(skel_data, (no_frames, no_joints, no_coord))
        # xdata = np.reshape(xdata, (no_frames, no_joints, no_coord))
        # ydata = np.reshape(ydata, (no_frames, no_joints, no_coord))
        # print(skel_data.shape)
        #
        skel_comp(skel_data[0:window_size, :, :], window_size, no_joints, no_coord, device='posenet')

    # print(np.min(posex), np.min(posey), np.max(posex), np.max(posey))

# pose_net('/Users/HuongNguyen/Downloads/movenet/data64.json')