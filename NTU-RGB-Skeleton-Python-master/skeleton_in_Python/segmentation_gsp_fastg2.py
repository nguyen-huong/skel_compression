#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:15:46 2018

@author: pratyusha
"""

from multiprocessing import Process
import collections
from collections import OrderedDict
import numpy as np
import scipy.io
from scipy.fftpack import idct, dct
from huffman import *
from read_skeleton import *

#function defination
def graphconstruction1():
    source_path_skelbasis="/Users/HuongNguyen/Downloads/workspace/compress_skel/skel_basis.mat" #trying to load the gsf basis
    mat = scipy.io.loadmat(source_path_skelbasis)
    return mat['v']

#reading the data
path='/Users/HuongNguyen/Downloads/workspace/NTURGB/nturgb+d_skeletons/S017C003P020R002A060.skeleton'
data=read_skeleton(path)
# print(len(data))    #keys=time stamp
#
# print(len(data['1'][0][0]))

#importing the skeleton gft basis
v  = graphconstruction1()
# print(type(v))
#data['time stamp'][subject id][no of joints][co-ordinates]
subject_id=[0]
no_frames=len(data)
no_joints=25
no_coord=3


#compute the motion vector

gft_dct_coeff_data=[]
recon_sig_data=[]
for j in subject_id:
    #spatial gft
    gft_coeff = []
    motion_vector=[]
    for i in range(1,no_frames):
            temp=np.array(data[str(i+1)][j])-np.array(data[str(i)][j])
            # print('first value', temp[0,0])
            motion_vector.append(temp)
            gft_coeff_temp=np.matmul(np.transpose(v), temp) #compute the gft coeff for each time frame

            gft_coeff.append(gft_coeff_temp)
    gft_coeff=np.array(gft_coeff)
    motion_vector=np.array(motion_vector)

    gft_dct_coeff=gft_coeff
    #temporal dct
    for k in range (no_joints):
        for l in range(no_coord):
            x=gft_coeff[:,k,l]
            gft_dct_coeff[:,k,l]=dct(x, norm='ortho')
            #input this into huffman
    lowest_val=np.min(gft_dct_coeff)
    gft_dct_coeff=gft_dct_coeff-lowest_val  # to start the data from 0



    # quantization
    gft_dct_coeff_round=np.round(gft_dct_coeff).astype(int)
    gft_dct_coeff_data.append(gft_dct_coeff)   # 2 sub data

    Q = np.array(gft_dct_coeff_round)
    Q_flat = np.reshape(Q,((no_frames-1)*no_joints*no_coord),'C')
    Q_list = Q_flat.tolist()
    # print(Q_list)
    encoding, tree = Huffman_Encoding(Q_list)
    decoded_op=Huffman_Decoding(encoding, tree)

    decoded_op=np.array(decoded_op)

    decoded_mt=np.reshape(decoded_op,(no_frames-1,no_joints,no_coord),'C')

    #reconstruction
    decoded_mt=decoded_mt+lowest_val
    recon_sig_idct=decoded_mt
    for k in range (no_joints):
        for l in range(no_coord):
            y=decoded_mt[:,k,l]
            recon_sig_idct[:,k,l]=idct(y, norm='ortho')
            #input this into huffman

    #inverse gft
    recon_sig=[]
    for t in range(no_frames-1):
        temp=recon_sig_idct[t,:,:]
        recon_sig_perframe = np.matmul(v, temp)
        recon_sig.append(recon_sig_perframe)
    recon_sig=np.array(recon_sig)
    recon_sig_data.append(recon_sig)

err=np.linalg.norm(recon_sig-motion_vector)
print('error', err)
print(v)

