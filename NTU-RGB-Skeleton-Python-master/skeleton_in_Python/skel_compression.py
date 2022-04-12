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
def graphconstruction_openpose():
    source_path_skelbasis="/Users/HuongNguyen/Downloads/workspace/compress_skel/skel_basis_openpose.mat" #trying to load the gsf basis
    mat = scipy.io.loadmat(source_path_skelbasis)
    return mat['v']

def graphconstruction_kinect():
    source_path_skelbasis="/Users/HuongNguyen/Downloads/workspace/compress_skel/skel_basis.mat" #trying to load the gsf basis
    mat = scipy.io.loadmat(source_path_skelbasis)
    return mat['v']

def skel_comp(data, no_frames, no_joints, no_coord, device):
    #importing the skeleton gft basis
    if device =='openpose':
        v = graphconstruction_openpose()
    elif device =='kinect':
        v  = graphconstruction_kinect()

    # no_frames=len(data)
    # no_joints=25
    # no_coord=3

    #compute the motion vector
    #data format -> no_frames, no_joints, no_coord in a numpy array
    motion_vector=[]
    gft_coeff=[]
    for i in range(1,no_frames):
            temp=data[i,:,:]-data[i-1,:,:]

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
    # print('min=',np.min(gft_dct_coeff),'max=',np.max(gft_dct_coeff))


    # quantization
    gft_dct_coeff_round=np.round(gft_dct_coeff).astype(int)


    Q = np.array(gft_dct_coeff_round)
    Q_flat = np.reshape(Q,((no_frames-1)*no_joints*no_coord),'C')
    Q_list = Q_flat.tolist()
    encoding, tree = Huffman_Encoding(Q_list)
    ## transfer encoding

    decoded_op=Huffman_Decoding(encoding, tree)
    decoded_op=np.array(decoded_op)
    # err = np.linalg.norm(np.array(Q_list) - np.array(decoded_op))
    # print('encoding error', err)

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

    err=np.linalg.norm(recon_sig-motion_vector)
    print('error', err)
    return encoding, recon_sig

if __name__=='__main__':
    skel_comp(data, no_frames, no_joints, no_coord, device)



