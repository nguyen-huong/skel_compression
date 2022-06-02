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
import matplotlib.pyplot as plt
from read_skeleton import *

#function defination
def quantization_matrix(window_size,no_joints):
    qt=np.ones((window_size,no_joints))
    inc=1500
    for i in range (window_size):
        qt[i,:]=inc*qt[i,:]
        inc-=50
    return qt

def graphconstruction_openpose():
    source_path_skelbasis="/Users/HuongNguyen/Downloads/workspace/compress_skel/skel_basis_openpose.mat" #trying to load the gsf basis
    mat = scipy.io.loadmat(source_path_skelbasis)
    return mat['v']

def graphconstruction_kinect():
    source_path_skelbasis="/Users/HuongNguyen/Downloads/workspace/compress_skel/skel_basis.mat" #trying to load the gsf basis
    mat = scipy.io.loadmat(source_path_skelbasis)
    return mat['v']

def graphconstruction_posenet():
    source_path_skelbasis="/Users/HuongNguyen/Downloads/workspace/posenet_gftbasis.mat" #trying to load the gsf basis
    mat = scipy.io.loadmat(source_path_skelbasis)
    return mat['v']

def skel_comp(data, no_frames, no_joints, no_coord, device):
    #importing the skeleton gft basis
    if device =='openpose':
        v = graphconstruction_openpose()
    elif device =='kinect':
        v  = graphconstruction_kinect()
    elif device =='posenet':
        v  = graphconstruction_posenet()

    window_size=no_frames


    #compute the motion vector
    #data format -> no_frames, no_joints, no_coord in a numpy array
    motion_vector=[]
    gft_coeff=[]
    for i in range(0,no_frames):

        # temp=data[i,:,:]-data[i-1,:,:]  #change here and use the position data
        # motion_vector.append(temp)

        temp=data[i,:,:]
        # print(v.shape)

        gft_coeff_temp=np.matmul(np.transpose(v), temp) #compute the gft coeff for each time frame
        gft_coeff.append(gft_coeff_temp)

    gft_coeff=np.array(gft_coeff)
    # motion_vector=np.array(motion_vector)

    gft_dct_coeff=gft_coeff
    #temporal dct
    for k in range (no_joints):
        for l in range(no_coord):
            x=gft_coeff[:,k,l]
            gft_dct_coeff[:,k,l]=dct(x, norm='ortho')
            #input this into huffman

    # lowest_val=np.min(gft_dct_coeff)
    # gft_dct_coeff=gft_dct_coeff-lowest_val  # to start the data from 0
    # print('min=',np.min(gft_dct_coeff),'max=',np.max(gft_dct_coeff))

    qt=quantization_matrix(window_size,no_joints)  ##non uniform quantization matrix
    # quantization
    # gft_dct_coeff_round=gft_dct_coeff
    gft_dct_coeff_round=np.zeros((no_frames,no_joints,no_coord))
    for l in range(no_coord):
        gft_dct_coeff_round[:,:,l]=(np.round(np.divide(gft_dct_coeff[:,:,l],qt),3).astype(int))


    #### do not use round off
    #use the quatization matrix and divide the coeff by the matrix

    # plt.imshow(gft_dct_coeff_round[:, :, 0], cmap='hot')
    # cbar = plt.colorbar()
    # plt.show()
    # plt.close('all')
    #
    # plt.imshow(gft_dct_coeff_round[:, :, 1], cmap='hot')
    # cbar = plt.colorbar()
    # plt.show()
    # plt.close('all')

    # print(gft_dct_coeff_round[:,:,0])

    Q = np.array(gft_dct_coeff_round)
    print(Q.shape)

    # data_dct =  np.dot(gft_dct_coeff, np.linalg.pinv(Q))
    # print(np.reshape(Q[:, :, 0], ((no_frames - 1) * no_joints), 'C'))

    Q_flat = np.reshape(Q,((no_frames)*no_joints*no_coord),'C')
    # print(type(Q_flat))
    Q_list = Q_flat.tolist()
    encoding, tree = Huffman_Encoding(Q_list)
    ## transfer encoding

    decoded_op=Huffman_Decoding(encoding, tree)
    decoded_op=np.array(decoded_op)
    # err = np.linalg.norm(np.array(Q_list) - np.array(decoded_op))
    # print('encoding error', err)

    decoded_mt=np.reshape(decoded_op,(no_frames,no_joints,no_coord),'C')

    # multiply with the quantization matrix

    #reconstruction
    # decoded_mt=decoded_mt+lowest_val
    recon_sig_idct=np.zeros((no_frames,no_joints,no_coord))
    for l in range(no_coord):
        recon_sig_idct[:,:,l] = np.multiply(decoded_mt[:,:,l],qt)

    recon_sig_idct=np.array(recon_sig_idct)
    for k in range (no_joints):
        for l in range(no_coord):
            y=recon_sig_idct[:,k,l]
            recon_sig_idct[:,k,l]=idct(y, norm='ortho')
            #input this into huffman
    # print('recon_sig_idct', recon_sig_idct.shape)
    #inverse gft
    recon_sig=[]
    for t in range(no_frames):
        temp=recon_sig_idct[t,:,:]
        # print('temp size', temp.shape)
        recon_sig_perframe = np.matmul(v, temp)
        recon_sig.append(recon_sig_perframe)
    recon_sig=np.array(recon_sig)

    #save window in json file

    err=np.linalg.norm(recon_sig-data)
    print('error', err)
    return encoding, recon_sig

if __name__=='__main__':
    #define data before you run this independently
    skel_comp(data, no_frames, no_joints, no_coord, device)
