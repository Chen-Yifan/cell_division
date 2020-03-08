from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.misc
from PIL import Image
import imageio
import os
import sys
import re
import glob
from utils import *

def resample_from_masks(array, input_shape=(1440, 1920),resize_shape=(224,224)): # 1 channel
    (h,w) = array.shape
    s = 480
    p = 240
    num_h = int(h/240)-1
    num_w = int(w/240)-1
    # loop to add each array to output array: out_array (list)  
    out_array = [] 
    for i in range(num_h):
        for j in range(num_w):
             #select the specific part as the array to preprocess
                im = array[i*p:(i*p+s),j*p:(j*p+s)]
                im = np.array(Image.fromarray(im).resize(resize_shape)) # default PIL.Image.NEAREST
                # print('#######im.range',np.max(im),np.min(im),im.shape)
                # im = np.eye(2)[im]
                out_array.append(im)
    return out_array

def resample_from_frames(array, input_shape=(1440, 1920),resize_shape=(224,224)):
    '''This function: array in shape (1440, 1920) into 240x240'''
    
    (h, w, c) = array.shape

    ''' split by 480 and resize'''
    s = 480
    p = 240
    num_h = int(h/240)-1
    num_w = int(w/240)-1
    # loop to add each array to output array: out_array (list)  
    out_array = [] 
    for i in range(num_h):
        for j in range(num_w):
             #select the specific part as the array to preprocess
            if c==2:
                im = array[i*p:(i*p+s),j*p:(j*p+s),:]
                resize = np.array(Image.fromarray(im[:,:,0]).resize(resize_shape,Image.BILINEAR))
                resize = np.stack((resize, np.array(Image.fromarray(im[:,:,1]).resize(resize_shape,Image.BILINEAR)) ),axis=-1)
            else:
                im = array[i*p:(i*p+s),j*p:(j*p+s),:]
                resize = np.array(Image.fromarray(im).resize(resize_shape,Image.BILINEAR))
            # print('#######resize.range',np.max(resize),np.min(resize),resize.shape)
            out_array.append(resize)
            
    return out_array


def load_data_array(mask_path, frame_path, w, h, resize_shape, n_channels=2):
    
    frame_files = os.listdir(frame_path)
    num_files = len(frame_files)
    # binary encode   
    x = []
    y = []
    for i in range(num_files):
        if frame_files[i][0]=='.':
            continue
        if(n_channels == 2):
            img = np.load(os.path.join(frame_path, frame_files[i]))
            mask_name = frame_files[i].replace('.npy','-label.png')
        else:  
            img = np.array(Image.open(os.path.join(frame_path, frame_files[i]))) 
            mask_name = frame_files[i].replace('RGB','label')
            
        mask = np.array(Image.open(os.path.join(mask_path, mask_name)))/255 # 255 to 1,0
        mask = mask.astype(np.uint8)
        
        x += resample_from_frames(img, (h,w), resize_shape)
        y += resample_from_masks(mask, (h,w), resize_shape)

    return np.array(x),np.array(y)


def resize_val(x, y, shape=224):
    n = x.shape[0]
    ret_x = []
    ret_y = []
    for i in range(n):
        im = scipy.misc.imresize(x[i], (shape,shape),interp='nearest')
        ret_x.append(im)
        mask = np.array(Image.fromarray(y[i,:,:,1]).resize((shape,shape), Image.NEAREST)).astype('uint8')
        ret_y.append(np.eye(2)[mask])
    return np.array(ret_x), np.array(ret_y)