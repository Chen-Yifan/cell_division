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

def resample_from_masks(array, input_shape=(1440, 1920),resize_shape=(224,224)):
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
            if mask:
                im = array[i*p:(i*p+s),j*p:(j*p+s)]
                resize_arr = scipy.misc.imresize(im, resize_shape,interp='nearest')
                #im = np.eye(2)[im]
            else:
                im = array[i*p:(i*p+s),j*p:(j*p+s),:]
                resize_arr = scipy.misc.imresize(im, resize_shape,interp='bilinear')
            out_array.append(resize_arr)
            
    return out_array

def resample_small_size(array, input_shape=(1440, 1920),resize_shape=(224,224)):
    '''This function: array in shape (1440, 1920) into 240x240'''
    
    if array.ndim == 3:
        (h, w, c) = array.shape
        mask = False
    else:
        (h,w) = array.shape
        mask = True # if this is mask array input here, we need to use one-hot encoding
        
    ''' split by 480 and resize to 240'''
    s = 480
    p = 240
    num_h = int(h/240)-1
    num_w = int(w/240)-1
    # loop to add each array to output array: out_array (list)  
    out_array = [] 
    for i in range(num_h):
        for j in range(num_w):
             #select the specific part as the array to preprocess
            if mask:
                im = array[i*p:(i*p+s),j*p:(j*p+s)]
                resize_arr = scipy.misc.imresize(im, resize_shape,interp='nearest')
                #im = np.eye(2)[im]
            else:
                im = array[i*p:(i*p+s),j*p:(j*p+s),:]
                resize_arr = scipy.misc.imresize(im, resize_shape,interp='bilinear')
            out_array.append(resize_arr)
            
    return out_array


def load_data_array(mask_path, frame_path, w, h, resize_shape, n_channels=2):
    
    frame_files = os.listdir(frame_path)
    num_files = len(frame_files)
    # binary encode   
    x = []
    y = []
        
    for i in range(num_files):
        if(n_channels = 2):
            img = np.load(os.path.join(frame_path, frame_files[i]))
            mask_name = frame_files[i].replace('.npy','-label.tif')
        else:  
            img = np.array(Image.open(os.path.join(frame_path, frame_files[i]))) 
            mask_name = frame_files[i].replace('RGB','label')
            
        print('#########Range of image, size of image', np.max(img), np.min(img), img.shape)
        # 255 to 1
        
        mask = np.array(Image.open(os.path.join(mask_path, mask_name)))/255
        mask = mask.astype(np.uint8)
        
        x += resample_small_size(img, (h,w), resize_shape)
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