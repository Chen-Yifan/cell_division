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

def resample_small_size(array, input_shape=(1440, 1920)):
    '''array in shape (1440, 1920)'''
    if array.ndim == 3:
        (h, w, c) = array.shape
    else:
        (h,w) = array.shape
        c = 0
    s = 480
    num_h = int(h/s)
    num_w = int(w/s)
    n = int(num_h*num_w)
    if c:
        out_array = np.empty((n, s, s, c)).astype(np.float32)
    else:
        out_array = np.empty((n, s, s, 2)).astype(np.uint8)
        
        array = np.eye(2)[array]
    
    for i in range(num_h):
        for j in range(num_w):
            out_array[i*num_w+j] = array[i*s:(i+1)*s,j*s:(j+1)*s,:]
    return out_array

    
def load_data(frame_path, mask_path, w, h):
    #training set
    train_x, train_y = xy_array(mask_path, frame_path, 'train', w, h)
    val_x, val_y = xy_array(mask_path, frame_path, 'val', w, h)
#     test_x, test_y = xy_array(mask_path, frame_path, 'test')

    return train_x, train_y, val_x, val_y

def xy_array(mask_path, frame_path, split, w, h, cl=2):
    
    mask_path = os.path.join(mask_path, split)
    frame_path = os.path.join(frame_path, split)
    
    frame_files = os.listdir(frame_path)
    
    # binary encode   
    num_files = len(frame_files)
    
    n = int(w*h/(480**2))
    
    x = np.zeros((num_files*n, 480, 480, 3)).astype(np.float32)
    y = np.zeros((num_files*n, 480, 480, cl)).astype(np.uint8)
    
    for i in range(num_files):
#         img = np.array(Image.open(os.path.join(frame_path, frame_files[i]))) # rescale to from 0-1
#         mask = np.array(Image.open(os.path.join(mask_path, frame_files[i][:-7]+'label.png'))/255)
        img = np.load(os.path.join(frame_path, frame_files[i]))
        mask = np.load(os.path.join(mask_path, frame_files[i][:-7]+'label.npy'))
        mask = mask.astype(np.uint8)
    
        x[i*n:(i+1)*n] = resample_small_size(img, input_shape=(h,w))
        y[i*n:(i+1)*n] = resample_small_size(mask, input_shape=(h,w))
        
    return x,y
         

def trainGen(train_x, train_y, batch_size):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    #has rescaled when loading the data
    x_gen_args = dict(
                    rescale = 1./255,
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    
    y_gen_args = dict(
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    
    img_datagen = ImageDataGenerator(**x_gen_args)
    mask_datagen = ImageDataGenerator(**y_gen_args)
    
    img_datagen.fit(train_x)
    mask_datagen.fit(train_y)

    seed = 2018
    img_gen = img_datagen.flow(train_x, seed = seed, batch_size=batch_size, shuffle=True)#shuffling
    mask_gen = mask_datagen.flow(train_y, seed = seed, batch_size=batch_size, shuffle=True)
    train_gen = zip(img_gen, mask_gen)

    return train_gen


def testGen(val_x, val_y, batch_size):
# val_gen
    img_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    
    img_datagen.fit(val_x)
    mask_datagen.fit(val_y)
    
    img_gen = img_datagen.flow(val_x, batch_size=batch_size, shuffle=True)
    mask_gen = mask_datagen.flow(val_y, batch_size=batch_size, shuffle=True)
    val_gen = zip(img_gen, mask_gen)    
        
    return val_gen


def save_results(result_dir, test_x, test_y, predict_y, split='test'):
    
    test_y = np.argmax(test_y, axis=-1).astype(np.uint8)
    predict_y = np.argmax(predict_y, axis=-1).astype(np.uint8)
    
    for i in range(len(test_x)):
        # 256,256,1 -- id --> change to color
        gt = test_y[i].astype('uint8')
        pred = predict_y[i].astype('uint8')
        imageio.imwrite(os.path.join(result_dir, str(i) + '_gt.png'), gt)
        imageio.imwrite(os.path.join(result_dir, str(i) + '_pred.png'), pred)
        
        np.save(os.path.join(result_dir, str(i) + '_gt.npy'), gt*255)
        np.save(os.path.join(result_dir, str(i) + '_pred.npy'), pred*255)
