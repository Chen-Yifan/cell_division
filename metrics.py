import keras.backend as K
import numpy as np
import os
import glob
import skimage.io as io
import tensorflow as tf

def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1] #20
    print(nb_classes)
    iou = []
    true_pixels = K.argmax(y_true, axis=-1) # = (n, h,w)
    pred_pixels = K.argmax(y_pred, axis=-1) 
    #ignore certain labels, those doesn't have one, and those are background
    void_labels = K.equal(true_pixels, 19) # ignore label 19, background

    # in our case, the last label is background (19)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)