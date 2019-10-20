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

def iou_label(y_true, y_pred):
    ''' 
    calculate iou for label class
    IOU = true_positive / (true_positive + false_positive + false_negative)
    '''
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    return TP/(TP+FP+FN)

def iou_back(y_true, y_pred):
    ''' 
    calculate iou for background class
    IOU = true_positive / (true_positive + false_positive + false_negative)
    '''
    y_pred = 1-K.argmax(y_pred)
    y_true = 1-K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    return TP/(TP+FP+FN)

def accuracy(y_true, y_pred):
    '''calculate classification accuracy'''
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    acc = (TP+TN)/(TP+TN+FP+FN)
    return acc

def per_pixel_acc(y_true, y_pred): # class1 and class0 actually the same
#     accuracy=(TP+TN)/(TP+TN+FP+FN)
    #class 1
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    acc0 = (TP+TN)/(TP+TN+FP+FN)
    return acc0

def precision_1(y_true, y_pred):
    """Precision metric.
    precision = TP/(TP + FP)
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    return TP/(TP + FP)

def precision_0(y_true, y_pred):
    """Precision metric.
    precision = TP/(TP + FP)
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_pred = 1-K.argmax(y_pred)
    y_true = 1-K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    return TP/(TP + FP)


def recall_1(y_true, y_pred):
    """Recall metric.
    recall = TP/(TP+FN)
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    return TP/(TP + FN)

def recall_0(y_true, y_pred):
    """Recall metric.
    recall = TP/(TP+FN)
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_pred = 1-K.argmax(y_pred)
    y_true = 1-K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    return TP/(TP + FN)

def f1score_1(y_true, y_pred):
    pre = precision_1(y_true, y_pred)
    rec = recall_1(y_true, y_pred)
    denominator = (pre + rec)
    numerator = (pre * rec)
    result = (numerator/denominator)*2
    return result

def f1score_0(y_true, y_pred):
    pre = precision_0(y_true, y_pred)
    rec = recall_0(y_true, y_pred)
    denominator = (pre + rec)
    numerator = (pre * rec)
    result = (numerator/denominator)*2
    return result
