from utils import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import os
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.optimizers import Adadelta, Adam
import matplotlib.pyplot as plt
from keras.optimizers import SGD,Adam,Adadelta
from custom_generator import *
from load_data import *
from keras.models import model_from_json
import argparse
from metrics import *
from visualize import *
#get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./dataset/')
parser.add_argument("--ckpt_path", type=str, default='./checkpoints/tryout')
parser.add_argument("--results_path", type=str, default='./results/tryout')
parser.add_argument("--network", type=str, default='Unet')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--w", type=int, default=1920)
parser.add_argument("--h", type=int, default=1440)
parser.add_argument("--shape", type=int, default=112)
parser.add_argument("--split", type=str, default='test')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
w,h = args.w, args.h
shape = args.shape

#9-11 the epoch
weights = os.listdir(args.ckpt_path)
weight = None
for i in weights:
    if i[8:10] == str(args.epochs):
        weight = i
print(weight)
Model_dir = os.path.join(args.ckpt_path,weight)

#model 
json_path = os.path.join(args.ckpt_path,'model.json')
json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()

m = model_from_json(loaded_model_json)
m.load_weights(Model_dir)


optimizer = Adam(lr=0.0001)
m.compile(loss='binary_crossentropy', metrics=[iou_label, per_pixel_acc,'accuracy'], optimizer=optimizer)

# load data
test_x = np.load(args.results_path +'/inputs.npy')
test_y = np.load(args.results_path +'/gt_labels.npy')

score = m.evaluate(test_x/255, test_y, verbose=0)

message = ''
for j in range(len(score)):
    print("%s: %.2f%%" % (m.metrics_names[j], score[j]*100))
    message += "%s: %.2f%% \n" % (m.metrics_names[j], score[j]*100)

with open(os.path.join(args.ckpt_path,'output_%s.txt') %args.epochs, "w") as file:
    file.write(message)
    file.write('\n')

# predict_y = m.predict_generator(test_gen, steps=(NO_OF_TEST_IMAGES//BATCH_SIZE), verbose=0)
predict_y = m.predict(test_x/255, verbose=0)

#save image
print('======Save Results======')
result_path = args.results_path +'/weights.%s-results-%s'%(args.epochs, args.split)
print(result_path)
mkdir(result_path)
np.save(result_path + '/pred_labels.npy', predict_y)

# visualize result
print('=====Visualize Results====')
img = test_x
real = test_y
pred = predict_y #after sigmoid 1 channel
predicted_data = (pred>0.5).astype('uint8')

# for i in range(100):
#     visualize(result_path,img,real,pred,predicted_data,i)
