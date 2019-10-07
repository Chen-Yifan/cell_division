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
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.optimizers import SGD,Adam,Adadelta
from dataGenerator import *
from keras.models import model_from_json
import argparse

#get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='/home/yifanc3/dataset2/cell_dataset/')
parser.add_argument("--ckpt_path", type=str, default='./checkpoints/tryout')
parser.add_argument("--results_path", type=str, default='./results/tryout')
parser.add_argument("--network", type=str, default='Unet')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1440)
parser.add_argument("--shape", type=int, default=480)
parser.add_argument("--opt", type=int, default=1)
parser.add_argument("--weights", type=str)
parser.add_argument("--split", type=str, default='val')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
frame_path = os.path.join(args.dataset_path,'frames')
mask_path = os.path.join(args.dataset_path,'masks')
w,h = args.width, args.height
shape = args.shape


Model_dir = os.path.join(args.ckpt_path,args.weights)

#model 
json_path = os.path.join(args.ckpt_path,'model.json')
json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()

m = model_from_json(loaded_model_json)
m.load_weights(Model_dir)

if args.opt==1:
    opt= Adam(lr = 1e-4)
elif args.opt==2:
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
else:
    opt = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[iou_score])

test_x, test_y = xy_array(mask_path, frame_path, 'val', w, h, cl=2)
score = m.evaluate(test_x/255, test_y, verbose=0)
# NO_OF_TEST_IMAGES = test_x.shape[0]
# test_gen = testGen(test_x/255, test_y, BATCH_SIZE)
# score = m.evaluate_generator(test_gen, steps=(NO_OF_TEST_IMAGES//BATCH_SIZE), verbose=0)

print("%s: %.2f%%" % (m.metrics_names[0], score[0]*100))
print("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))
with open(os.path.join(args.ckpt_path,'output%s.txt'% args.weights[8:10]), "w") as file:
    file.write("%s: %.2f%%" % (m.metrics_names[0], score[0]*100))
    file.write("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))

predict_y = m.predict(test_x/255)
# predict_y = m.predict_generator(test_gen, steps=(NO_OF_TEST_IMAGES//BATCH_SIZE), verbose=0)

result_path = os.path.join(args.results_path, args.weights[0:-5]+'-iou%.2f-results_%s'%(score[1]*100, args.split))
print(result_path)
mkdir(result_path)

#save image
save_results(result_path, test_x, test_y, predict_y, split=args.split)