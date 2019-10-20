from segmentation_models import Unet
# from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.optimizers import SGD,Adam,Adadelta
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from dataGenerator import *
from utils import *
from metrics import *
import os
import argparse
from model import *

def get_callbacks(name_weights, path, patience_lr, opt=1):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=False, monitor='iou_score', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(factor=0.5)
    logdir = os.path.join(path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                                write_graph=True, write_images=True)
    if (opt == 3):
        return [mcp_save, tensorboard]
        
    else:
        return [mcp_save, reduce_lr_loss, tensorboard]

#get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./dataset/')
parser.add_argument("--ckpt_path", type=str, default='./checkpoints/tryout')
parser.add_argument("--results_path", type=str, default='./results/tryout')
parser.add_argument("--network", type=str, default='Unet')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1440)
parser.add_argument("--shape", type=int, default=240)
parser.add_argument("--opt", type=int, default=1)
parser.add_argument("--split", type=str, default='test')

args = parser.parse_args()

mkdir(args.ckpt_path)
with open(os.path.join(args.ckpt_path,'args.txt'), "w") as file:
    for arg in vars(args):
        print(arg, getattr(args, arg))
        file.write('%s: %s \n' % (str(arg),str(getattr(args, arg))))

BATCH_SIZE = args.batch_size
frame_path = os.path.join(args.dataset_path,'frames')
mask_path = os.path.join(args.dataset_path,'masks')
w,h = args.width, args.height
shape = args.shape

# define model
input_shape = (shape, shape, 3)
if (args.network == 'Unet'):
    m = Unet(classes = 2, input_shape=input_shape, activation='softmax')
#     m = get_unet()
elif (args.network == 'unet_noskip'):
    m = unet_noskip(n_classes=2, input_shape=input_shape)
elif (args.network == 'unet'):
    m = get_unet(n_classes=2,input_shape=input_shape)

# load data to lists
x, y = xy_array(mask_path, frame_path, '', w, h, cl=2)
assert len(x) == len(y)
print('x,y shape', x.shape, y.shape)

N = len(x)
a = int(0.7*N)
b = int(0.85*N)
train_x, val_x, test_x = x[:a],x[a:b],x[b:]
train_y, val_y, test_y = y[:a],y[a:b],y[b:]
NO_OF_TRAINING_IMAGES = a
NO_OF_VAL_IMAGES = b-a
NO_OF_TEST_IMAGES = N-b

print('train_y.shape:',train_y.shape)
print('train: val: test', NO_OF_TRAINING_IMAGES, NO_OF_VAL_IMAGES, NO_OF_TEST_IMAGES)

#DATA AUGMENTATION
train_gen = trainGen(train_x, train_y, BATCH_SIZE)
val_gen = testGen(val_x, val_y, 1)

#optimizer
if args.opt==1:
    opt= Adam(lr = 1e-4)
elif args.opt==2:
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
else:
    opt = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[iou_score, iou_label, f1_score1, per_pixel_acc])

# fit model
weights_path = args.ckpt_path + '/weights.{epoch:02d}-{val_loss:.2f}-{val_iou_score:.2f}.hdf5'
callbacks = get_callbacks(weights_path, args.ckpt_path, 5, args.opt)
history = m.fit_generator(train_gen, epochs=args.epochs,
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen,
                          validation_steps=NO_OF_VAL_IMAGES//BATCH_SIZE,
                          shuffle = True,
                          callbacks=callbacks)
#save model structure
model_json = m.to_json()
with open(os.path.join(args.ckpt_path,"model.json"), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
print("Saved model to disk")
m.save(os.path.join(args.ckpt_path,'model.h5'))

#prediction
print('======Start Evaluating======')
#don't use generator but directly from array
test_gen = testGen(test_x, test_y, BATCH_SIZE)
score = m.evaluate_generator(test_gen, steps=(NO_OF_TEST_IMAGES//BATCH_SIZE), verbose=0)
# score = m.evaluate(test_x/255, test_y, verbose=0)
with open(os.path.join(args.ckpt_path,'output.txt'), "w") as file:

message = ''
for j in range(11):
    print("%s: %.2f%%" % (m.metrics_names[j], score[j]*100))
    message += "%s: %.2f%% \n" % (m.metrics_names[j], score[j]*100)
        
with open(os.path.join(args.ckpt_path,'output_%s.txt') %args.epochs, "w") as file:           file.write(message)
    file.write('\n')

print('======Start Testing======')
predict_y = m.predict_generator(test_gen, steps=(NO_OF_TEST_IMAGES//BATCH_SIZE), verbose=0)

#save image
print('======Save Results======')
result_path = args.results_path +'weights.%s-results-%s'%(args.epochs, args.split)
mkdir(results_path)
save_results(results_path, test_x, test_y, predict_y, 'test')



