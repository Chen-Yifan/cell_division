from segmentation_models import Unet
# from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.optimizers import SGD,Adam,Adadelta
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from custom_generator import *
from load_data import *
from utils import *
from metrics import *
import os
import argparse
from model import *
from sklearn.model_selection import train_test_split
from visualize import *

def get_callbacks(weights_path, model_path, patience_lr):

    logdir = os.path.join(model_path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                            write_graph=True, write_images=True)
    if weights_path:
        mcp_save = ModelCheckpoint(weights_path, save_best_only=False)
        return [mcp_save, tensorboard]
    return [tensorboard]

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
parser.add_argument("--shape", type=int, default=240)
parser.add_argument("--opt", type=int, default=1)
parser.add_argument("--split", type=str, default='test')
parser.add_argument("--learn_rate", type=float, default=3e-4)
parser.add_argument("--num_filters", type=int, default=112)

args = parser.parse_args()
mkdir(args.ckpt_path)
with open(os.path.join(args.ckpt_path,'args.txt'), "w") as file:
    for arg in vars(args):
        print(arg, getattr(args, arg))
        file.write('%s: %s \n' % (str(arg),str(getattr(args, arg))))

BATCH_SIZE = args.batch_size
frame_path = os.path.join(args.dataset_path,'2channel_image')
mask_path = os.path.join(args.dataset_path,'label')
w,h = args.w, args.h
shape = args.shape
num_filters = args.num_filters

# load data to lists
frame_data, mask_data = load_data_array(mask_path, frame_path, '', w, h,(shape,shape), cl=2)
assert len(frame_data) == len(mask_data)

print('x,y shape', frame_data.shape, mask_data.shape)
print('point1, finished load data')

print('point2, shape frame mask', frame_data.shape, mask_data.shape)
'''2. split train_val_test:
        input_train/val/test
        label_train/val/test  '''
train_x, test_x, train_y, test_y = train_test_split(
            frame_data, mask_data, test_size=0.15, shuffle=False)

train_x, val_x, train_y, val_y = train_test_split(
            frame_data, mask_data, test_size=0.1, shuffle=False)

mkdir(args.results_path)
np.save(args.results_path + '/inputs.npy', test_x)
np.save(args.results_path + '/gt_labels.npy', test_y)

print('point3, shape frame mask', train_x.shape, train_y.shape)
n_train, n_test, n_val = len(train_x), len(test_x), len(val_x)
print('***** #train: #test: #val = %d : %d :%d ******'%(n_train, n_test, n_val))

#DATA AUGMENTATION
train_gen = trainGen(train_x, train_y, BATCH_SIZE)
# val_gen = testGen(val_x, val_y, 1)

# define model
input_shape = (shape, shape, 3)
if (args.network == 'Unet'):
    m = Unet(classes = 2, input_shape=input_shape, activation='softmax')
#     m = get_unet()
elif (args.network == 'unet_noskip'):
    m = unet_noskip(n_classes=2, input_shape=input_shape)
elif (args.network == 'unet'):
    m = get_unet(n_classes=2,input_shape=input_shape)
else:
    learn_rate = 0.0001
    drop = 0.15
    num_filter = 3
    m = build_model(shape, learn_rate, 1e-6, drop, 3, 'he_normal', num_filters)

# fit model
weights_path = args.ckpt_path + '/weights.{epoch:02d}-{val_loss:.2f}-{val_iou_label:.2f}.hdf5'
callbacks = get_callbacks(weights_path, args.ckpt_path, 5)
history = m.fit_generator(train_gen, epochs=args.epochs,
                          steps_per_epoch = (n_train//BATCH_SIZE),
                          validation_data=(val_x/255, val_y),
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
BATCH_SIZE = 1 # for test
# test_gen = testGen(test_x, test_y, BATCH_SIZE)
# score = m.evaluate_generator(test_gen, steps=(NO_OF_TEST_IMAGES//BATCH_SIZE), verbose=0)
score = m.evaluate(test_x/255, test_y, verbose=0)
message = ''
for j in range(len(score)):
    print("%s: %.2f%%" % (m.metrics_names[j], score[j]*100))
    message += "%s: %.2f%% \n" % (m.metrics_names[j], score[j]*100)
        
with open(os.path.join(args.ckpt_path,'output_%s.txt') %args.epochs, "w") as file:
    file.write(message)
    file.write('\n')

print('======Start Testing======')
# predict_y = m.predict_generator(test_gen, steps=(NO_OF_TEST_IMAGES//BATCH_SIZE), verbose=0)
predict_y = m.predict(test_x/255)

#save image
print('======Save Results======')
result_path = args.results_path +'/weights.%s-results-%s'%(args.epochs, args.split)
print(result_path)
mkdir(result_path)
np.save(result_path + '/pred_labels.npy', predict_y)
# save_results(results_path, test_x, test_y, predict_y, 'test')


# visualize result
img = test_x
real = test_y
pred = predict_y #after sigmoid 1 channel

predicted_data = np.zeros(pred.shape)
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        for k in range(pred.shape[2]):
            if (pred[i,j,k]>=0.5):
                predicted_data[i,j,k] =1
            else:
                predicted_data[i,j,k] =0

for i in range(100):
    visualize(result_path,img,real,pred,predicted_data,i)