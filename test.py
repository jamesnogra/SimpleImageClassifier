import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import sys

from settings import *

TRAIN_DIR = 'train'
MODEL_NAME = 'cnn{}-{}.model'.format(LR, '4convlayers')

#get the image from the command 'python test.py "sample.png"'
temp_image = sys.argv[1]
test_image = cv2.imread(temp_image, cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (IMG_SIZE,IMG_SIZE))

#getting all folders
def define_classes():
	print('Defining classes...')
	all_classes = []
	for folder in tqdm(os.listdir(TRAIN_DIR)):
		all_classes.append(folder)
	return all_classes, len(all_classes)

#define labels using the folders
def define_labels(all_classes):
	print('Defining labels...')
	all_labels = []
	for x in tqdm(range(len(all_classes))):
		all_labels.append(np.array([0. for i in range(len(all_classes))]))
		all_labels[x][x] = 1.
	return all_labels

all_classes, NUM_OUTPUT = define_classes()
all_labels = define_labels(all_classes)


##START of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, FIRST_NUM_CHANNEL, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)

convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*2, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)

convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*4, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)

convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*8, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, FILTER_SIZE)

convnet = fully_connected(convnet, FIRST_NUM_CHANNEL*16, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, NUM_OUTPUT, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
##END of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/


print('LOADING MODEL:', '{}.meta'.format(MODEL_NAME))
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model',MODEL_NAME,'loaded...')


#identify image
print('\nRESULTS:\n')
data = test_image.reshape(IMG_SIZE,IMG_SIZE,1)
data_res_float = model.predict([data])[0]
data_res = np.round(data_res_float, 0)
str_label = '?'
for x in range(len(all_labels)):
	print(all_classes[x], str(round((data_res_float[x]*100),4)),'%')
	#print("Comparing:", data_res, " and ", all_labels[x])
	if ((data_res==all_labels[x]).all()):
		str_label = all_classes[x]
print("\nImage is class", str_label)