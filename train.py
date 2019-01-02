import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import matplotlib.pyplot as plt

from settings import *

TRAIN_DIR = 'train'
MODEL_NAME = 'cnn{}-{}.model'.format(LR, '4convlayers')

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

def create_train_data(all_classes, all_labels):
	training_data = []
	label_index = 0
	for specific_class in all_classes:
		current_dir = TRAIN_DIR + '/' + specific_class
		print('Loading all', all_classes[label_index], 'images...')
		for img in tqdm(os.listdir(current_dir)):
			path = os.path.join(current_dir,img)
			if (IMAGE_CHANNELS==1):
				img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			elif (IMAGE_CHANNELS==3):
				img = cv2.imread(path)
			img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
			training_data.append([np.array(img),np.array(all_labels[label_index])])
		label_index = label_index + 1
	shuffle(training_data)
	return training_data

all_classes, NUM_OUTPUT = define_classes()
all_labels = define_labels(all_classes)
training_data = create_train_data(all_classes, all_labels)


##START of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS], name='input')

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


#define the training data and test/validation data
train = training_data[:int(len(training_data)*0.8)] #80% of the training data will be used for training
test = training_data[-int(len(training_data)*0.2):] #20% of the training data will be used for validation
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=NUM_EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
print('MODEL SAVED:', MODEL_NAME)


#validate and plot
fig=plt.figure()
for num,data in enumerate(training_data[:12]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
    
    data_res = np.round(model.predict([data])[0], 0)
    str_label = '?'
    for x in range(len(all_labels)):
    	#print("Comparing:", data_res, " and ", all_labels[x])
    	if ((data_res==all_labels[x]).all()):
    		str_label = all_classes[x]
    		#print("Label is", str_label)
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()