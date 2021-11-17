##START of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from settings import *

def makeModel(NUM_OUTPUT):
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

	return model