import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays

from settings import *

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
			training_data.append([np.array(img), np.array(all_labels[label_index])])
		label_index = label_index + 1
	shuffle(training_data)
	return training_data