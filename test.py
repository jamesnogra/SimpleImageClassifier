import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import sys
import urllib.request

from settings import *
from dataloader import *
from model import *

#get the image from the command 'python test.py "sample.png"'
temp_image = sys.argv[1]
# check if image is from internet
if 'http' in temp_image:
	url_response = urllib.request.urlopen(temp_image)
	test_image = cv2.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), -1)
elif (IMAGE_CHANNELS==1):
	test_image = cv2.imread(temp_image, cv2.IMREAD_GRAYSCALE)
elif (IMAGE_CHANNELS==3):
	test_image = cv2.imread(temp_image)
test_image = cv2.resize(test_image, (IMG_SIZE, IMG_SIZE))

all_classes, NUM_OUTPUT = define_classes()
all_labels = define_labels(all_classes)

# make the model from model.py
model = makeModel(NUM_OUTPUT)

print('LOADING MODEL:', '{}.meta'.format(MODEL_NAME))
if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('Model',MODEL_NAME,'loaded...')

	#identify image
	print('\nRESULTS:\n')
	data = test_image.reshape(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
	data_res_float = model.predict([data])[0]
	data_res = np.round(data_res_float, 0)
	str_label = '?'
	for x in range(len(all_labels)):
		print(all_classes[x], str(round((data_res_float[x]*100),4)),'%')
		#print("Comparing:", data_res, " and ", all_labels[x])
		if ((data_res==all_labels[x]).all()):
			str_label = all_classes[x]
	print("\nImage is class", str_label)