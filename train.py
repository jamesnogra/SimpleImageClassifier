import numpy as np         # dealing with arrays
import matplotlib.pyplot as plt

from settings import *
from dataloader import *
from model import *

all_classes, NUM_OUTPUT = define_classes()
all_labels = define_labels(all_classes)
training_data = create_train_data(all_classes, all_labels)

# make the model from model.py
model = makeModel(NUM_OUTPUT)

#define the training data and test/validation data
train = training_data[:int(len(training_data)*(PERCENT_TRAINING_DATA/100))] #data will be used for training
test = training_data[-int(len(training_data)*((100-PERCENT_TRAINING_DATA)/100)):] #data will be used for validation
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=NUM_EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# save the model after training
model.save(MODEL_NAME)
print('MODEL SAVED:', MODEL_NAME)