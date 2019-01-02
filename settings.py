# Increase this number if you have more data or the validation accuracy (val_acc) is still low.
NUM_EPOCHS = 100

# Increase or decrease this number.
#Uusually, for character recognition, a 28x28 image is enough.
IMG_SIZE = 64

# This is the number of channels for the first convolution layer.
# The next layer will be twice as many channels as its previous layer.
FIRST_NUM_CHANNEL = 32

# Filter size for the convolution and pooling layers.
FILTER_SIZE = 3

# 1 if grayscale, 3 if RGB (colored image)
IMAGE_CHANNELS = 3

# Learning rate of the model.
# 1e-4 means 0.0001
# 1e-2 means 0.01
LR = 1e-4