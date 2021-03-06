# Simple Image Classifier
Convolutional Neural Network (CNN) using TensorFlow Library.

![Apple](https://www.thoughtco.com/thmb/QJSxu_9p3SRxDSEBZ14vgZQz6ys=/400x0/filters:no_upscale():max_bytes(150000):strip_icc()/herbstfarben-144476015-5867d0525f9b586e02213437.jpg)
```
RESULTS:

apples 54.6645 %
dragon fruit 7.697 %
grapes 10.9816 %
oranges 26.6568 %

Image is class apples
```

## Prerequisites
* Make sure you have more than 100 images per class. For characterter recognition, sample images of 20 per character is acceptable.
* Install python 3.6 if you are in Windows. For Mac or Ubuntu, make sure your python version is 3.x
* If you are using Python 2.X, then use pip2 for the installations that follow
* Install OpenCV for image manipulation ```pip install opencv-python```
* Install numpy for arrays ```pip install numpy```
* Install tensorflow ```pip install tensorflow```
* Install TFLearn ```pip install tflearn```
* Install tqdm for progress bars ```pip install tqdm```
* Install matplotlib for displaying graphed results ```pip install matplotlib```
* Install other libraries ```apt update && apt install -y libsm6 libxext6```
* Install optional libraries ```pip3 install flask```
* Install optional libraries ```pip3 install flask-cors```
* For Ubuntu or Mac users, you might be required to install Tkinter Package, to do this, just execute ```apt-get install -y python3-tk``` for Python3.X or ```apt-get install python2.7-tk``` for Python 2.7

## Training the Model
* To run the model, make sure all of your test images are inside the train folder. Inside the train folder are folders of the classes of the images. For example, if you are trying to classify a dog image and cat image, then inside the train folder are two folders of cat (with images of cats in it) and dog (with images of dogs in it). Just to be safe, make sure the images are JPG/JPEG because transparency in images can break the code.
* To run the training, just execute ```python train.py```. This will run the training and at the end of the training, a sample result will be shown using the matplotlib. After the training, model files will be created which will be used later on the test.py.

## Testing the Model
To test the model, just run the command ```python test.py "C:\Users\lenovo\Documents\Images\grapes-italy.jpg"```. The output would be something like this:
```
RESULTS:
apples 0.1196 %
dragon fruit 0.2781 %
grapes 99.0874 %
oranges 0.5148 %
Image is class grapes
```

## Structure of the CNN
All images are converted to grayscale and rescaled to 64x64 pixels. Filter sizes for all layers are 3x3 (both Convolution and Max Pooling layers). The activation function used in both the Convolution Layer and Fully Connected Layer is ReLU. Learning rate used is 0.0001 and the number of epochs is 100. All of these hyper parameters can be changed in the settings.py.
1. First layer is a Convolution Layer with 32 channels
2. After the first layer is a Max Pooling Layer
3. Next is a Convolution Layer with 64 channels
4. After that is a Max Pooling Layer
5. Next is a Convolution Layer with 128 channels
6. After that is a Max Pooling Layer
7. After that is a Fully Connected Layer with 512 channels
8. Then a dropout of 80% will be applied to prevent overfitting
9. After that, a Fully Connected Layer is applied with X number of channels where X is the number of classes (folders) in the train folder. The activation function used in this last layer is the softmax function.

[Follow me on Instagram](https://www.instagram.com/thejamesarnold/)
