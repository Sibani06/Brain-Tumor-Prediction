import os
#Importing the necessary libraries
import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf

#Python Imaging Library
import PIL
from PIL import Image
import imageio

#For Data Viz
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pathlib


import os
for dirname, _, filenames in os.walk('E:\\BrainTumor\\archive'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("###############################")      
        
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

##########################################
# Get the path of files
image_directory='E:\\BrainTumor\\archive\\'

no_tumor_images=os.listdir(image_directory+ 'no\\')
yes_tumor_images=os.listdir(image_directory+ 'yes\\')
# initialize dataset and label arrays
dataset=[]
label=[]
# set input size
INPUT_SIZE=64
##########################################
#loop over each image in each category
for i , image_name in enumerate(no_tumor_images):
    #read the image if its extension is .jpg
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        #resize the image
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        #append image arry in dataset list and its label in label list
        dataset.append(np.array(image))
        label.append(0)

        # same for yes images
for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)
print("############dataset#######################")
print(dataset)
print("############Label#######################")
print(label)
print(len(label))
print("##########################################")
# Convert the dataset & label to numpy array 
dataset = np.array(dataset) 
label = np.array(label)
print("####################################")
# Convert the dataset & label to numpy array 
dataset = np.array(dataset) 
label = np.array(label)
print("################################################")
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state = 42)
              
print(x_train.shape) 
print(y_train.shape) 
print("#######################################")
print(x_test.shape)
print(y_test.shape)
print("#########################################")
# Normalise the data for training purpose
x_train = normalize( x_train, axis =1)
x_test = normalize( x_test, axis =1)

# Normalise the data for training purpose
#y_train = to_categorical(y_train, num_classes=2)
#y_test = to_categorical(y_test, num_classes=2)

print("#####################################################################")
# Building a simple CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3),  kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
#model.add(Dense(2))
model.add(Activation('sigmoid'))
# model.add(Activation('softmax'))
model.summary()

print("######################################################")
'''tf.keras.utils.plot_model(model,
                          to_file="model.png",
                          show_shapes=True,
                          expand_nested=True)'''

print("###################################################")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, 
batch_size = 16, 
verbose = 1, epochs = 10, 
validation_data = (x_test, y_test),
shuffle = 'False')
model.save('Brain_Tumor_detection.h5')
# Testing the model 

import cv2
from keras.models import load_model 
from PIL import Image
import numpy as np

model = load_model('Brain_Tumor_detection.h5')
# Testing on one image 
image = cv2.imread('E:\\BrainTumor\\archive\\pred\\pred13.jpg') #tumor detected 
img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img)
print(img)
input_img = np.expand_dims(img, axis=0)

result = model.predict(input_img)
print(result)
print("####################################################")
# Testing on one image 
image = cv2.imread('E:\\BrainTumor\\archive\\pred\\pred0.jpg') # no tumor detected 
img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img)
print(img)
input_img = np.expand_dims(img, axis=0)

result = model.predict(input_img)
print(result)







