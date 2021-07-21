# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 01:05:58 2021

@author: abc
"""

from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from skimage.transform import resize
import tensorflow as tf
import numpy as np

#Give the path of images
path = "Colorization images\\"

#Normalize images - divide by 255 (Convert integer vallue into floating point value)
train_datagen = ImageDataGenerator(rescale=1. / 255)

#Resize images, if needed
train = train_datagen.flow_from_directory(path,
                                          target_size=(256,256),
                                          batch_size=340,
                                          class_mode=None)

#Convert From RGB to Lab
"""

by iterating on each image, we convert the RGB to Lab.
Think of LAB as a grey image in L channel and all color info stored in A and B channel
The input to the network will be the L channel , so we assign L channel to X vector
and assign A and B to Y.

"""


x = []
y = []
for img in train[0]:
    try:
        lab = rgb2lab(img)
        x.append(lab[:,:,0])
        y.append(lab[:,:,1] / 128)  #A and B values range from -127 to 128
        #So divide the values by 128 to restrict values to between -1 and 1.
    except:
        print("Error")
        
x = np.array(x)
y = np.array(y)

x = x.reshape(x.shape+(1,)) #Dimension to be the same for x and y
print(x.shape)
print(y.shape)

#AUTOENCODER

#Define Encoder
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", padding="same", strides=2, input_shape=()))
model.add(Conv2D(128, (3,3), activation="relu", padding="same"))
model.add(Conv2D(128, (3,3), activation="relu", padding="same", strides=2))
model.add(Conv2D(256, (3,3), activation="relu", padding="same"))
model.add(Conv2D(256, (3,3), activation="relu", padding="Same", strides=2))
model.add(Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(Conv2D(512, (3,3), activation="relu", padding="same"))


#Decoder
#Note : for the last layer we use tanh instead of relu
#This is because we are colorizing the image in this layer using 2 filters A and B
#A and B values range between -1 and 1 so tanh (Or hyperbolic tangent) is used 
#as it also has the range between -1 and 1
#Other function go from 0 to 1

model.add(Conv2D(128, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(64, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32, (3,3), activation="relu", padding="same"))
model.add(Conv2D(16, (3,3), activation="relu", padding="same"))
model.add(Conv2D(2, (3,3), activation="tanh", padding="same"))
model.add(UpSampling2D((2,2)))
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model.summary()


#Fit the model
model.fit(x, y, validation_split = 0.1, epochs=5, batch_size=16)

#Save the model
model.save("Colorization_autoencoder.model")



