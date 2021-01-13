import matplotlib.pyplot as plt 
import seaborn as sns 

from config import *
import keras 
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.matrices import classification_report, confusion_matrix

import tensorflow as tf 

import cv2 
import os 

import numpy as np 

labels = ['done', 'in_progress']

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #switch BGR to RGB
                resized_arr = cv2.resize(img_arr, (IMG_WIDTH, IMG_HEIGHT))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e) 
    return np.array(data)

data_dir = 'dummy'
train = get_data('data/train_sets/%s/train', data_dir)
val = get_data('data/train_sets/%s/test', data_dir)

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

x_train.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
y_train = np.array(y_train)

x_val.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
y_val = np.array(y_val)

#data augmentation 
datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=30, #any random angle between 0 and 180
        zoom_range = 0.2,
        width_shift_range=0.1,
        heigh_shift_range=0.1,
        horizontal_flip=True,
        vetical_flip=False)

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.summary()

opt = Adam(lr=0.000001)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(x_train,y_train, epochs=epochs, validation_data=(x_val,y_val))
