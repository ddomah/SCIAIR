#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 22:20:11 2022

@author: dhiren
"""


import os
import sys
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import metrics
import tensorflow
import cv2 as cv
import cv2 
from glob import glob
import fnmatch
from PIL import Image
import pydicom as dicom
from tensorflow import keras

from keras.applications import DenseNet201
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation


imagePatches = glob('/complete_dataset', recursive=True)
patternZero = '*class0.png'
patternOne = '*class1.png'
#saves the image file location of all images with file name 'class0' 
classZero = fnmatch.filter(imagePatches, patternZero) 
#saves the image file location of all images with file name 'class1'
classOne = fnmatch.filter(imagePatches, patternOne)

num_classes = 2
epochs = 10
height = 50
width = 50
channels = 3
opt = RMSprop(lr=0.0001, decay=1e-6)
resnet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(50,50,3)
)

     
class Multipleleaf:
    def __init__(self, train_aug,test_data,train_set,test_aug,test_set):
        
        self.train_aug = train_aug
        self.train_set = train_set
        self.test_aug = test_aug
        self.test_set = test_set
        self.cnn = Sequential()
        
        
    def process_multiple_leaf(self):
        self.train_aug = ImageDataGenerator(shear_range=0.4,horizontal_flip=True,
                               zoom_range=0.3,rescale=1./255)
        self.train_set = self.train_aug.flow_from_directory('CNN/dataset/train',
                                                  target_size=(64,64),batch_size=32,class_mode ='binary')
        
        self.test_aug = ImageDataGenerator(rescale=1./255)
        
        self.test_set = self.test_aug.flow_from_directory('CNN/dataset/test',
                                                target_size=(64,64),batch_size=32,class_mode = 'binary')
        
    
    def build_multiple_leaf_model(self):
        self.cnn.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
        self.cnn.add(MaxPooling2D(pool_size=2,strides=2,))
        
        self.cnn.add(Conv2D(64, (3,3), activation='relu'))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.cnn.add(Conv2D(128, (3, 3), activation='relu'))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.cnn.add(Conv2D(256, (3, 3), activation='relu'))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.cnn.add(Flatten())
        
        self.cnn.add(Dense(128, activation='relu')) 
        # Add a softmax layer with 10 output units:
        self.cnn.add(Dense(1, activation='sigmoid'))
        
        self.cnn.compile(optimizer="adam",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        EPOCHS = 10
        checkpoint_filepath = 'multi_leaf_sigmoid.h5'
        
        model_checkpoint = ModelCheckpoint('multi_detection_sigmoid.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        
        self.cnn.fit(self.train_set, steps_per_epoch=8000/32, epochs=1, validation_data=self.test_set, validation_steps=2000/32, callbacks=[model_checkpoint])
        # cnn.fit_generator(train_set, steps_per_epoch=8000/32, epochs=1, validation_data=test_set, validation_steps=2000/32)
        print(self.cnn.summary())
        
    def predict_multiple_leaf_model(self):
        imagePrediction = 'CNN/dataset/prediction/cane.jpg'

        
        substring = '.dcm'
        
        if substring in imagePrediction:
            from pathlib import Path
            data = Path('dataset/prediction/')
            path = list(data.glob('*.dcm'))
            
            x = [] 
            for p in path:
                x.append(p)
                print(p)
                ds = dicom.dcmread(p)
                plt.axis('off')
                img = plt.imshow(ds.pixel_array)
                plt.savefig('image/dataset/prediction/data.jpg')
                imagePrediction = 'dataset/prediction/data.jpg'
        else:
            img = mpimg.imread(imagePrediction,0)
            imgplot = plt.imshow(img)
            plt.show()
                            
        cnn = tf.keras.models.load_model('multi_detection_sigmoid.h5')

        test_image = image.load_img(imagePrediction, target_size = (64, 64))
        
        test_image = image.img_to_array(test_image)
        
        test_image = np.expand_dims(test_image, axis = 0)
        
        resultProstate = cnn.predict(test_image)
        
        print(resultProstate)
        
        if resultProstate[0][0] == 1:
            print("the leaf type is: cane")
        else:
            print("the leaf type is: weed")

                
def start_menu():
    X = 0
    Y = 0
    X_train = np.array([])
    X_test = np.array([]) 
    y_train = np.array([])
    y_test = np.array([])
    model =  Sequential()
    flag = 0 #represent the method for prediction
    confusion_matrix = 0
    imageData = 0 
    learn_control = 0
    flag_data = 0
    
    train_aug = 0
    test_data =0 
    train_set = 0
    test_aug = 0
    test_set =0
        
    multipleleaf = Multipleleaf(train_aug,test_data,train_set,test_aug,test_set)
    while True:
        
        choice = input("\n1. Click \n0. Exit \nEnter your value: ")
       
        if choice == '1':
            multipleleaf.process_multiple_leaf()
            multipleleaf.build_multiple_leaf_model()
            multipleleaf.predict_multiple_leaf_model()
        elif choice == '0':
            sys.exit(0)
        #print(choice)
        
if __name__ == '__main__':
   start_menu()
    


















