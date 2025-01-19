# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:34:09 2022

@author: Jay9696
"""
import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
model = Sequential()
(x_Train,y_Train),(x_Test,y_Test)=keras.datasets.mnist.load_data()
x_Train=x_train_image.reshape(x_train_image.shape[0],28,28,1).astype('float32')
x_Test=x_test_image.reshape(x_test_image.shape[0],28,28,1).astype('float32')
x_Train_normalize = x_Train/255
x_Test_normalize = x_Test/255
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)
model.add(Conv2D(filters=16,
                 kernel_size=(3,3), 
                 padding = 'same',
                 input_shape=(28,28,1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(filters=36,
                 kernel_size=(6,6), 
                 padding = 'same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


print(model.summary())
'''model.compile(loss='cate_gorical_crossentropy',optimiser='adam',metrics='accurancy')'''