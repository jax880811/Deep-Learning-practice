# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:28:44 2022

@author: Jay9696
"""

from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
x_Train=x_train_image.reshape(60000,784).astype('float32')
x_Train_normalize = x_Train/255
y_Train_OneHot = np_utils.to_categorical(y_train_label)

model = Sequential()
model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
print(model.summary())