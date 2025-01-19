# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:21:31 2022

@author: Jay9696
"""

from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
x_Train=x_train_image.reshape(60000,784).astype('float32')
x_Train_normalize = x_Train/255

print(x_train_image[0])
print(x_Train[0])
print(x_Train_normalize[0])
y_Train_OneHot = np_utils.to_categorical(y_train_label)
print(y_train_label[100])
print(y_Train_OneHot[100])