# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:08:20 2022

@author: Jay9696
"""
from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
x_Train=x_train_image.reshape(60000,784).astype('float32')
x_Test=x_test_image.reshape(10000,784).astype('float32')
x_Train_normalize = x_Train/255
x_Test_normalize = x_Test/255
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)

model = Sequential()
model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x=x_Train_normalize,y=y_Train_OneHot,validation_split=0.1,epochs=50,batch_size=300,verbose=2)
scores = model.evaluate(x_Test_normalize,y_Test_OneHot)
print(scores[1])
model.save('6666666666.h5')