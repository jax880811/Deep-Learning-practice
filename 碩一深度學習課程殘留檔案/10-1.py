# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:42:04 2022

@author: Jay9696
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout,Activation
from keras.layers.recurrent import LSTM
import keras

stockdf = pd.read_csv('dataset.csv',index_col = 0)
stockdf.dropna(how='any',inplace=True)

#print(stockdf)

min_max_scaler = preprocessing.MinMaxScaler()

newdf = stockdf.copy()
flagdf = stockdf.copy()

newdf['open'] = min_max_scaler.fit_transform(stockdf.open.values.reshape(-1,1))
newdf['low'] = min_max_scaler.fit_transform(stockdf.low.values.reshape(-1,1))
newdf['high'] = min_max_scaler.fit_transform(stockdf.high.values.reshape(-1,1))
newdf['close'] = min_max_scaler.fit_transform(stockdf.close.values.reshape(-1,1))
newdf['volume'] = min_max_scaler.fit_transform(stockdf.volume.values.reshape(-1,1))
#print(newdf)

datavalue = newdf.values
result = []
#print(datavalue)

time_frame = 10
for index in range(len(datavalue)-(time_frame+1)):
    result.append(datavalue[index:index+(time_frame+1)])
#print(result)
result = np.array(result)
#print(result.shape[0])
'''
for i in range (result.shape[0]):
    print(result[i])
    '''
#到10-5的作業

number_train = round(0.9 * result.shape[0])
X_train = result[:int(number_train), :-1 ,0:5]  
Y_train = result[:int(number_train),-1][:,-1]
Y_train_onehot = np_utils.to_categorical(Y_train)

#到10-6的作業

X_test = result[int(number_train):, :-1 ,0:5]  
Y_test = result[int(number_train):,-1][:,-1]

Y_test_onehot = np_utils.to_categorical(Y_test)
#print(X_test.shape)


d = 0.5
model = Sequential()
model.add(LSTM(256,input_shape=(10,5),return_sequences=True,activation='tanh'))
model.add(Dropout(d))
model.add(LSTM(128,return_sequences=False,activation='tanh'))
model.add(Dropout(d))
model.add(Dense(units=16,activation = 'relu'))
model.add(Dense(units=3,activation = 'softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(X_train,Y_train_onehot,batch_size=32,epochs=50,validation_split=0.0001,verbose=1)
pred = model.predict(X_test)
score = model.evaluate(X_train,Y_train_onehot)
print(score[1])
score = model.evaluate(X_test,Y_test_onehot)
print(score[1])
model.save('stock.h5')
#到10-7 10-8的作業
#X_train = result[:int]
'''
X_train = []
y_train = []
for i in range(60,2035):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
'''