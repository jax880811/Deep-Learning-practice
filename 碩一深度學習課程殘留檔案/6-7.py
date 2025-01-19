# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:39:41 2022

@author: Jay9696
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
img = Image.open('777.png')
reIm = img.resize((28,28))
im1 = np.array(reIm.convert("L"))
plt.imshow(im1,cmap=plt.get_cmap('gray'))
plt.show

im1 = im1.reshape(1,28*28)
im1 = im1.astype('float32')/255

model = load_model('first.h5')
x = model.predict(im1)
prediction = np.argmax(x,axis=1)
print('預測結果',str(x))