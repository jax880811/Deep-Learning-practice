# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:50:32 2022

@author: Jay9696
"""

from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
print(x_train_image[0])