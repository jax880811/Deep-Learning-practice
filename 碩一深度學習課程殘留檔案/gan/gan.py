# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:25:59 2022

@author: Jay9696
"""
from keras.datasets import mnist
from keras.layers import Dense,Dropout,Input
from keras.models import Model,Sequential
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#作業2
def load_data():
    (x_train,y_train),(_,_) = mnist.load_data()
    x_train = (x_train.astype(np.float32)-127.5)/127.5
    x_train = x_train.reshape(60000,-1)
    return (x_train,y_train)


x_train,y_train = load_data()
print(x_train.shape,y_train.shape)

def build_generator():
    model=Sequential()
    model.add(Dense(units=256,input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(units=1024))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(units=784,activation='tanh'))
    model.compile(loss='binary_crossentropy',optimizer=Adam(0.0002,0.5))
    return model

generator=build_generator()
generator.summary()

#以上作業3
def build_discriminator():
    model=Sequential()
    
    model.add(Dense(units=1024,input_dim=784))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    model.add(Dense(units=1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',optimizer=Adam(0.0002,0.5))
    return model

discriminator=build_discriminator()
discriminator.summary()
#以上作業4
def build_GAN(discriminator,generator):
    discriminator.trainable=False
    GAN_input=Input(shape=(100,))
    x=generator(GAN_input)
    GAN_output=discriminator(x)
    GAN=Model(inputs=GAN_input,outputs=GAN_output)
    GAN.compile(loss='binary_crossentropy',optimizer=Adam(0.0002,0.5))
    return GAN
GAN=build_GAN(discriminator, generator)
GAN.summary()
#以上作業5

def draw_images(generator,epoch,examples=25,dim=(5,5),figsize=(10,10)):
    noise=np.random.normal(loc=0,scale=1,size=[examples,100])
    generated_images=generator.predict(noise)
    generated_images=generated_images.reshape(25,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        plt.imshow(generated_images[i],interpolation='nearest',cmap='Greys')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Generated_image %d.png'%epoch)
    
def train_GAN(epochs=1,batch_size=128):
    X_train,y_train = load_data()
    generator = build_generator()
    discriminator = build_discriminator()
    GAN = build_GAN(discriminator, generator)
    
    for i in range(1,epochs+1):
        print("Epoch %d"%i)
        
        for _ in tqdm(range(batch_size)):
            noise = np.random.normal(0,1,(batch_size,100))
            fake_images = generator.predict(noise)
            
            real_images = x_train[np.random.randint(0,x_train.shape[0],batch_size)]
            
            label_fake = np.zeros(batch_size)
            label_real = np.ones(batch_size)
            
            x = np.concatenate([fake_images,real_images])
            y = np.concatenate([label_fake,label_real])
            
            discriminator.trainable=True
            discriminator.train_on_batch(x,y)
            
            discriminator.trainable=False
            GAN.train_on_batch(noise,label_real)
        
        if i==1 or i%2==0:
            draw_images(generator, i)


train_GAN(epochs = 600,batch_size = 64)








    