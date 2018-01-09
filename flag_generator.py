# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:32:26 2018

@author: admin
"""
import os
import numpy as np
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.layers import Dense, Activation, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, LeakyReLU, Reshape, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from matplotlib import pyplot as pp

from scipy.misc import imsave, imread, imresize

#image data folder
imdata_folder = "./image_data"

#loading flag data
def load_images(directory):
    path = directory
    images = []
    
    for _, _, filenames in os.walk(path):
        for file in filenames:
            images.append(imread(path+"/"+file, mode="RGB"))
            
    return images

flags = load_images(imdata_folder)

#scalling all data to the same size
def rescale_images(images, new_width, new_height):
    new_images = []
    for image in images:
        new_images.append(imresize(image, (new_height, new_width), interp='nearest'))
        
    return new_images

flags = rescale_images(flags, 100, 70)
pp.imshow(flags[5])

#normalize images
def normalize_images(images):
    new_images = []
    for image in images:
        image = image/255.
        new_images.append(image)
        
    return new_images

flags = normalize_images(flags)


#defining discriminator
def D():
    model = Sequential()
    model.add(Conv2D(32, 4, kernel_initializer='random_uniform', input_shape=(70, 100, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 4, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, 4, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, 4, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform'))

    return model

d_nn = D()
d_optimizer = RMSprop(lr=0.0001)
d_model = Sequential()
d_model.add(d_nn)
d_model.compile(optimizer=d_optimizer, loss='binary_crossentropy')

#defining generator
def G():
    model = Sequential()
    model.add(Dense(70*100, input_shape=(100,)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Reshape((70, 100, 1)))
    model.add(Conv2DTranspose(256, 2, kernel_initializer='random_uniform', padding='same'))
    model.add(BatchNormalization(momentum=0.3))
    model.add(LeakyReLU(0.02))
    model.add(Conv2DTranspose(128, 2, kernel_initializer='random_uniform', padding='same'))
    model.add(BatchNormalization(momentum=0.3))
    model.add(LeakyReLU(0.02))
    model.add(Conv2DTranspose(64, 2, kernel_initializer='random_uniform', padding='same'))
    model.add(BatchNormalization(momentum=0.3))
    model.add(LeakyReLU(0.02))
    model.add(Conv2DTranspose(32, 2, kernel_initializer='random_uniform', padding='same'))
    model.add(BatchNormalization(momentum=0.3))
    model.add(LeakyReLU(0.02))
    model.add(Conv2DTranspose(3, 2, kernel_initializer='random_uniform', padding='same'))
    model.add(Activation('tanh'))
   
    return model

g_nn = G()

s_optimizer = RMSprop(lr=0.005)
s_model = Sequential()
s_model.add(g_nn)
s_model.add(d_nn)
s_model.compile(optimizer=s_optimizer, loss='binary_crossentropy')

#noise for generator
def noise_generator():
    return np.array([np.random.uniform(-1., 1., size=100)])

#generate  fake image for generator error calculation
def error_image(error, out_image):
    out_image = out_image+error    
    return out_image

def b_crossentropy(target, prediction):
    if target == 1.:
        return -np.log(prediction)
    else:
        return -np.log(1.-prediction)

epochs=500

#training!
for epoch in range(epochs):
    np.random.shuffle(flags)
    for nb, image in enumerate(flags):
        y = np.array([1.])
        x = np.array([image])
        d_r_err = d_model.train_on_batch(x, y)
        
        noise = noise_generator()
        x = noise
        x = g_nn.predict(x)
        
        y = np.array([0.])
        d_f_err = d_model.train_on_batch(x, y)
        weights_save = d_model.get_weights()
        
        noise = noise_generator()
        y = np.array([1.])
        g_err = s_model.train_on_batch(noise, y)
        d_model.set_weights(weights_save)
        
        print("g_err = %f, d_r_err = %f, d_f_err = %f, image = %d" % (g_err, d_r_err, d_f_err, nb))
        
    print("finished epoch %d" % epoch)
    g_nn.save('g_nn')
    d_nn.save('d_nn')
    
    noise = noise_generator()
    image = g_nn.predict(noise)[0]
    image = image*255
    
    imsave('epoch'+str(epoch)+'.png', image)
    
