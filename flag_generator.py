# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:32:26 2018

@author: admin
"""
import os
import numpy as np
import datetime
from keras.models import Sequential, load_model
from keras.losses import binary_crossentropy
from keras.layers import Dense, Activation, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, LeakyReLU, Reshape, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from matplotlib import pyplot as pp

import pandas as pnds

from scipy.misc import imsave, imread, imresize, toimage

np.random.seed(1)

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
        image = (image/255.)*2.
        image = image - 1.
        new_images.append(image)
        
    return new_images

flags = normalize_images(flags)


#defining discriminator
def D():
    model = Sequential()
    model.add(Conv2D(16, 4, kernel_initializer='random_uniform', input_shape=(70, 100, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 4, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 4, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, 4, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform'))

    return model

d_nn = None

if os.path.exists('d_nn'):
    d_nn = load_model('d_nn')
else:
    d_nn = D()
    
d_optimizer = RMSprop(lr=0.001)
d_model = Sequential()
d_model.add(d_nn)
d_model.compile(optimizer=d_optimizer, loss='binary_crossentropy')

#defining generator
def G():
    model = Sequential()
    model.add(Dense(70*100, input_shape=(100,)))
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

g_nn = None
if os.path.exists('g_nn'):
    g_nn = load_model('g_nn')
else:
    g_nn = G()

s_optimizer = RMSprop(lr=0.002)
s_model = Sequential()
s_model.add(g_nn)
s_model.add(d_nn)
s_model.compile(optimizer=s_optimizer, loss='binary_crossentropy')

#noise for generator
def noise_generator():
    return np.random.uniform(-1., 1., size=100)

#generate  fake image for generator error calculation
def error_image(error, out_image):
    out_image = out_image+error    
    return out_image

def b_crossentropy(target, prediction):
    if target == 1.:
        return -np.log(prediction)
    else:
        return -np.log(1.-prediction)

training_est = pnds.DataFrame(columns=["g_err", "d_r_err", "d_f_err", "batch"])

if os.path.exists('training_est.csv'):
    training_est = pnds.read_csv('training_est.csv')


final_time = datetime.datetime(year=2018, month=1, day=19, hour=23, minute=30)
epochs = 100000
batch_size = 5
g_errs = []
d_r_errs = []
d_f_errs = []
#training!
for epoch in range(epochs):
    if datetime.datetime.now() >= final_time:
            break
        
    np.random.shuffle(flags)
    
    for batch in range(int(np.ceil(len(flags)/batch_size))):
        start = batch*batch_size
        end = start+batch_size
        end = min(len(flags), end)
        curr_batch_size = end-start
        curr_batch = np.array(flags[start:end])
        
        y = np.array([1.]*curr_batch_size)
        x = curr_batch
        d_r_err = d_model.train_on_batch(x, y)
        
        noise = np.array([noise_generator() for i in range(curr_batch_size)])
        x = noise
        x = g_nn.predict(x)
        
        y = np.array([0.]*curr_batch_size)
        d_f_err = d_model.train_on_batch(x, y)
        weights_save = d_model.get_weights()
        
        noise = np.array([noise_generator() for i in range(curr_batch_size)])
        y = np.array([1.]*curr_batch_size)
        g_err = s_model.train_on_batch(noise, y)
        d_model.set_weights(weights_save)
        
        g_errs.append(g_err)
        d_r_errs.append(d_r_err)
        d_f_errs.append(d_f_err)
        
        print("g_err = %f, d_r_err = %f, d_f_err = %f, batch = %d" % (g_err, d_r_err, d_f_err, batch))
        training_est = training_est.append({"g_err": g_err, "d_r_err": d_r_err, "d_f_err": d_f_err, "batch": batch}, ignore_index=True)
        training_est.to_csv("training_est.csv")
        
        if datetime.datetime.now() >= final_time:
            break
        
    print("finished epoch %d" % epoch)
    g_nn.save('g_nn')
    d_nn.save('d_nn')
    
    noise = np.array([noise_generator() for i in range(curr_batch_size)])
    images = g_nn.predict(noise)
    images = images + 1.
    images = images/2.
    images = images*255.
    
    for i, image in enumerate(images):
        toimage(image, cmin=0, cmax=255).save('./epochs/epoch'+str(epoch)+'_'+str(i)+'.png')

