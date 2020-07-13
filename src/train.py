#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:43:57 2020

@author: MenesesGHZ
"""
from .read_dataset import mnist_read
from .functions import flat_images, normalization
from .ann import ANN
import numpy as np

"""
READING DATASET:
    
train_images: 60,000 images 28x28.
train_labels: true interpretation of number by index of train_images.
ntrimages: number of training images 60,000.
test_images: never seen 10,000 images 28x28.
test_labels:  true interpretation of number by index of test_images.
nteimages: number of test images 10,000.
"""
print("LOADING DATASET...")
train_images, train_labels, ntrimages, test_images, test_labels, nteimages = mnist_read()
print("done.\n")

"""
ADAPTING TRAINING DATASET:
train_images -> flat_images = 60,000 images 1x748
train_images -> normalization = element value is between or equal to (0,1)
"""
print("ADAPTING TRAINING DATASET")
train_images = flat_images(train_images)
train_images = normalization(train_images)
print("done.\n")

"""
TRUE INDEPENDT VARIABLES:
This is the set of the real outputs.
 The needing of this set is to calculate delta (prediction - data_y)
 to then calculate the weights_delta (delta * input) which is going to tell us
 the amount and the direction to correct each weight.
"""
data_y = np.array( [
        [1,0,0,0,0,0,0,0,0,0], #0
        [0,1,0,0,0,0,0,0,0,0], #1
        [0,0,1,0,0,0,0,0,0,0], #2
        [0,0,0,1,0,0,0,0,0,0], #3
        [0,0,0,0,1,0,0,0,0,0], #4
        [0,0,0,0,0,1,0,0,0,0], #5
        [0,0,0,0,0,0,1,0,0,0], #6
        [0,0,0,0,0,0,0,1,0,0], #7
        [0,0,0,0,0,0,0,0,1,0], #8
        [0,0,0,0,0,0,0,0,0,1], #9
        ])

    
model = ANN()
print("Training Phase Has Been Started.\n   -The weights are saved each 1000 predictions-\n")
for i in range(ntrimages):
    prediction = model.predict(train_images[i]) 
    model.train(data_x=train_images[i],
                data_y=data_y[train_labels[i]],
                prediction=prediction)
    if(i%1000==0 and i != 0):
        model.save_weights("weights")
        print("Weights Have Been Updated :D {}/{}".format(i,ntrimages))
