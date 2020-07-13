#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 00:27:43 2020

@author: MenesesGHZ
"""

from .read_dataset import mnist_read
from .functions import flat_images, normalization, display_flat_img
from .ann import ANN
import numpy as np
import random as r


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
ADAPTING DATASET:

test_images -> flat_images = 10,000 images 1x748
test_images -> normalization = element value is between or equal to (0,1)
"""
print("ADAPTING TEST DATASET...")
test_images = flat_images(test_images)
test_images = normalization(test_images)
print("done.\n")

"""
ANN
"""
model = ANN()
model.load_weights("weights")

print("\nThe model by default was trained with only 19,000 images from 60,000 images in total.")
print("*If you want to train the model by you own, run: main.py --option train \n\n")

while True:
    number = str()
    while not number.isnumeric():
        number = input("Choose a discret number from the interval (0,9): ").strip()
    number = int(number)
    random_index = r.randint(0,test_labels.tolist().count(number))
    img_index = np.where(test_labels == number)[0][random_index]
    prediction = model.predict(test_images[img_index]).argmax()
    display_flat_img(test_images[img_index])
    print("Prediction by ANN: %i" % prediction)
    print("Real Output: %i\n" % number)
   


    
    






