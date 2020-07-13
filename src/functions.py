#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:45:24 2020

@author: MenesesGHZ
"""
import numpy as np

"""
FLAT IMAGES:
Convert a set of images i*j dimension to 1 x (i x j) dimension
"""
def flat_images(images_array):

    def flat_image(image):
        img = list()
        for array in image:
            img.extend(array)
        return img
    
    output = list()
    for image in images_array:
        output.append(flat_image(image))
    return np.array(output)


"""
OWN NORMALIZATION:

Convert the rgb values
to values between or equal to (0,1)
"""
def normalization(numpy_array,max_num = 255.0):
    numpy_array = np.array(object=numpy_array,dtype=float)
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[i])):
            numpy_array[i][j] /= max_num
    return numpy_array


"""
DISPLAY A FLAT IMG:
    
Display a flat img of 1 x (i x j) 
to the wanted dimesion width x height 

"""

import matplotlib.pyplot as plt

def display_flat_img(flat_img,width=28,height=28):
    # Reshapping the flat_img into an image of width * height
    img = np.zeros(shape=(width,height))
    flat_img_index = 0
    for i in range(width):
        for j in range(height):
            img[i][j] = flat_img[flat_img_index] * 255
            flat_img_index += 1

    plt.imshow(img)
    plt.title("Test Image")
    plt.grid()
    plt.show(block=False)
        
    
    
         