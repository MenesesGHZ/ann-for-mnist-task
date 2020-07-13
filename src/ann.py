#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:39:29 2020

@author: MenesesGHZ

"""

import numpy as np
import json

class ANN:
    def __init__(self,input_size=784,output_size=10):
        self.weights = np.ones(shape=(output_size,input_size))
        self.output = np.zeros(shape=(output_size))
        self.alpha = 0.01 
        
        
    def predict(self,x):
        """
        FORWARD PROPAGATION
        """
        def w_sum(x,weights):
            sum_x_w = float()
            for i in range(len(x)):
                sum_x_w += x[i] * weights[i]
            return sum_x_w
        
        for i in range(len(self.output)):
            self.output[i]=w_sum(x,self.weights[i]) 
            
        return self.output
    
    def train(self,data_x,data_y,prediction):
        """
        REDUCING THE ERROR
        """
        delta = prediction - data_y
        weights_delta = np.zeros(shape=self.weights.shape)
        for i in range(len(weights_delta)):
            for j in range(len(weights_delta[i])):
                weights_delta[i][j] = delta[i]*data_x[j] 
                
        self.weights = self.weights - weights_delta * self.alpha
    
    def save_weights(self,filename):
        with open(filename+".json","w") as file:
            json_instance = json.dumps(self.weights.tolist())
            file.write(json_instance)
            
    
    def load_weights(self,filename):
        with open(filename+'.json', 'r') as file: 
            self.weights = np.array(json.load(file))