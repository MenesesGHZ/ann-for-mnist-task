#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:20:13 2020

@author: MenesesGHZ
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--option", "-op", help="test:choose an number and compare with the ANN\ntrain: improve accuracy of the model training with 60,000 images")

args = parser.parse_args()



if __name__ == "__main__":
    if args.option:
        if args.option == "test":
            from src import test
        elif args.option == "train":
            from src import train
        else:
            print("main.py --option train\nor\nmain.py --option test")


    


"""
NOTES:   
    Artificial Neuronal Network
    
    + What is it?
        An artificial neural network is computational implementation about 
        how the brain itself process the information. Where this computational
        implementation works in base of input(s)
        to have some output(s) as a result. Generally the goal 
        is to yield predictons that fits 
        the distribution of acceptable answers.
    
    + What are the components of an ANN?
        The main components of an ANN with their respective purpose are the next ones:
            Node: 
            Input Layer: It is a set of node(s) that has the main purpose to receive the
            input numeric data.
            Output Layer: It is a set of node(s) that has the main purpose to yield the numerical prediction
            result following the forward propagation.
            Weight: It is a numerical variable that is in charge to be multiplied 
            with the data that pass trough to the node. With the main purpose to be changed. 
            If the prediction was not accurated we can manipulate the set of ANN's weights
            to have predictions which fit aceptable results.
                
    + How does it learn?
        All the ANN learns by reducing the error. 
        The error is generated by the unaccuaricy of the prediction in contrast
        with the real result. In order to make right predictions, we are able to adjust
        some variables inside of this network. This variables are called weights.
        In this case we are supervising the predictions that this ANN is doing,
        and telling to the network how amount of error have predicted for each output.
        Calculating the weight delta which is the amount of and the direction 
                
    + How to make a prediction?
        Making a prediction is basically take some numerical input(s)
        and follow the weighted sum of each layer's node until we reach the output layer.
        It is also called forward propagation because you are propaging the data 
        forward trough the network making some transormation 
        to this input data, due to the weights and the activation functions. 
        In order to obtain a suitable prediction.
        
    
    + How to train the ANN?
        In order make the ANN more accurate
        we need to reduce the error that is producing.
        
        First at all, we have to measure the error. 
        Nowadays exist plenty of ways to calculate this prediction's error.
        The basic idea is:
            error = prediction - real_result
            
        Another ideas:
            error_abs = abs(prediction - real_result) = abs(error)
            error_squared = (prediction - real_result)^2 = error^2
              
    
"""
    