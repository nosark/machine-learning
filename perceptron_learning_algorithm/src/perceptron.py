#!/usr/bin/env python
__author__ = 'kyle nosar'


import numpy as np


class Perceptron(object):

    '''
    Perceptron Class (Based off of Rosenblatt's Perceptron Rule)
    Params
    ===========================================================
    eta (float): The learning rate ( float between 0 and 1)
    n_iters (int) : Number of passes over the training set

    Attributes
    ===========================================================
    Note: all attributes not created upon initialization will be followed by underscore
    example self.weights_

    weights_ (1d array): weights after fitting the data
    errors_ (list): number of misclassifications in every epoch
    '''

    def __init__(self, eta=0.01, n_iters=10):
        self.eta = eta
        self.n_iters = n_iters


        '''
        Fitting the data

        Parameters
        ===========================================================
        X {array-like}: shape = [n_samples, n_features]
        Training vectors. where n_samples is the number
        of samples and n_features is the number of features.
        y {array-like}: shape = [n_samples]
        Target values.

        returns self : object
        '''
    def fit(self, X, y):
        self.weights_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iters):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self


    '''
        net_input
        params: X (training vector)
        Calculates net input
    '''
    def net_input(self, X):
        #calculates vector dot product w^t x( where ^t is transpose)
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    '''
    Return class label after unit step
    '''
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
