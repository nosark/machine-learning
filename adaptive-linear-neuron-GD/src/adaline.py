__author__ = 'kyle nosar'

import numpy as np

class AdalineGD(object):
    """
    Adaptive Linear Neuron classifier

    Parameters:
    ----------
    eta: float
    learning rate between 0.0 - 1.0

    n_iters: int
    number of iterations over the training set.

    Attributes:
    ----------
    w_: 1d-array
    weights after fitting

    errors_: list
    number of misclassifications in every epoch

    """

    def __init__(self, eta=0.01, n_iters=50):
        self.eta = eta
        self.n_iters = n_iters

    def fit(self, X, y):
        """
        fit the training data.

        Parameters:
        ----------
        X: {array-like}, shape = [n_samples, n_features
            Training vectors,
            where n_samples is the number of samples and n_features
            is the number of features in the training data.

        y: {array-like}, shape = [n_samples]
            Target values

        Returns:
        -------
        self: object

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iters):
            #net_input = self.net_input(X)
            output = self.net_input(X) #TODO: should not be returning nonetype(should be dot product)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """ Calculates the net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ Computes linear activation """
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation() >= 0.0, 1, -1)
