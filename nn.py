import numpy as np
from math import exp


class NeuralNetwork():

    def __init__(self, sizes):

        # sizes example: [4, 10, 2]

        self.baises = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) / np.sqrt(x) 
                        for (x, y) in zip(sizes[1:], sizes[:-1]) ] 

        self.activation = np.vectorize(self.activation) 

    def activation(self, x):
        x[x>0] = 1 / (1 + exp(-x))
        x[x<0] = exp(x) / (1 + exp(x))
        return x 

    def forward(self, x):
        
        # x example: np.array([[0.1], [0.2], [0.3]])
        for w, b in zip(self.weights, self.biases) :
            a = self.activation(w.T @ x + b)

        return a 