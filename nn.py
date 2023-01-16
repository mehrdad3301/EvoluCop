import numpy as np


class NeuralNetwork():

    def __init__(self, sizes):

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) 
                        for (x, y) in zip(sizes[1:], sizes[:-1]) ] 

    def activation(self, x):

        return 1 / ( 1 + np.exp(-x) )

    def forward(self, x):

        a = x 
        for w, b in zip(self.weights, self.biases) :
            a = self.activation(np.matmul(w, a) + b)

        return float(a)