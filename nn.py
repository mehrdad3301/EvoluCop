import numpy as np


class NeuralNetwork():

    def __init__(self, sizes):

        # sizes example: [4, 10, 2]

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) / np.sqrt(x) 
                        for (x, y) in zip(sizes[1:], sizes[:-1]) ] 

        self.activation = np.vectorize(self.activation) 

    def activation(self, x):
        x[x>0] = 1 / (1 + np.exp(-x[x>0]))
        x[x<0] = np.exp(x[x<0]) / (1 + np.exp(x[x<0]))
        return x 

    def forward(self, x):
        
        # x example: np.array([[0.1], [0.2], [0.3]])
        a = x.reshape(-1, 1) 
        for w, b in zip(self.weights, self.biases) :
            a = (np.matmul(w, a) + b)

        return float(a)