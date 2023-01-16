import numpy as np 

def crossover(players):

    def cross_weights(p1, p2):
        n1, n2 = p1.nn, p2.nn
        for i in range(len(n1.weights)):
            temp = n1.weights[i][:, ::2].copy()
            n1.weights[i][:, ::2] = n2.weights[i][:, ::2]
            n2.weights[i][:, ::2] = temp

    def cross_biases(p1, p2):
        n1, n2 = p1.nn, p2.nn
        for i in range(len(n1.biases)):
            temp = n1.biases[i][::2].copy()
            n1.biases[i][::2] = n2.biases[i][::2]
            n2.biases[i][::2] = temp

    np.random.shuffle(players)
    for p1, p2 in zip(players[::2], players[1::2]):
        cross_weights(p1, p2)
        cross_biases(p1, p2)

    return players
