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


def mutate(child, pw, pb, var):
    """
    mutate takes a child, which is instance of class Player, 
    and adds gaussian noise to weights and biases given 
    probabilities `pw` and `pb`. 
    """

    net = child.nn
    for k in range(len(child.nn.weights)):
        if np.random.rand() < pw:
            net.weights[k] += np.random.normal(0,
                                               var, size=net.weights[k].shape)

    for k in range(len(child.nn.biases)):
        if np.random.rand() < pb:
            net.biases[k] += np.random.normal(0,
                                              var, size=net.biases[k].shape)

    return child