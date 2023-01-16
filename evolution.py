from hashlib import new
from player import Player
import numpy as np
from config import CONFIG
import copy


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    def copy_players(self, players):
        return [copy.deepcopy(p) for p in players]

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child, pw, pb, var):

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

    def generate_new_population(self, num_players, prev_players=None):

        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            players_copy = self.copy_players(prev_players)
            parents = self.roulette_wheel(players_copy, num_players)
            children = self.crossover(parents)
            children = [self.mutate(c, 0.5, 0.7, 1) for c in children]
            return children

    def next_population_selection(self, players, num_players):
        return self.top_k(players, num_players)

    def top_k(self, players, k):
        return sorted(players)[-k:]

    def rank_based_selection(self, players, k):
        players.sort()
        p = np.arange(len(players)) ** 2
        p = p / p.sum()
        return self.copy_players(np.random.choice(players, size=k, p=p))

    def q_tournament(self, players, num_players, q):
        new_players = []
        for _ in range(num_players):
            tournoment = np.random.choice(players, q)
            new_players.append(copy.deepcopy(max(tournoment)))
        return new_players

    def roulette_wheel(self, players, num_players):

        fitness = [p.fitness for p in players]
        print(sorted(fitness))
        fitness = fitness / np.sum(fitness)
        return self.copy_players(np.random.choice(players, size=num_players, 
                                                    p=fitness, replace=False))

    def sus(self, players):
        pass

    def crossover(self, players):

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
