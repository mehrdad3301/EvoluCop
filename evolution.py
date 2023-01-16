from player import Player
import numpy as np
from copy import deepcopy

from recombination import crossover, mutate

class Evolution():

    def __init__(self, mode):
        self.mode = mode

    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]


    def generate_new_population(self, num_players, prev_players=None):

        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            players_copy = deepcopy(prev_players)
            parents = self.roulette_wheel(players_copy, num_players)
            children = crossover(parents)
            children = [mutate(c, 0.5, 0.7, 1) for c in children]
            return children

    def next_population_selection(self, players, num_players):
        return self.top_k(players, num_players)

    def top_k(self, players, k):
        return sorted(players)[-k:]

    def rank_based_selection(self, players, k):
        players.sort()
        p = np.arange(len(players)) ** 2
        p = p / p.sum()
        return deepcopy(np.random.choice(players, size=k, p=p))

    def q_tournament(self, players, num_players, q):
        new_players = []
        for _ in range(num_players):
            tournoment = np.random.choice(players, q)
            new_players.append(deepcopy(max(tournoment)))
        return new_players

    def roulette_wheel(self, players, num_players):

        fitness = [p.fitness for p in players]
        print(sorted(fitness))
        fitness = fitness / np.sum(fitness)
        return list(np.random.choice(players, size=num_players, 
                                p=fitness, replace=False))

    