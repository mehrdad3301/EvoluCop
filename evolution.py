from player import Player
import numpy as np
from copy import deepcopy

from recombination import crossover, mutate
from selection import roulette_wheel, q_tournament

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
            parents = q_tournament(players_copy, num_players, 10)
            children = crossover(parents)
            children = parents
            children = [mutate(c, 0.5, 0.7, 1) for c in children]
            return children

    def next_population_selection(self, players, num_players):
        return roulette_wheel(players, num_players, False)
