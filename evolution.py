from player import Player
import numpy as np
from copy import deepcopy

from recombination import crossover, mutate
from selection import rank_based_selection, roulette_wheel, q_tournament, top_k 

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
            children = [mutate(c, 0.3, 0.6, 0.2) for c in children]
            return children

    def next_population_selection(self, players, num_players):
        return rank_based_selection(players, num_players) 
