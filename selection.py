import numpy as np 
from copy import deepcopy

def top_k(players, k):
    return sorted(players)[-k:]

def rank_based_selection(players, k):
    players.sort()
    p = np.arange(len(players)) ** 2
    p = p / p.sum()
    return deepcopy(list(np.random.choice(players, size=k, p=p)))

def q_tournament(players, num_players, q):
    new_players = []
    for _ in range(num_players):
        tournoment = np.random.choice(players, q)
        new_players.append(deepcopy(max(tournoment)))
    return new_players

def roulette_wheel(players, num_players, replace=True):

    fitness = [p.fitness for p in players]
    fitness = fitness / np.sum(fitness)
    return deepcopy(list(np.random.choice(players, 
    size=num_players, p=fitness, replace=replace)))

    
