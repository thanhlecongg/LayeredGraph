# Created by Thanh C. Le at 2019-11-22
import itertools
import math

import numpy as np
import networkx as nx
from algorithm.Tabu_Search.fitness_fuction import *


class candidate:
    def __init__(self, solution, paths, cost):
        self.solution = solution
        self.paths = paths
        self.fitness = cost

    def copy(self):
        return candidate(np.copy(self.solution), np.copy(self.paths), self.fitness)

    def __str__(self):
        return "Solution: {}\nPaths: {}\nFitness: {}".format(self.solution,self.paths,self.fitness)


class tabu_search:
    def __init__(self, max_tabu_list, n_neighbors):
        self.sBest = None
        self.best_candidate = None
        self.max_tabu_list = max_tabu_list
        self.tabu_list = np.array([])
        self.n_neighbors = n_neighbors
        self.tabu_list_index = 1

    def run(self, A, L, Profits,n_items, source,target,iteration=10):
        self.init(A, L, Profits,n_items,source,target)
        for iter in range(iteration):
            self.iteration(A, L, Profits)
            # if iter % 50 == 0:
            #     print("Iteration {}:, Fitness ={}".format(iter, self.best_candidate.fitness))
        return self.sBest

    def init(self, A, L, Profits, n_items,source,target):
        sol = np.zeros(len(Profits)+2,dtype=int)
        sol[1:n_items+1] = self.generate_first_solution(Profits)
        sol[0] = source
        sol[-1] = target
        fitness, paths = compute_Fitness(A, L, Profits, sol)
        s0 = candidate(np.copy(sol), paths.copy(), fitness)
        # print("Init solution: {}".format(s0))
        self.sBest = s0.copy()
        self.best_candidate = s0.copy()

    def iteration(self, A, L, Profits):
        G = nx.from_numpy_matrix(A)
        neighbors = self.generate_neighbors(self.sBest.solution, G, self.n_neighbors)
        neighbors_fitness = []
        neighbors_paths = []
        for neighbor in neighbors:
            fitness, paths = compute_Fitness(A, L, Profits, neighbor)
            neighbors_fitness.append(fitness)
            neighbors_paths.append(paths)
        neighbor_order = np.argsort(neighbors_fitness)
        min_index = 0
        self.best_candidate = candidate(neighbors[neighbor_order[min_index]], neighbors_paths[neighbor_order[min_index]],
                                        neighbors_fitness[neighbor_order[min_index]])
        if self.best_candidate.fitness < self.sBest.fitness:
            self.sBest = self.best_candidate.copy()

    def generate_first_solution(self, Profits):
        return np.argmin(Profits, axis=1)

    def generate_neighbors(self, solution, G, n_neighbors):
        neighbors = []
        index = np.random.randint(1, len(solution) - 1)
        v = solution[index]
        neighbor_v = self.neighbor_vertex(v, G, n_neighbors)
        choice = np.arange(len(neighbor_v))
        np.random.shuffle(choice)
        choice = choice[:n_neighbors]
        for i in range(n_neighbors):
            neighbor = np.copy(solution)
            neighbor[index] = neighbor_v[choice[i]]
            neighbors.append(neighbor)
        return neighbors

    def neighbor_vertex(self, vertex, G, n_neighbors):
        neighbors_v = list(G.neighbors(vertex))
        while (len(neighbors_v) < n_neighbors):
            temp = np.copy(neighbors_v)
            for v in neighbors_v:
                temp = np.unique(np.concatenate((neighbors_v, list(G.neighbors(v)))))
            neighbors_v = temp
        return neighbors_v

    def check_tabu_list(self, solution):
        return solution in self.tabu_list


if __name__ == '__main__':
    sol = [0,2,1,3,0]
    search = tabu_search(5,3)
