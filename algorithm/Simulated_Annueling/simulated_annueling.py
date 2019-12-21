# Created by Thanh C. Le at 2019-12-10
import numpy as np

from algorithm.Simulated_Annueling.fitness_function import compute_Fitness
import networkx as nx

class candidate:
    def __init__(self, solution, paths, cost):
        self.solution = solution
        self.paths = paths
        self.fitness = cost

    def copy(self):
        return candidate(np.copy(self.solution), np.copy(self.paths), self.fitness)

    def __str__(self):
        return "Solution: {}\nPaths: {}\nFitness: {}".format(self.solution,self.paths,self.fitness)

class simulated_annealing:
    def __init__(self, init_temp=100, temperate_function="linear"):
        self.sBest = None
        self.best_candidate = None
        self.init_temp = init_temp
        self.function = temperate_function

    def generate_first_solution(self, Profits):
        return np.argmin(Profits, axis=1)

    def init(self, A, L, Profits, n_items,source,target):
        sol = np.zeros(len(Profits)+2,dtype=int)
        sol[1:n_items+1] = self.generate_first_solution(Profits)
        sol[0] = source
        sol[-1] = target
        fitness, paths = compute_Fitness(A, L, Profits, sol)
        s0 = candidate(np.copy(sol), paths.copy(), fitness)
        self.sBest = s0.copy()
        self.candidate = s0.copy()

    def generate_neighbors(self, solution, G):
        index = np.random.randint(1, len(solution) - 1)
        v = solution[index]
        neighbor_v = self.neighbor_vertex(v, G)
        neighbor = np.copy(solution)
        neighbor[index] = neighbor_v
        return neighbor

    def neighbor_vertex(self, vertex, G):
        neighbors_v = list(nx.neighbors(G,vertex))
        l = len(neighbors_v)
        index = np.random.randint(0,l)
        return neighbors_v[index]

    def get_prob(self, delta, T):
        if delta < 0:
            return 1
        else:
            return np.exp(-delta/T)
    def get_temperature(self,T,cooling_rate,function="linear"):
        if function == "linear":
            return T - cooling_rate*self.init_temp

    def iteration(self,A, L, Profits,T):
        G = nx.from_numpy_matrix(A)
        neighbor = self.generate_neighbors(self.candidate.solution,G)
        fitness, paths = compute_Fitness(A, L, Profits, neighbor)
        delta = fitness - self.candidate.fitness
        prob = self.get_prob(delta,T)
        rand = np.random.rand()
        if rand < prob:
            self.candidate = candidate(neighbor,paths,fitness)
            if fitness < self.sBest.fitness:
                self.sBest = self.candidate
        return prob

    def run(self,A, L, Profits,n_items, source, target, n_iteration):
        if self.function == "linear":
            self.cooling_rate = 1/(n_iteration+1)
        self.T = self.init_temp
        self.init(A, L, Profits, n_items,source,target)
        for iter in range(n_iteration):
            self.T = self.get_temperature(self.T,self.cooling_rate)
            prob = self.iteration(A, L, Profits,self.T)
            # if iter % 50 == 0:
            #     print("Iteration {}: T={}, Fitness ={}, Prob={}".format(iter, self.T, self.candidate.fitness, prob))
        return self.sBest

