# Created by Thanh C. Le at 2019-12-10
import numpy as np
import networkx as nx

def compute_Fitness(A, L, Profits, solution):
    paths = []
    fitness = 0
    for i in range(len(solution) - 1):
        G = nx.from_numpy_matrix(A + L)
        path = nx.algorithms.dijkstra_path(G, solution[i], solution[i + 1])
        for j in range(len(path) - 1):
            x = path[j]
            y = path[j + 1]
            fitness += G[x][y]['weight']
            G[x][y]['weight'] = A[x][y]
            G[y][x]['weight'] = A[x][y]
            paths.append(path)
        if i > 0 and i < len(solution) - 2:
            fitness += Profits[i - 1, solution[i]]
    return fitness, paths

if __name__ == '__main__':
    Profits = np.array([
        [1, 3, 4, 5, 9, 1, 3],
        [2, 3, 1, 9, 9, 1, 3],
        [3, 4, 2, 1, 2, 1, 3],
        [3, 4, 2, 1, 2, 1, 3]
    ])
    A = np.array([
        [0, 1, 3, 1, 0, 3, 8],
        [1, 0, 2, 9, 0, 6, 0],
        [3, 2, 0, 1, 1, 0, 2],
        [1, 9, 1, 0, 4, 5, 6],
        [0, 0, 1, 4, 0, 8, 9],
        [3, 6, 0, 5, 8, 0, 1],
        [8, 0, 2, 6, 9, 1, 0]
    ])

    L = np.array([
        [0, 1, 3, 1, 0, 3, 8],
        [1, 0, 2, 9, 0, 6, 0],
        [3, 2, 0, 1, 1, 0, 2],
        [1, 9, 1, 0, 4, 5, 6],
        [0, 0, 1, 4, 0, 8, 9],
        [3, 6, 0, 5, 8, 0, 1],
        [8, 0, 2, 6, 9, 1, 0]
    ])
    candidate = [2,3,0,1]
    print(compute_Fitness(A,L,Profits,candidate))

