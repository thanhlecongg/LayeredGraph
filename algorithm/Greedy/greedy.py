# Created by Thanh C. Le at 2019-12-09
import numpy as np
import networkx as nx
def build_layer_graph(A, profits):
    n_items = len(profits)
    dimension = len(A)
    layered_dimension = n_items * dimension
    layered_A = np.zeros(layered_dimension ** 2).reshape(layered_dimension, layered_dimension)
    print(np.linalg.norm(A-A.T))
    for i in range(n_items):
        layered_A[i * dimension:(i + 1) * dimension, i * dimension:(i + 1) * dimension] = A
        if i < n_items-1:
            for j in range(dimension):
                layered_A[i * dimension + j, (i + 1) * dimension + j] = profits[i, j]
                layered_A[(i + 1) * dimension + j, i * dimension + j] = 0
    return layered_A

def cost_path(G,L,path):
    dimension = len(L)
    copy_L = np.copy(L)
    cost = 0
    for i in range(len(path)-1):
        cost += G[path[i]][path[i + 1]]['weight']
        if copy_L[path[i]%dimension][path[i + 1]%dimension]:
            cost += copy_L[path[i]%dimension][path[i + 1]%dimension]
            copy_L[path[i]%dimension][path[i + 1]%dimension] = 0
    return cost

class greedy_search:
    def __init__(self):
        pass

    def run(self,A,L,profits,s,t):
        n_items = len(profits)
        dimension = len(A)
        layered_graph = build_layer_graph(A,profits)
        G = nx.from_numpy_matrix(layered_graph,create_using=nx.DiGraph())
        shortest_path = nx.shortest_path(G,s,t+(n_items-1)*dimension)
        return shortest_path, cost_path(G,L,shortest_path)
