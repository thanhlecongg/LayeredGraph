# Created by Thanh C. Le at 2019-11-23
import os

import numpy as np
import networkx as nx
import argparse
from random_connected_graph import create_graph



def create_new_data(path, dimension, n_items, range, num_edges):
    G = create_graph(dimension,num_edges)
    A = np.zeros(dimension ** 2, dtype=int).reshape(dimension, dimension)
    L = np.zeros(dimension ** 2, dtype=int).reshape(dimension, dimension)
    for x, y in G.edges():
        A[x, y] = np.random.randint(1, range)
        A[y, x] = A[x, y]
        L[x, y] = np.random.randint(1, range)
        L[y, x] = L[x, y]
    profits = np.random.randint(int(-0.3 * range), range, size=dimension * n_items).reshape(n_items, dimension)
    profits[profits==0] = 1
    np.save(path + "Transport.npy", A)
    np.save(path + "Licensing.npy", L)
    np.save(path + "Profits.npy", profits)
    return A, L, profits


def load_data(path):
    A = np.load(path + "Transport.npy")
    L = np.load(path + "Licensing.npy")
    profits = np.load(path + "Profits.npy")
    return A, L, profits


def build_layer_graph(A, profits):
    n_items = len(profits)
    dimension = len(A)
    layered_dimension = n_items * dimension
    layered_A = np.zeros(layered_dimension ** 2).reshape(layered_dimension, layered_dimension)

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

def generate_data_random(num_nodes, num_edges, num_items, seed=1, n_pair = 5):
    dimension = num_nodes
    n_items = num_items
    random_range = 50
    num_edges = num_edges
    directory = "random_data/data_{}/".format(seed)
    try:
        os.mkdir(directory)
    except OSError:
        pass
    A, L, profits = create_new_data(directory, dimension, n_items, random_range, num_edges)
    n_items = len(profits)
    dimension = len(A)
    layered_A = build_layer_graph(A, profits)
    G = nx.from_numpy_matrix(layered_A,create_using=nx.DiGraph())
    node_done = []
    with open(directory + "pairs.txt", "w") as g:
        for i in range(n_pair):
            node1 = np.random.randint(0, dimension)
            node2 = np.random.randint(0, dimension)
            while node2 == node1:
                node2 = np.random.randint(0, dimension)
            while (node1, node2) in node_done:
                node1 = np.random.randint(0, dimension)
                node2 = np.random.randint(0, dimension)
                while node2 == node1:
                    node2 = np.random.randint(0, dimension)
            node_done.append((node1,node2))

            sub_path = "Source({})_Target({})/".format(node1, node2)
            all_simple_paths = nx.all_simple_paths(G, node1, (n_items - 1) * dimension + node2)
            paths = np.array([path for path in all_simple_paths])
            path_cost = np.array([cost_path(G, L, path) for path in paths], dtype=int)
            g.write("{}\t{}\n".format(node1, node2))
            min_index = np.argmin(path_cost)
            best = list(paths)[int(min_index)]
            for i in range(len(best) - 1):
                print("({},{})={},{}".format(best[i], best[i + 1], G[best[i]][best[i + 1]],
                                             L[best[i] % dimension][best[i + 1] % dimension]))
            try:
                os.mkdir(directory + sub_path)
            except:
                pass
            with open(directory + sub_path + "stats.txt", "w") as f:
                f.write("Dimension: {}\n".format(dimension))
                f.write("Number of items: {}\n".format(n_items))
                f.write("Random range: {}\n".format(random_range))
                f.write("Num_edges: {}\n".format(num_edges))
                f.write("Seed: {}\n".format(seed))
                f.write("Source: {}\n".format(node1))
                f.write("Target: {}\n".format(node2))
                f.write("Minimum: {}\n".format(path_cost[min_index]))
                for x in paths[int(min_index)]:
                    f.write("{}\n".format(x))

if __name__ == '__main__':
    generate_data_random(8,20,2,seed=10,n_pair=3)

