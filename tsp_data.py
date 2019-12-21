# Created by Thanh C. Le at 2019-12-10
import os

import numpy as np
import networkx as nx
from random_connected_graph import create_graph
def euclid_distance(a,b):
    return np.linalg.norm(a - b)


def read_tsp_data_noneu(path="bays29.tsp.txt"):
    with open(path, "r") as f:
        start = False
        dimension = None
        A = np.array([])
        for l in f:
            line = l.split()
            if line[0] == 'DIMENSION:':
                dimension = int(line[1])
            if line[0] == 'DISPLAY_DATA_SECTION':
                break
            if start:
                A = np.append(A, np.array(line, dtype=int))
            if line[0] == 'EDGE_WEIGHT_SECTION':
                start = True

    return A.reshape(dimension, dimension), dimension

def read_tsp_data_eu(path="tsp_data/eil51.tsp.txt"):
    coord = []
    with open(path, "r") as f:
        start = False
        dimension = None
        for l in f:
            line = l.split()
            if line[0] == 'DIMENSION:':
                dimension = int(line[1])
            if line[0] == 'EOF':
                break
            if start:
                coord.append([line[1],line[2]])
            if line[0] == 'NODE_COORD_SECTION':
                start = True
    coord = np.array(coord,dtype=int)
    A = np.array([euclid_distance(coord[i],coord[j]) for i in range(dimension) for j in range(dimension)])
    return A.reshape(dimension, dimension), dimension

def create_new_data_from_tsp(tsp_data, tsp_dimension, path="data_from_tsp/random_1/"):
    try:
        os.mkdir(path)
    except OSError:
        pass
    G = create_graph(tsp_dimension, int(0.2*(tsp_dimension*(tsp_dimension-1))))
    A = np.zeros(tsp_dimension ** 2, dtype=int).reshape(tsp_dimension, tsp_dimension)
    L = np.zeros(tsp_dimension ** 2, dtype=int).reshape(tsp_dimension, tsp_dimension)
    max = np.max(tsp_data)
    for x, y in G.edges():
        A[x, y] = tsp_data[x, y]
        A[y, x] = A[x, y]
        L[x, y] = np.random.randint(0, max)
        L[y, x] = L[x, y]
    n_items = np.random.randint(int(0.2 * tsp_dimension), int(0.8 * tsp_dimension))
    profits = np.random.randint(int(-0.3 * max), max, size=tsp_dimension * n_items).reshape(n_items, tsp_dimension)
    profits[profits == 0] = 1.0
    np.save(path + "Transport.npy", A)
    np.save(path + "Licensing.npy", L)
    np.save(path + "Profits.npy", profits)
    return A, L

A, dimension = read_tsp_data_noneu("tsp_data/bays29.tsp.txt")
create_new_data_from_tsp(A,dimension,"data_from_tsp/random_1/")