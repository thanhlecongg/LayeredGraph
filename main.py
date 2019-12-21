# Created by Thanh C. Le at 2019-11-23
import numpy as np
import networkx as nx

from algorithm.Greedy.greedy import greedy_search
from algorithm.Simulated_Annueling.simulated_annueling import simulated_annealing
from algorithm.Tabu_Search.tabu_search import tabu_search
from algorithm.Tabu_Search.fitness_fuction import compute_Fitness
import argparse

def get_distribution(array):
    return np.mean(np.array(array)), np.std(np.array(array))

def random_data():
    parser = argparse.ArgumentParser("Create Data")
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    dataset = "random_data/data_{}/".format(args.seed)
    Profits = np.load(dataset + "Profits.npy")
    n_items = len(Profits)
    A = np.load(dataset + "Transport.npy")
    L = np.load(dataset + "Licensing.npy")
    with open(dataset + "pairs.txt", 'r') as f:
        for line in f:
            s, t = line.split()
            res_path = "Source({})_Target({})".format(s, t)
            print("=========" + res_path + "=========")
            with open(dataset + res_path + "/stats.txt") as g:
                for line in g:
                    l = line.split()
                    if l[0] == "Minimum:":
                        print("Minimum: {}".format(l[-1]))
            greedy = greedy_search()
            _, greedy_res = greedy.run(A, L, Profits, int(s), int(t))
            print("Greedy: {}\n".format(greedy_res))
            print("Tabu:")
            tabu = []
            for i in range(5):
                search = tabu_search(max_tabu_list=5, n_neighbors=3)
                tabu_res = search.run(A, L, Profits, n_items, int(s), int(t), 100)
                tabu.append(tabu_res.fitness)
            mean_tabu, std_tabu = get_distribution(tabu)
            print("SA:")
            sa = []
            for i in range(5):
                search = simulated_annealing()
                SA_res = search.run(A, L, Profits, n_items, int(s), int(t), 300)
                sa.append(SA_res.fitness)
            mean_sa, std_sa = get_distribution(sa)
            print("{}\n{}\n{}\n".format(greedy_res, mean_tabu, mean_sa))
def tsp_data():
    parser = argparse.ArgumentParser("Create Data")
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    dataset = "data_from_tsp/random_{}/".format(args.seed)
    Profits = np.load(dataset + "Profits.npy")
    n_items = len(Profits)
    A = np.load(dataset + "Transport.npy")
    L = np.load(dataset + "Licensing.npy")
    dimension = len(A)
    s = np.random.randint(0, dimension)
    t = np.random.randint(0, dimension)
    greedy = greedy_search()
    _, greedy_res = greedy.run(A, L, Profits, int(s), int(t))
    print("{}\t{}".format(s, t))
    print("Greedy:")
    print("{}".format(greedy_res))
    print("SA:")
    sa = []
    for i in range(10):
        search = simulated_annealing()
        simu_res = search.run(A, L, Profits, n_items, int(s), int(t), 1000)
        print("{}".format(simu_res.fitness))
        sa.append(simu_res.fitness)
    print("Tabu:")
    tabu = []
    for i in range(10):
        search = tabu_search(max_tabu_list=10, n_neighbors=5)
        tabu_res = search.run(A, L, Profits, n_items, int(s), int(t), 200)
        print("{}".format(tabu_res.fitness))
        tabu.append(tabu_res.fitness)

if __name__ == '__main__':
    random_data()




