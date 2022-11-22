#!/usr/bin/env python3
import sys
from collections import namedtuple
import numpy as np
from random import randint, random
import networkx as nx
import matplotlib.pyplot as plt

coords = namedtuple("coords", ["x", "y"])

def usage():
    print("Usage: cfla FILE\nWhere is a .tsp file")
    exit()

def nodelist_create(file):
    nodelist = dict()
    with open(file) as f:
        for line in f:
            # Skipping lines until the edge data starts
            if line != "NODE_COORD_SECTION\n":
                continue
            else:
                for line in f:
                    # Getting all the data untill EOF is indicated
                    if line != "EOF\n":
                        line = line.split()
                        nodelist[int(line[0])] = coords(float(line[1]),
                                                   float(line[2]))
                    else:
                        break
    return nodelist

def euclidean_distance(u, v):
    return np.sqrt(np.sum(np.square(np.subtract(u,v))))

def path_len(frog):
    tmp = 0
    for k,v in enumerate(frog[1:]):
        tmp += euclidean_distance(nodelist[v], nodelist[frog[k-1]])
    return tmp



def frog_gen(num_frogs):
    nodes = np.fromiter(nodelist.keys(), dtype=int)
    return  np.array([np.random.permutation(nodes) for _ in range(num_frogs)])

def frog_sort(frogs, num_memeplex):
    fitness_list = np.array(list(map(path_len,frogs)))
    sorted_fitness = np.argsort(fitness_list)
    # stores the indices
    # hence dtype=int
    memeplexes = np.zeros((num_memeplex, int(frogs.shape[0]/num_memeplex)),
                          dtype=int)
    for j in range(memeplexes.shape[1]):
        for i in range(num_memeplex):
            memeplexes[i, j] = sorted_fitness[i+(num_memeplex*j)]
    return memeplexes

def frog_valid(frog):
    return len(frog) == len(np.unique(frog))



def memeplexes_shuffle(memeplexes):
    tmp = memeplexes.flatten()
    np.random.shuffle(tmp)
    tmp = tmp.reshape((memeplexes.shape[0], memeplexes.shape[1]))
    return tmp

def submemeplex_gen(memeplex):
    submemeplex = list()
    n = len(memeplex)
    # Probability that a frog is picked
    p = np.fromiter([2*(n+1-j)/(n*(n+1)) for j in range(0,n)], dtype=float)
    # Normalizing to 0,1
    p = (p - np.min(p)) / (np.max(p) - np.min(p))
    for k,v in enumerate(p):
        if random() < v:
            submemeplex.append(memeplex[k])
    return submemeplex

# Put a slice from start to end from src to dest and returns a copy
def frog_update(src, dest):
    tmp = dest.copy()
    start = randint(0,len(src)//2)
    end = randint(start,len(src))
    tmp[start:end] = src[start:end]
    while not frog_valid(tmp):
        tmp = dest.copy()
        start = randint(0,len(src))
        end = randint(start,len(src))
        tmp[start:end] = src[start:end]
    return tmp

def local_search(frogs, submemeplex):
    global_max = frogs[0]
    local_max = frogs[submemeplex[0]]
    local_min = frogs[submemeplex[-1]]

    tmp = frog_update(local_max, local_min)

    if path_len(tmp) > path_len(local_min):
        local_max = global_max.copy()
        tmp = frog_update(local_max, local_min)
    if path_len(tmp) > path_len(local_min):
        tmp = frog_gen(1)[0]
    local_min = tmp
    return frogs

def sfla(num_frogs, num_memeplexes, submemplex_iter, total_iteration):
    frogs = frog_gen(num_frogs)
    memeplexes = frog_sort(frogs, num_memeplexes)
    sol = frogs[memeplexes[0, 0]].copy()
    for _ in range(total_iteration):
        memeplexes = memeplexes_shuffle(memeplexes)
        for memeplex in memeplexes:
            submemeplex = submemeplex_gen(memeplex)
            tmp = list()
            for _ in range(submemplex_iter):
                frogs = local_search(frogs, submemeplex)
                tmp.append(path_len(frogs[submemeplex[-1]]))
            print(tmp)
        memeplexes = frog_sort(frogs, num_memeplexes)
        new_sol = frogs[memeplexes[0,0]].copy()
        if path_len(new_sol) < path_len(sol):
            sol = new_sol.copy()
        print(sol,path_len(sol))
    return sol


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
    G = nx.Graph()
    rng = np.random.default_rng(69420)

    nodelist = nodelist_create(sys.argv[1])
    sol = sfla(num_frogs=10*len(nodelist), submemplex_iter=len(nodelist),
                                    num_memeplexes=10, total_iteration= 10)
    print(sol, path_len(sol))
    nx.add_path(G, sol)
    fig, ax = plt.subplots(1,1)
    nx.draw(G, pos=nodelist, with_labels=True)
    plt.show()
