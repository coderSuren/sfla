#!/usr/bin/env python3
import sys
from collections import namedtuple
import numpy as np

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


def fitness(frog):
    tmp = 0
    for k,_ in enumerate(frog[:-1]):
        tmp += euclidean_distance(nodelist[frog[k]], nodelist[frog[k+1]])
    return 1/tmp


def frog_gen(num_frogs):
    nodes = np.fromiter(nodelist.keys(), dtype=int)
    return  np.array([np.random.permutation(nodes) for _ in range(num_frogs)])

def sort_frogs(frogs, num_memeplex, fitness):
    fitness_list = np.array(list(map(fitness, frogs)))
    sorted_fitness = np.argsort(fitness_list)
    # stores the indices
    # hence dtype=int
    memeplexes = np.zeros((num_memeplex, int(frogs.shape[0]/num_memeplex)),
                          dtype=int)
    for j in range(memeplexes.shape[1]):
        for i in range(num_memeplex):
            memeplexes[i, j] = sorted_fitness[i+(num_memeplex*j)]
    return memeplexes

def frog_update(u, v):
    for k,v in enumerate(v):
        if np.random.rand() > 0.5:
            u[k] = v

def frog_valid(frog):
    return len(frog) == len(np.unique(frog))

def frog_worse(u, v, fitness):
    return fitness(u) < fitness(v)

def local_search(frogs, memeplex, fitness):
    frog_worst = frogs[memeplex[-1]]
    frog_best = frogs[memeplex[0]]
    frog_greatest = frogs[0]
    frog_worst_new = frog_worst
    # Move wrt to best frog
    frog_update(frog_worst_new, frog_best)
    if not frog_valid(frog_worst_new) or\
        frog_worse(frog_worst_new, frog_worst, fitness):
        frog_worst_new = frog_worst
        # Move wrt to greatest frog
        frog_update(frog_worst_new, frog_greatest)
    if not frog_valid(frog_worst_new) or\
            frog_worse(frog_worst_new, frog_worst, fitness):
                # generate new random frog
                frog_worst_new = frog_gen(1)
    frogs[memeplex[-1]] = frog_worst_new
    return frogs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
    nodelist = nodelist_create(sys.argv[1])
