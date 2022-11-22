#!/usr/bin/env python3
import sys
from collections import namedtuple
import numpy as np
from random import randint

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
    # Last node to 
    tmp += euclidean_distance(nodelist[frog[0]], nodelist[frog[-1]])
    return tmp



def frog_gen(num_frogs):
    nodes = np.fromiter(nodelist.keys(), dtype=int)
    return  np.array([np.random.permutation(nodes) for _ in range(num_frogs)])

def frog_sort(frogs, num_memeplex):
    fitness_list = np.array(list(map(path_len,frogs)))
    min_dist = np.min(fitness_list)
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
    start = randint(0,len(u)//2)
    end = randint(start+2,len(u))
    tmp = u
    tmp[start:end] = v[start:end]
    while not frog_valid(tmp):
        frog_update(u, v)
    return tmp

def frog_valid(frog):
    return len(frog) == len(np.unique(frog))

def local_search(frogs, memeplex):
    frog_worst = frogs[memeplex[-1]]
    frog_best = frogs[memeplex[0]]
    frog_greatest = frogs[0]
    # Move wrt to best frog
    frog_worst_new = frog_update(frog_worst, frog_best)
    if path_len(frog_worst_new) > path_len(frog_worst):
        # Move wrt to greatest frog
        frog_worst_new = frog_update(frog_worst, frog_greatest)
    if path_len(frog_worst_new) > path_len(frog_worst):
                # generate new random frog
        frog_worst_new = frog_gen(1)[0]
    frogs[memeplex[-1]] = frog_worst_new
    return frogs

def memeplexes_shuffle(frogs, memeplexes):
    tmp = memeplexes.flatten()
    np.random.shuffle(tmp)
    tmp = tmp.reshape((memeplexes.shape[0], memeplexes.shape[1]))
    return tmp

def sfla(num_frogs, num_memeplex, iter_memeplex, iter_sols):
    frogs = frog_gen(num_frogs)
    memeplexes = frog_sort(frogs, num_memeplex)
    sol_best = frogs[memeplexes[0, 0]].copy()
    for i in range(iter_sols):
        memeplexes = memeplexes_shuffle(frogs, memeplexes)
        # We are modifying memeplexes, so can't use "for i in memeplexes"
        for j in range(len(memeplexes)):
            for _ in range(iter_memeplex):
                frogs = local_search(frogs, memeplexes[j])
            memeplexes = frog_sort(frogs, num_memeplex)
            sol_best_new = frogs[memeplexes[0, 0]]
            if path_len(sol_best_new) < path_len(sol_best):
                sol_best = sol_best_new
            print(sol_best, path_len(sol_best))
    return sol_best, frogs, memeplexes.astype(int)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
    nodelist = nodelist_create(sys.argv[1])
    sol, frogs, memeplexes = sfla(num_frogs=10*len(nodelist),
                                    num_memeplex=10, iter_memeplex = 10,
                                    iter_sols = 50)
    print(sol, path_len(sol))
