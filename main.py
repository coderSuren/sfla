#!/usr/bin/env python3
import sys
from collections import namedtuple
import itertools
import numpy as np
import random
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


def snn(start, remaining, path):
    if len(remaining) > 0:
        dist = sorted(remaining, key=lambda x:euclidean_distance(nodelist[start],nodelist[x]),
                      reverse = True)
        dest = dist.pop()
        path.append(dest)
        snn(dest, dist, path)
    else:
        return


def frog_rand():
    return np.array(rng.permutation(np.fromiter(nodelist.keys(), dtype=int)))

def frog_gen():
    nodes = np.fromiter(nodelist.keys(), dtype=int)
    best = list()
    ret = list()
    for i in nodes:
        remaining = nodes.copy()
        tmp = list()
        snn(i, remaining, tmp)
        best.append(tmp)
    for route in best:
        for _ in range(9):
            ret.append(rng.permutation(route))
        ret.append(route)
    return np.array(ret)
        

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
        if random.random() < v:
            submemeplex.append(memeplex[k])
    return submemeplex

def slice_gen(left,right):
    ind1 = random.randint(left, right)
    ind2 = random.randint(left, right)
    start = min(ind1, ind2)
    end = max(ind1, ind2)
    if start == end:
        return slice_gen(left, right)
    else:
        return (start, end)


# Takes src[start:end] puts it in tmp[start:end]
# Takes the remaining elements from dest[] and put it in tmp in the same order
# Leaves out the elemenets in dest[] if it is already in src[start:end]
def arr_swap(src, dest):
    start, end = slice_gen(0, len(src)-1)
    tmp = np.zeros(src.shape)
    best_path = src[start:end]
    tmp[start:end] = best_path
    curr_tmp = 0
    curr_local = 0
    while curr_tmp < len(src):
        if curr_tmp in range(start,end):
            curr_tmp += 1
            continue
        else:
            if dest[curr_local] not in best_path:
                tmp[curr_tmp] = dest[curr_local]
                curr_tmp += 1
            curr_local += 1
    return tmp


def local_search(frogs, submemeplex):
    local_max = frogs[submemeplex[0]]
    local_min = frogs[submemeplex[-1]]
    tmp = arr_swap(local_max, local_min)
    if path_len(tmp) < path_len(local_min):
        frogs[submemeplex[-1]] = tmp
    return frogs


def sfla(num_frogs, num_memeplexes, submemplex_iter, total_iteration):
    # Initial frog gen
    frogs = frog_gen()
    # Sorting to find the initial global minimum
    memeplexes = frog_sort(frogs, num_memeplexes)
    sol = frogs[memeplexes[0, 0]].copy()
    for _ in range(total_iteration):
        memeplexes = memeplexes_shuffle(memeplexes)
        for memeplex in memeplexes:
            # Taking the a sample of frog from the memeplex
            # probability that a frog is picked is inversely proportional to
            # path length
            submemeplex = submemeplex_gen(memeplex)
            for _ in range(submemplex_iter):
                tmp = frogs[submemeplex[-1]]
                frogs = local_search(frogs, submemeplex)
                tmp2 = frogs[submemeplex[0]]
                print(f"{tmp},{path_len(tmp)}")
                print(f"{tmp2},{path_len(tmp2)}")
        memeplexes = frog_sort(frogs, num_memeplexes)
        # updating global min
        new_sol = frogs[memeplexes[0,0]].copy()
        if path_len(new_sol) < path_len(sol):
            sol = new_sol.copy()
    return sol

def swap_2opt(route, i, j):
    new_route = route[:i]
    new_route.extend(reversed(route[i:j+1]))
    new_route.extend(route[j+1:])
    return new_route


def two_opt(route):
    improved = True
    best_found_route = route
    best_found_route_cost = path_len(route)
    while improved:
        improved = False
        for i in  range(1, len(best_found_route) - 1):
            for j in range(i+1, len(best_found_route)):
                new_route = swap_2opt(best_found_route,i,j)
                new_route_cost = path_len(new_route)
                if new_route_cost < best_found_route_cost:
                    best_found_route = new_route
                    best_found_route_cost = new_route_cost
                    improved = True
    return best_found_route

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
    G = nx.Graph()
    rng = np.random.default_rng(69420)

    nodelist = nodelist_create(sys.argv[1])
    sol = sfla(num_frogs = 10*len(nodelist), submemplex_iter=len(nodelist),
               num_memeplexes=10, total_iteration= 20)
    new_sol = two_opt(sol.tolist())
    print(sol, path_len(sol))
    print(new_sol, path_len(new_sol))

    nx.add_path(G, sol)
    plt.subplot(121)
    nx.draw(G, pos=nodelist, with_labels=True)
    G2 = nx.Graph()
    nx.add_path(G2, new_sol)
    plt.subplot(122)
    nx.draw(G2, pos=nodelist, with_labels=True)
    plt.show()
