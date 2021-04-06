from collections import Counter, defaultdict
from itertools import product, permutations
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def sum_to_n(n, size, limit=None, nMin=1):
    """Produce all lists of `size` positive integers in decreasing order
    that add up to `n`."""
    if size == 1:
        yield [n] 
        return
    if limit is None:
        limit = n 
    start = max((n + size - 1) // size, nMin)
    stop = min(limit, n - size + 1) + 1 
    for i in range(start, stop):
        for tail in sum_to_n(n - i, size - 1, i, nMin=nMin):
            yield [i] + tail


def get_all_possible_scales_general(i, nI=240, iLimit=80, nMin=4):
    ints = []
    timeS = time.time()
    print(i)
    print(nI, i, iLimit, nMin)
    for partition in sum_to_n(nI, i, limit=iLimit, nMin=nMin):
        ints.append([float(x)*(1200./float(nI)) for x in partition])
    print(len(ints), ' scales found after ...')
    print((time.time()-timeS)/60., ' minutes')
    return np.array(ints)


def possible_scales(d, lo, hi, N):
    X = list(np.linspace(lo, hi, N))
    return np.array(list(set(permutations(X))))


def expand_set(scales):
    all_scales = set()
    for s in scales:
        all_scales = all_scales.union(set(permutations(s)))
    return np.array(list(all_scales))


def d_by_c(df7, dist):
    cont = df7.Continent
    cuniq = np.unique(cont)
    ckey = {c:i for i, c in enumerate(cuniq)}
    cidx = np.array([ckey[c] for c in cont])
    cdist = defaultdict(dict)
    for i in range(len(cuniq)):
        for j in range(len(cuniq)):
            cdist[i][j] = dist[cidx==i][:,cidx==j]
        
    return ckey, cdist


def continent_overlap(df):
    cont = df.Continent
    cuniq = np.unique(cont)

    cuniq = ['Western', 'East Asia', 'South Asia', 'Middle East', 'Oceania', 'South America',
       'Africa', 'South East Asia']

    overlap = np.zeros((len(cuniq), len(cuniq)), float)
    ofrac = np.zeros((len(cuniq), len(cuniq)), float)
    onorm = np.zeros((len(cuniq), len(cuniq)), float)
    onorm2 = np.zeros((len(cuniq), len(cuniq)), float)
    for i in range(len(cuniq)):
        for j in range(len(cuniq)):
            if i == j:
                continue
            u1 = set(df.loc[(df.Continent==cuniq[i]), 'disc'])
            u2 = set(df.loc[(df.Continent==cuniq[j]), 'disc'])
            l1 = len(u1)
            l2 = len(u1.intersection(u2))

            overlap[i,j] = l2
            ofrac[i,j] = float(l2 / l1)
            onorm[i,j] = float(l2 / (l1 * len(u2)**0.5))
            onorm2[i,j] = float(l2 / (l1 + len(u2)))

    return cuniq, overlap, ofrac, onorm, onorm2


def plot_overlap(o, l):
    fig, ax = plt.subplots()
    im = ax.imshow(o)
    fig.colorbar(im)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(l)))
    ax.set_xticklabels(l, rotation=90)
    ax.set_yticks(np.arange(len(l)))
    ax.set_yticklabels(l, rotation=0)
        

    

