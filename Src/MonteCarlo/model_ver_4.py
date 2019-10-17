import argparse
import glob
import os
import sys
import time
import pickle

from itertools import product
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns

import utils
from bias_beta import BETA_BIAS

# PATHS
SRC_DIR = '/home/jmcbride/Scales/Toy_model/Src'
DATA_DIR = '/home/jmcbride/Scales/Toy_model/Data/Raw_tmp'
DATA_DIR_2 = '/home/jmcbride/Scales/Toy_model/Data/Raw'

# Model Parameters
MIarr = np.array([0.] + list(np.arange(50., 110., 10.)))
MAarr = np.array(list(np.arange(400., 600., 50.)) + [1200.])
Narr = np.array([5, 7])

TEMP_MIN = 50. 
TEMP_MAX = 300.
TEMP_LOW_MARGIN = 0.50
TEMP_HI_MARGIN = 1.50
N_TRIALS = 50

ALPHA_W = 0.1
ALPHA_HS = 1.

#MIarr = [50.]
#MAarr = [550.]
#Narr  = [5] 

biases = ['none', 
         'distI_1_0', 'distI_2_0', 'distI_0_1', 'distI_0_2',
         'distI_1_1', 'distI_2_1', 'distI_1_2', 'distI_2_2',
         'opt_c', 'opt_c_I1', 'opt_c_I2', 'opt_c_s2', 'opt_c_s3',
         'hs_n1_w05', 'hs_n1_w10', 'hs_n1_w15', 'hs_n1_w20',
         'hs_n2_w05', 'hs_n2_w10', 'hs_n2_w15', 'hs_n2_w20']

#beta_params = pickle.load(open(os.path.join(SRC_DIR, 'beta_param_biases.pickle'), 'rb'))


# Parallel parameters
N_PROC = 28
CHUNKSIZE = 5

# Initialisation parameters
N_SCALES = 10000


def calculate_most_harmonic_neighbour(int_cents, sim_only=False, CENT_DIFF_MAX=22):
    best_ratio = [1,1]
    max_similarity = 0.0
    cents = 0.0
    for x in np.arange(1,75, dtype=float):
        cent_diff = 1200.*np.log10((x+1.)/x)/np.log10(2.) - int_cents
        if cent_diff > CENT_DIFF_MAX:
            continue
        for y in np.arange(x+1.,99., dtype=float):
            cent_diff = abs(1200.*np.log10(y/x)/np.log10(2.)- int_cents)
            if cent_diff > CENT_DIFF_MAX:
                continue
            simil = ((x+y-1.)/(x*y))*100.
            if simil > max_similarity:
                cents = 1200.*np.log10(y/x)/np.log10(2.)
                best_ratio = [y,x]
                max_similarity = simil
    if sim_only:
        return max_similarity
    else:
        return max_similarity, best_ratio, cents

def get_attractors(n, dI=5., diff=22):
    sc_i = np.arange(dI, 1200.+dI, dI)
    sc_f = set()
    attract = []
    ratios = []
    simils = []
    for s in sc_i:
        max_similarity, best_ratio, cents = calculate_most_harmonic_neighbour(s, CENT_DIFF_MAX=diff)
        if max_similarity == 0.0:
            continue
        if round(cents,2) not in sc_f:
            sc_f.add(round(cents,2))
            attract.append(round(cents,2))
            ratios.append(best_ratio)
            simils.append(max_similarity ** n / 100.**(n-1))
    return sc_i, np.array(attract), ratios, simils

if os.path.exists('hs_attractors.pickle'):
    ATTRACTORS = pickle.load(open('hs_attractors.pickle', 'rb'))
else:
    ATTRACTORS = {f"hs_n{n}_w{w:02d}":get_attractors(n, diff=w) for n in [1,2,3] for w in [5,10,15,20]}
    pickle.dump(ATTRACTORS, open('hs_attractors.pickle', 'wb'))

FILES = sorted(glob.glob("/home/jmcbride/Scales/Toy_model/Src/Alternative_attractors/*"))
ALT_ATT = {os.path.split(f)[1][3:9]: np.load(f) for f in FILES}
IM5_ATT = pickle.load(open('im5_attractors.pickle', 'rb'))

def generate_new_scale(inp):
    n_notes, INT_MIN, INT_MAX, BETA, bias = inp
    np.random.seed()
    count = 0 
    switch = True
    while switch:
        intervals = np.random.rand(n_notes) * (INT_MAX - INT_MIN) + INT_MIN
        intervals *= 1200. / np.sum(intervals)
        switch = sum([1 for x in intervals if not INT_MIN < x < INT_MAX])
        if BETA != 0.0:
            if not switch:
                energy = calculate_energy_by_bias(intervals, bias)
                if np.random.rand() > np.exp(-BETA * energy):
                    switch = 1
        count += 1
    return (intervals, count)

def calculate_energy_by_bias(ints, bias):
    bias = bias.strip('N')
    if 'distI' in bias:
        m, n = [float(x) for x in bias.split('_')[1:]]
        return template_function(ints, m, n)
    if 'TRANSB' in bias:
        n = int(bias.split('_')[1])
        return template_function(ints, n, 0, new=True)
    if bias == 'opt_c':
        return calculate_optimum_window_size(ints)
    if bias == 'opt_c_I1':
        return calculate_optimum_window_size(ints) * calculate_energy_from_intervals(ints, min(ints), 0, 1)
    if bias == 'opt_c_I2':
        return calculate_optimum_window_size(ints) * calculate_energy_from_intervals(ints, min(ints), 0, 2)
    if bias == 'opt_c_s2':
        return calculate_optimum_window_size(ints) * np.sum(ints**2) / 1200.**2
    if bias == 'opt_c_s3':
        return calculate_optimum_window_size(ints) * np.sum(ints**3) / 1200.**3
    elif 'hs' in bias:
        all_ints = [x for i in range(len(ints)) for x in np.cumsum(np.roll(ints,i))[:-1]]
        return 1. / np.mean([get_similarity_of_nearest_attractor(x, ATTRACTORS[bias][1], ATTRACTORS[bias][3]) for x in all_ints])
    elif 'im5' in bias:
        all_ints = [x for i in range(len(ints)) for x in np.cumsum(np.roll(ints,i))[:-1]]
        return 1. / (ALPHA_HS + np.mean([get_similarity_of_nearest_attractor(x, IM5_ATT[bias][0], IM5_ATT[bias][1]) for x in all_ints]))
    

def calculate_optimum_window_size(ints):
    w_arr = np.arange(15,61)
    d_arr = []
    for w in w_arr:
        d_arr.append(calculate_distance_between_windows(sorted(ints), w))
    d_arr = np.array(d_arr)
    cost = np.zeros(len(w_arr), dtype=float)
    idx = np.where(d_arr)[0]
    cost[idx] = w_arr[idx] / d_arr[idx]
    return  np.min(cost) + ALPHA_W

def calculate_distance_between_windows(ints, w):
    windows = [[ints[0]]]
    for i in ints[1:]:
        if i - windows[-1][0] < w:
            windows[-1].append(i)
        else:
            windows.append([i])
    if len(windows) == 1:
        return 0
    else:
        dist = [windows[i+1][0] - windows[i][-1] for i in range(len(windows)-1)]
        return min(dist)

def calculate_energy_from_intervals(ints, base, m, n): 
    return np.mean([abs(round(i/base) - i/base)**m * float(round(i/base))**n for i in ints])


def calculate_energy_from_intervals_2(ints, base, m, n): 
    return np.mean([(abs(round(i/base) - i/base) * base)**m * float(round(i/base))**n for i in ints]) / (20.**m)


def template_function(ints, m, n, new=False):
    temp_min = max(TEMP_MIN, min(ints)*TEMP_LOW_MARGIN)
    temp_max = min(TEMP_MAX, min(ints)*TEMP_HI_MARGIN)
    baseArr = np.linspace(temp_min, temp_max, num=N_TRIALS)
    energies = np.zeros(baseArr.size, dtype=float)
    for i, base in enumerate(baseArr):
        if new:
            energies[i] = calculate_energy_from_intervals_2(ints, base, m, n)
        else:
            energies[i] = calculate_energy_from_intervals(ints, base, m, n)
    if len(np.where(energies==0)[0]) > 1:
        idxMin = np.where(energies==0)[0][-1]
    else:
        idxMin = np.argmin(energies)
    return energies[idxMin]

def get_similarity_of_nearest_attractor(x, sc_f, simil):
    minIdx = np.argmin(np.abs(sc_f - x))
    return simil[minIdx]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-t', action='store', default=0, type=int)
    args = parser.parse_args()

#   inputs =  np.array(list(product(Narr, MIarr, MAarr, biases)))
#   np.save('input_parameters', inputs)

#   inputs = np.load('input_parameters.npy')
    inputs = np.load('new_inputs_4.npy')

    n, INT_MIN, INT_MAX, bias = inputs[args.t-1]

#   n, INT_MIN, INT_MAX, bias, beta = inputs[args.t-1]
#   beta = float(beta)
#   BETAarr = [beta]

    n = int(n)
    INT_MIN = float(INT_MIN)
    INT_MAX = float(INT_MAX)

    print(f'INPUT T#{args.t}')

    count = 0

    BETAarr = BETA_BIAS[bias] / float(n)

    for BETA in BETAarr:
        count += 1
        print(n, INT_MIN, INT_MAX, bias)
        fName = f"{DATA_DIR}/n{n}_{bias}_MI{int(INT_MIN):d}_MA{int(INT_MAX):d}_BETA_{BETA:07.3f}.feather"
        fName_2 = f"{DATA_DIR_2}/n{n}_{bias}_MI{int(INT_MIN):d}_MA{int(INT_MAX):d}_BETA_{BETA:07.3f}.feather"
        if os.path.exists(fName) or  os.path.exists(fName_2):
            continue
        print(f"\nRound {count}")
        print(f"\tINT_MIN = {INT_MIN}\tINT_MAX = {INT_MAX}\t{bias}\tBETA = {BETA}")
        timeS = time.time()

        pool = mp.Pool(N_PROC)
        output = list(pool.imap_unordered(generate_new_scale, [(n, INT_MIN, INT_MAX, BETA, bias)]*N_SCALES, CHUNKSIZE))
#       output = [ generate_new_scale( (n, INT_MIN, INT_MAX, BETA, bias) ) for i in range(N_SCALES) ]
        scales = [o[0] for o in output]
        n_samp = [o[1] for o in output]
        print(f"{len(scales)} scales accepted out of {sum(n_samp)} scales generated")
        print('Time taken:  ', (time.time()-timeS)/60.)

        str_ints = [';'.join([str(int(round(x))) for x in sc]) for sc in scales]
        df = pd.DataFrame(data={'pair_ints':str_ints, 'n_att':n_samp})
        df['n_notes'] = n
        df = utils.get_scale_from_pair_ints(df)
        df = utils.get_all_ints_from_pair_ints(df)
#       df = utils.get_harmonic_similarity_score_df(df)
#       df = utils.get_attractors_in_scale(df)

        df.to_feather(fName)
        print('Time taken:  ', (time.time()-timeS)/60.)

#       if sum(n_samp) > 50000000:
#           break
        if ((time.time()-timeS)/60.) > 100:
            break


