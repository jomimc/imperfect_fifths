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
DATA_DIR = '/home/jmcbride/Scales/Toy_model/Data/Raw_tmp3'
DATA_DIR_2 = '/home/jmcbride/Scales/Toy_model/Data/Raw3'

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



# Parallel parameters
N_PROC = 28
CHUNKSIZE = 5

# Initialisation parameters
N_SCALES = 10000


def load_norm_const():
    df_list = []
    for s in ['HAR', 'TRANSA', 'TRANSB']:
        df_list.append(pd.read_csv(f"/home/jmcbride/Scales/Toy_model/Params/{s}.txt", header=0))
        df_list[-1]['bias'] = s
    return pd.concat(df_list, ignore_index=True)


NORM_CONST = load_norm_const()


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
    N = len(ints)
    b = bias.split('_')[0]
    try:
        w = int(bias.split('_')[1])
    except IndexError:
        pass
    if 'TRANSA' in bias:
        return 1.0  - (template_function(ints, w, 0) - min_score) / (max_score - min_score)
    elif 'TRANSB' in bias:
        return 1.0  - (template_function(ints, w, 0, new=True) - min_score) / (max_score - min_score)
    if 'HAR' in bias:
        m = int(bias.split('_')[2])
        min_score, max_score = NORM_CONST.loc[(NORM_CONST.bias==b)&(NORM_CONST.N==N)&(NORM_CONST.W==w)&(NORM_CONST.M==float(m)),  ['Min', 'Max']].values[0]
        all_ints = [x for i in range(len(ints)) for x in np.cumsum(np.roll(ints,i))[:-1]]
        b2 = f"hs_n{bias.split('_')[2]}_w{w:02d}"
#       try:
#       except:
#           b2 = f'hs_n1_w{w:02d}'
        score = np.mean([get_similarity_of_nearest_attractor(x, ATTRACTORS[b2][1], ATTRACTORS[b2][3]) for x in all_ints])
        return 1.0 - ((score - min_score) / (max_score - min_score))**2
    elif 'FIF' in bias:
        all_ints = [x for i in range(len(ints)) for x in np.cumsum(np.roll(ints,i))[:-1]]
        score = np.mean([1 if abs(int(x)-702) <= w else 0  for x in all_ints])
        return 1. - (score * float(N))**2
    elif bias == 'none':
        return 0
    

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


def calculate_energy_from_intervals_2(ints, base, n): 
    return np.mean([(abs(round(i/base) - i/base) * base)**n for i in ints])


def calculate_energy_from_intervals(ints, base, n): 
    return np.mean([abs(round(i/base) - i/base)**n for i in ints])


def template_function(ints, m, n, new=False):
    temp_min = max(TEMP_MIN, min(ints)*TEMP_LOW_MARGIN)
    temp_max = min(TEMP_MAX, min(ints)*TEMP_HI_MARGIN)
    baseArr = np.linspace(temp_min, temp_max, num=N_TRIALS)
    energies = np.zeros(baseArr.size, dtype=float)
    for i, base in enumerate(baseArr):
        if new:
            energies[i] = calculate_energy_from_intervals_2(ints, base, m)
        else:
            energies[i] = calculate_energy_from_intervals(ints, base, m)
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

    inputs = np.load('new_inputs_6.npy')

    n, INT_MIN, INT_MAX, bias = inputs[args.t-1]
#   BETAarr = BETA_BIAS[bias]
#   BETAarr = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50, 55, 60]
#   BETAarr = [0.1, 1, 3, 5, 7, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50, 55, 60]
#   BETAarr = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 100, 1000]
    BETAarr = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50]

#   n, INT_MIN, INT_MAX, bias, beta = inputs[args.t-1]
#   beta = float(beta)
#   BETAarr = [beta]

    n = int(n)
    INT_MIN = float(INT_MIN)
    INT_MAX = float(INT_MAX)


    print(f'INPUT T#{args.t}')

    count = 0

    for BETA in BETAarr:
#   INT_MIN, INT_MAX, bias, BETA = 80, 1200, 'none', 1
#   for n in range(4,10):
        count += 1
        print(n, INT_MIN, INT_MAX, bias)
        fName = f"{DATA_DIR}/n{n}_{bias}_MI{int(INT_MIN):d}_MA{int(INT_MAX):d}_BETA_{BETA:07.3f}.feather"
        fName_2 = f"{DATA_DIR_2}/n{n}_{bias}_MI{int(INT_MIN):d}_MA{int(INT_MAX):d}_BETA_{BETA:07.3f}.feather"
        if os.path.exists(fName):  # or  os.path.exists(fName_2):
            continue
        print(f"\nRound {count}")
        print(f"\tINT_MIN = {INT_MIN}\tINT_MAX = {INT_MAX}\t{bias}\tBETA = {BETA}")
        timeS = time.time()

        with mp.Pool(N_PROC) as pool:
            output = list(pool.imap_unordered(generate_new_scale, [(n, INT_MIN, INT_MAX, BETA, bias)]*N_SCALES, CHUNKSIZE))
#       output = [ generate_new_scale( (n, INT_MIN, INT_MAX, BETA, bias) ) for i in range(N_SCALES) ]
        scales = [o[0] for o in output]
        n_samp = [o[1] for o in output]
        print(f"{len(scales)} scales accepted out of {sum(n_samp)} scales generated")
        print(f"Acceptance rate = {len(scales)/sum(n_samp)}")
        print('Time taken:  ', (time.time()-timeS)/60.)

        str_ints = [';'.join([str(int(round(x))) for x in sc]) for sc in scales]
        df = pd.DataFrame(data={'pair_ints':str_ints, 'n_att':n_samp})
        df['n_notes'] = n
        df = utils.get_scale_from_pair_ints(df)
        df = utils.get_all_ints_from_pair_ints(df)
        df = utils.get_harmonic_similarity_score_df(df)
#       df = utils.get_attractors_in_scale(df)

        df.to_feather(fName)
        print('Time taken:  ', (time.time()-timeS)/60.)

#       if sum(n_samp) > 5000000:
#           break
        if ((time.time()-timeS)/60.) > 100:
            break

