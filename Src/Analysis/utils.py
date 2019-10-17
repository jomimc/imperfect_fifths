import re
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
from itertools import permutations, product
from matching import Player, StableMarriage, HospitalResident
import multiprocessing as mp
import numpy as np
from palettable.colorbrewer.qualitative import Paired_12, Set1_9, Dark2_8
from palettable.colorbrewer.diverging import  RdYlGn_6, RdYlGn_11
import pandas as pd
from pyarrow.lib import ArrowIOError
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.spatial.distance import pdist
from scipy.stats import linregress, pearsonr
from sklearn.cluster import DBSCAN
import statsmodels.nonparametric.api as smnp


N_PROC = 28

BASE_DIR = "/home/johnmcbride/projects/Scales/Data_compare"
DIST_DIR = "/home/johnmcbride/projects/Scales/Toy_model/Data/None_dist"
SRC_DIR = "/home/johnmcbride/projects/Scales/Data_compare/Src"
CLEAN_DIR = "/home/johnmcbride/projects/Scales/Data_compare/Processed/Cleaned"


INST = np.array([1,0,1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,0,1,0,1,1,1,1], dtype=bool)
CENT_DIFF_MAX =  20.0

BETA = 50.

PYT_INTS = np.array([0., 90.2, 203.9, 294.1, 407.8, 498.1, 611.7, 702., 792.2, 905., 996.1, 1109.8, 1200.])
EQ5_INTS = np.linspace(0, 1200, num=6, endpoint=True, dtype=float)
EQ7_INTS = np.linspace(0, 1200, num=8, endpoint=True, dtype=float)
EQ9_INTS = np.linspace(0, 1200, num=10, endpoint=True, dtype=float)
EQ10_INTS = np.linspace(0, 1200, num=11, endpoint=True, dtype=float)
EQ12_INTS = np.linspace(0, 1200, num=13, endpoint=True, dtype=float)
EQ24_INTS = np.linspace(0, 1200, num=25, endpoint=True, dtype=float)
JI_INTS = np.array([0., 111.7, 203.9, 315.6, 386.3, 498.1, 590.2, 702., 813.7, 884.4, 1017.6, 1088.3, 1200.])
SLENDRO = np.array([263., 223., 253., 236., 225.])
PELOG   = np.array([167., 245., 125., 146., 252., 165., 100.])
DASTGAH = np.array([0., 90., 133.23, 204., 294.14, 337.14, 407.82, 498., 568.72, 631.28, 702., 792.18, 835.2, 906., 996., 1039.1, 1109.77, 1200.])
TURKISH = {'T':203.8, 'K':181.1, 'S':113.2, 'B':90.6, 'F':22.6, 'A':271, 'E':67.9}
KHMER_1 = np.array([185., 195., 105., 195., 195., 185., 140.])
KHMER_2 = np.array([190., 190., 130., 190., 190., 190., 120.])
VIET    = np.array([0., 175., 200., 300., 338., 375., 500., 520., 700., 869., 900., 1000., 1020., 1200.])
CHINA   = np.array([ 113.67291609,  203.91000173,  317.73848174,  407.83554758, 520.68758457,  611.71791523,  701.95500087,  815.62791696, 905.8650026 , 1019.47514332, 1109.76982292, 1201.27828039])

OCT_CUT = 50

STATS_PATH = "/home/johnmcbride/projects/Scales/Data_compare/Stats"


NORM_CONST = pd.read_feather("/home/johnmcbride/projects/Scales/Data_compare/Src/norm_const.feather")

def load_data_for_figs():
    # DataFrame for the first set of results
    # (contains MIN, RAN, TRANS and FIF main results)
    f = os.path.join(CLEAN_DIR, "monte_carlo_results_1.feather")
    if os.path.exists(f):
        df = pd.read_feather(f)
    else:
        df = reformat_old_df(normalise_metrics(pd.read_feather(os.path.join(BASE_DIR, 'Processed', 'monte_carlo_comparison.feather'))))
        df.to_feather(f)

    # DataFrame for the second set of results
    # (contains HAR, HAR2, and HAR3 main results)
    f = os.path.join(CLEAN_DIR, "monte_carlo_results_2.feather")
    if os.path.exists(f):
        df2 = pd.read_feather(f)
    else:
        df2 = normalise_metrics(pd.read_feather(os.path.join(BASE_DIR, 'Processed', 'monte_carlo_comparison3.feather')))
        df2.to_feather(f)

    # DataFrame for the scales database
    df_real = pd.read_feather(os.path.join(BASE_DIR, 'Processed', 'Real', 'theories_real_scales.feather'))

    # Best performing models (optimized beta)
    # for each set of parameters
    df_b1, paths1 = pick_best_models(df, df_real)
    df_b2, paths2 = pick_best_models(df2, df_real)
    paths = amalgamate_paths(paths1, paths2)

    # Update DataFrame for the scales
    f = os.path.join(CLEAN_DIR, "scales_database.feather")
    if os.path.exists(f):
        df_real = pd.read_feather(f)
    else:
        df_real = calculate_fifths_bias_all_w(update_hss_scores_df(df_real, n=1))
        df_real = probability_of_finding_scales(paths, df_real) 
        df_real.to_feather(f)

    # Theoretical populations for the model scales
    model_df = {}
    for l in ['RAN', 'MIN', 'TRANS', 'FIF', 'HAR', 'HAR2', 'HAR3']:
        f = os.path.join(CLEAN_DIR, "Cleaned", "scales_database.feather")
        try:
            model_df.update({l:{n: pd.read_feather(os.path.join(CLEAN_DIR, "Cleaned", "Models", f"{l}_{n:02d}.feather")) for n in range(4,10)}})
        except ArrowIOError:
            model_df.update({l:{n: update_hss_scores_df(calculate_fifths_bias_all_w(get_all_ints(pd.read_feather(paths[l][n])))) for n in range(4,10)}})
            for n in range(4,10):
                model_df[l][n].to_feather(os.path.join(CLEAN_DIR, "Models", f"{l}_{n:02d}.feather"))

    return df, df2, df_real, df_b1, df_b2, paths, model_df
    

def get_beta_values(df):
    out_dict = {}
    for bias in df.bias.unique():
        out_dict.update({bias: [np.mean(df.loc[(df['quantile']==q)&(df.bias==bias),'beta']) for q in sorted(df['quantile'].unique())]})
    return out_dict

def reformat_comparison_df(df):
    df2 = df.loc[df.cat=='pair_ints'].reset_index(drop=True).rename(columns={'JSD':'pJSD'})
    df2['sJSD'] = df.loc[df.cat=='scale','JSD'].reset_index(drop=True)
    df2['aveJSD'] = (df2['pJSD'] + df2['sJSD']) / 2.0
    return df2

def filter_df(df, filters):
    idx = df.index
    for k, v in filters.items():
        if v[0]=='<':
            idx = df.loc[idx,:].loc[df[k]<v[1]].index
        elif v[0]=='>':
            idx = df.loc[idx,:].loc[df[k]>v[1]].index
        elif v[0]=='==':
            idx = df.loc[idx,:].loc[df[k]==v[1]].index
    return df.loc[idx]

def extract_ints_from_string(df):
    return [int(x) for y in df for x in y.split(';') if len(y)]

def extract_floats_from_string(df):
    try:
        return [float(x) for y in df for x in y.split(';')]
    except:
        return [float(x) for y in df for x in y.decode().split(';')]

def calc_relative_entropy(pk, qk):
    RE = 0.0
    for i in range(len(pk)):
        if pk[i] <= 0 or qk[i] <= 0:
#           print(f"{i} out of {len(pk)} is zero\npk = {pk[i]}\tqk = {qk[i]}")
#           print("{} out f {} is zero\npk = {}\tqk = {}".format(i, len(pk), pk[i], qk[i]))
            pass
        else:
            RE += pk[i] * np.log2(pk[i] / qk[i])
    return RE

def convert_grid(grid, y, num=1201):
    new_grid = np.linspace(0, 1200, num=num)
    new_y = np.zeros(num, dtype=float)
    if grid[0] < 0:
        start_point = 0
    else:
        start_point = np.where(new_grid - grid[0] > 0)[0][0]
    if grid[-1] > 1200:
        end_point = num
    else:
        end_point = np.where(new_grid - grid[-1] > 0)[0][0]

    for i in range(start_point, end_point):
        idx = np.where(grid - new_grid[i] > 0)[0][0]
        new_y[i] = y[idx-1] + (new_grid[i] - grid[idx-1]) * (y[idx] - y[idx-1]) / (grid[idx] - grid[idx-1])

    return new_grid, new_y

def smooth_dist_kde(df, cat='pair_ints', hist=False):
    X = [float(x) for y in df.loc[:,cat] for x in y.split(';')]
    kde = smnp.KDEUnivariate(np.array(X))
    kde.fit(kernel='gau', bw='scott', fft=1, gridsize=10000, cut=20)
    grid = np.linspace(0, 1200, num=1201)
    y = np.array([kde.evaluate(x) for x in grid]).reshape(1201)
    if hist:    
        hist, edges = np.histogram(X, bins=grid)
        xxx = grid[:-1] + (grid[1] - grid[0]) * 0.5    
        return grid, y, xxx, hist
    else:
        return grid, y

def exp_fit(X):
    grid = np.linspace(0, 1, num=101)
    hist, grid = np.histogram(X, bins=grid)
#   fit_fn = lambda x, a, b, c: b / ((x+c) ** a)
    fit_fn = lambda x, a, b, c: a * (1.0-x)**b / x**c
    popt, pcov = curve_fit(fit_fn, grid[1:], list(hist), p0=[0.2, 1, 1])
    Y = fit_fn(grid[1:], *popt)
    return grid[1:], Y


def separate_clusters(pos, w, n_clu):
    idx_sort = np.argsort(pos)
    cum_diff = [p - pos.min() for p in pos[idx_sort]]
    new_clu  = []
    offset = 0
    for cd in cum_diff:
        if (cd - offset) > w:
            n_clu += 1
            offset += cd
        new_clu.append(n_clu)
    pos[idx_sort] = new_clu
    return pos 

def get_clusters(df, w=20, cat='pair_ints'):
    clusters = []
    for pi in [ [int(x) for x in y.split(';')] for y in df.loc[:,cat]]:
        pi_inp = np.array([pi, [0]*len(pi)]).T
        cluster = DBSCAN(eps=w, min_samples=1).fit(pi_inp)
        clu_idx = cluster.labels_
        clu_set = set(clu_idx)
        for clu_id in sorted(list(clu_set)):
            clu_pos = pi_inp[clu_idx==clu_id,0]
            clu_range = clu_pos.max() - clu_pos.min()
            if clu_range > w:
                new_clu = separate_clusters(clu_pos, w, len(clu_set))
                clu_idx[clu_idx==clu_id] = new_clu
                [clu_set.add(x) for x in new_clu]
        clusters.append(len(set(clu_idx)))
    return clusters


def get_ratio_from_cents(cents):
    return 2 ** (cents / 1200.)

def get_cents_from_ratio(ratio):
    return 1200.*np.log10(ratio)/np.log10(2)

def sum_to_n(n, size, limit=None, nMin=1):
    """Produce all lists of `size` positive integers in decreasing order
    that add up to `n`."""
    if size == 1:
        yield [n]
        return
    if limit is None:
        limit = n
    start = (n + size - 1) // size
    stop = min(limit, n - size + 1) + 1
    for i in range(start, stop):
        for tail in sum_to_n(n - i, max(size - 1, nMin), i, nMin=nMin):
            yield [i] + tail

def get_all_possible_scales_12_tet(df):
    codes = set(df.code.unique())
    for i in range(4,13):
        for partition in sum_to_n(12, i, limit=4):
            ints = [len(np.where(np.array(partition)==j)[0]) for j in range(1,5)]
            code =  ''.join([str(x) for x in ints])
            if code not in codes:
                codes.add(code)
                df.loc[len(df), ['1','2','3','4','notes_in_scale','code']] = ints + [i] + [code]
    return df
        
def get_all_possible_scales_general(nI=240, iLimit=80, nSmin=4, nSmax=9, nMin=1):
    df = pd.DataFrame(columns=['n_notes', 'interval'])
    last_len = 0
    for i in range(nSmin, nSmax+1):
        timeS = time.time()
        print(i)
        for partition in sum_to_n(nI, i, limit=iLimit, nMin=nMin):
            ints = [float(x)*(1200./float(nI)) for x in partition]
            code =  ';'.join([str(x) for x in ints])
            df.loc[len(df), ['n_notes','interval']] = [i] + [code]
        print(len(df) - last_len, ' scales found after ...')
        last_len = len(df)
        print((time.time()-timeS)/60., ' minutes')
    return df

def check_for_allowed_ratios(df):
    def fn(x):
        ints = [float(y) for y in x.split(';')]
        ratios = [i / min(ints) for i in ints]
        if sum([1 for r in ratios if r not in [1., 1.5, 2., 2.5, 3., 3.5, 4.]]):
#       if sum([1 for r in ratios if r not in [1., 2., 3., 4.]]):
            return False
        else:
            return True
    df['allowed_ratios'] = df.interval.apply(lambda x: fn(x))
    return df

def are_intervals_distinct(df, cent_tol=30.):
    def fn(x):
        ints = [float(y) for y in x.split(';')]
        diffs = np.abs([ints[i] - ints[j] for i in range(len(ints)) for j in range(len(ints))])
        return sum([1 for d in diffs if 0.0 < d < cent_tol])
    df['distinct'] = df.interval.apply(lambda x: fn(x))
    return df

def get_scale_energies_exact_ratios(df):
    def fn(x):
        ints = [float(y) for y in x.split(';')]
        ratios = [i / min(ints) for i in ints]
        return np.mean([1./(float(i)-0.5)**3 + np.exp(float(i)) for i in ratios])
    df['energy'] = df.interval.apply(lambda x: fn(x))
    return df

def get_scale_energies_real_ratios(df):
    def fn(x):
        ratios = [float(y) for y in x.split(';')]
#       ratio_term  = np.mean([1./(y-0.5)**3 + np.exp(y) for y in ratios])
        ratio_term  = np.mean([1./(y/40.) + np.exp(y)*2.0 for y in ratios])
        microtuning = BETA * np.mean([(round(y) - y)**2 * float(round(y)) for y in ratios])
        return ratio_term + microtuning
    df['energy'] = df.ratios.apply(lambda x: fn(x))
    return df

def get_scale_from_pair_ints(df):
    def fn(x):
        ints = [int(y) for y in x.split(';')]
        return ';'.join(['0'] + [str(y) for y in np.cumsum(ints)])
    df['scale'] = df.pair_ints.swifter.apply(lambda x: fn(x))
    return df

def get_ratios_from_ints(df):
    def fn(x):
        ratios = [float(y) for y in x.split(';')]
        base_ints = np.arange(25., min(ratios)*1.2, 5.)
        min_energy = 10.e10
        max_base   = 1.
        for i, base in enumerate(base_ints):
            energy = calculate_energy_harmonic(ratios, base)
            if energy <= min_energy:
                min_energy = energy
                max_base = base
        return ';'.join([str(round(y / max_base, 2)) for y in ratios ])
    df['ratios'] = df.interval.apply(lambda x: fn(x))
    return df

def get_min_energy_integer_ratios(scale):
    ratios = []
    base_o = []
    base_ints = np.arange(35., 155., 5.)
    for pi in pair_ints:
        energy = np.zeros(base_ints.size, dtype=float)
        for i, base in enumerate(base_ints):
            energy[i] = calculate_energy(pi, base)
        ratios.append([x / base_ints[np.argmin(energy)] for x in pi ])
        base_o.append(base_ints[np.argmin(energy)]) 
        print(base_ints[np.argmin(energy)], pi)
        print(energy)
    return ratios, base_o

def plot_real_vs_derived_ints(df, pair_ints, iMin=100., n=7, weights=[]):
    fig, ax1 = plt.subplots(2,1)
    idx = [i for i in range(len(pair_ints)) if len(pair_ints[i])==n]

    b = [[float(y) for y in x.split(';')] for x in df.loc[(df.min_int.apply(lambda x: x>=iMin))&(df.n_notes==n)&(df.allowed_ratios), 'interval']]
    sns.distplot([float(y) for x in df.loc[(df.allowed_ratios)&(df.n_notes==n)&(df.min_int.apply(lambda x: x>=iMin)), 'interval'] for y in x.split(';')], bins=100, ax=ax1[0], label='derived')
    sns.distplot([c/min(a) for a in b for c in a], bins=100, ax=ax1[1], label='derived')

    if len(weights):
        w1 = [weights[i] for i in idx for y in range(len(pair_ints[i]))]
        sns.distplot([y for x in np.array(pair_ints)[idx] for y in x], bins=100, ax=ax1[0], label='real', hist_kws={'weights':w1})
        sns.distplot([y/min(x) for x in np.array(pair_ints)[idx] for y in x], bins=100, ax=ax1[1], label='real', hist_kws={'weights':w1})
    else:
        sns.distplot([y for x in np.array(pair_ints)[idx] for y in x], bins=100, ax=ax1[0], label='real')
        sns.distplot([y/min(x) for x in np.array(pair_ints)[idx] for y in x], bins=100, ax=ax1[1], label='real')
    ax1[0].legend(loc='best')
    ax1[1].legend(loc='best')
    plt.show()

def plot_real_vs_derived_ints_energy_cutoff(df, pair_ints, eCut=1000., iMin=100., n=7, weights=[]):
    fig, ax1 = plt.subplots(2,1)
    idx = [i for i in range(len(pair_ints)) if len(pair_ints[i])==n]
    idx2 = df.loc[(df.energy<eCut)&(df.min_int>=iMin)&(df.n_notes==n)].index 

#   idx2 = df.loc[(df.energy<eCut)&(df.min_int>=iMin)].index 
#   idx = range(len(pair_ints))

    b = [[float(y) for y in x.split(';')] for x in df.loc[idx2, 'interval']]
    sns.distplot([float(y) for x in df.loc[idx2, 'interval'] for y in x.split(';')], bins=100, ax=ax1[0], label='derived')
    sns.distplot([c/min(a) for a in b for c in a], bins=100, ax=ax1[1], label='derived')

    if len(weights):
        w1 = [weights[i] for i in idx for y in range(len(pair_ints[i]))]
        sns.distplot([y for x in np.array(pair_ints)[idx] for y in x], bins=100, ax=ax1[0], label='real', hist_kws={'weights':w1})
        sns.distplot([y/min(x) for x in np.array(pair_ints)[idx] for y in x], bins=100, ax=ax1[1], label='real', hist_kws={'weights':w1})
    else:
        sns.distplot([y for x in np.array(pair_ints)[idx] for y in x], bins=100, ax=ax1[0], label='real')
        sns.distplot([y/min(x) for x in np.array(pair_ints)[idx] for y in x], bins=100, ax=ax1[1], label='real')
    ax1[0].legend(loc='best')
    ax1[1].legend(loc='best')
    plt.show()

def plot_all_scales_ints_ratios(df):
    n_notes = df.n_notes.unique()
    fig, ax = plt.subplots(6,2)
    for i, n in enumerate(n_notes):
        ints = [[float(x) for x in y.split(';')] for y in df.loc[df.n_notes==n,'interval']]
        sns.distplot([x for y in ints for x in y], bins=100, ax=ax[i,0], label=str(n))
        sns.distplot([y / min(x) for x in ints for y in x], bins=100, ax=ax[i,1], label=str(n))
        ax[i,0].legend(loc='best')
        ax[i,1].legend(loc='best')
    plt.show()
        
def get_attractors(n, dI=5., diff=11.):
    sc_i = np.arange(0, 1200.+dI, dI)
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
            simils.append(max_similarity**n / 100**(n-1))
    return sc_i, np.array(attract), ratios, simils

def get_harmonic_similarity_score(pair_ints):
    output = []
    for x in pair_ints:
        scores = []
        for i in [y for i in range(len(x)) for y in np.cumsum(x[i:])]:
            if i == 0:
                continue
            sc, _, _ = calculate_most_harmonic_neighbour(i)
            scores.append(sc) 
        output.append( np.mean(scores) )
    return output

def get_similarity_of_nearest_attractor(x, sc_f, simil):
    minIdx = np.argmin(np.abs(sc_f - x)) 
    return simil[minIdx]

def get_nearest_attractor(x, sc_f, ratio):
    minIdx = np.argmin(np.abs(sc_f - x)) 
    return ':'.join([str(int(r)) for r in ratio[minIdx]])

def get_harmonic_similarity_score_df(df, n=1, diff=11, r=0):
    sc_i, sc_f, ratios, simil  = get_attractors(n, diff=diff)
    if r:
        cut = simil[np.argsort(simil)[-r-1]]
        simil = np.array(simil)
        simil[np.where(simil<cut)] = 0
        simil[simil==100] = 0
#   return df.all_ints.apply(lambda x: np.mean([get_similarity_of_nearest_attractor(float(y), sc_f, simil) for y in x.split(';')]))
    return df.pair_ints.apply(lambda x: np.mean([get_similarity_of_nearest_attractor(y, sc_f, simil) for i in range(len(x.split(';'))) for y in np.cumsum(np.roll([float(z) for z in x.split(';')],i))[:-1]]))

def update_hss_scores_df(df, n=1):
    simil = []
    sc = []
    for w in [5, 10, 15, 20]:
        sc_i, sc_f, r, s  = pickle.load(open(os.path.join(SRC_DIR, "hs_attractors.pickle"), 'rb'))[f"hs_n{n}_w{w:02d}"]
        simil.append(s)
        sc.append(sc_f)
    for j, w in enumerate([5, 10, 15, 20]):
        df[f"hs_n{n}_w{w}"] = df.all_ints2.apply(lambda x: np.mean([get_similarity_of_nearest_attractor(int(y), sc[j], simil[j]) for y in x.split(';')]))
    return df
        

def update_hss_scores_for_set_of_df(df_list, n=1):
    simil = []
    sc = []
    for w in [5, 10, 15, 20]:
#       sc_i, sc_f, r, s  = get_attractors(n, diff=w)
        sc_i, sc_f, r, s  = pickle.load(open(os.path.join(SRC_DIR, "hs_attractors.pickle"), 'rb'))[f"hs_n{n}_w{w:02d}"]
        simil.append(s)
        sc.append(sc_f)
    if isinstance(df_list, list):
        idx = range(len(df_list))
    elif isinstance(df_list, dict):
        idx = df_list.keys()
    for i in idx:
        df_list[i] = get_all_ints(df_list[i])
        for j, w in enumerate([5, 10, 15, 20]):
            df_list[i][f"hs_n{n}_w{w}"] = df_list[i].all_ints2.apply(lambda x: np.mean([get_similarity_of_nearest_attractor(int(y), sc[j], simil[j]) for y in x.split(';')]))
    return df_list
        

def get_attractors_in_scale(df):
    sc_i, sc_f, ratios, simil  = get_attractors()
    df['attractors'] =  df.all_ints.apply(lambda x: ';'.join([str(get_nearest_attractor(float(y), sc_f, ratios)) for y in x.split(';')]))
    return df

def get_weighted_harmonic_similarity_score(pair_ints):
    output = []
    for x in pair_ints:
        scores = []
        for i in range(len(x)):
            for j, y in enumerate(np.cumsum(x[i:])):
                if y == 0:
                    continue
                sc, _, _ = calculate_most_harmonic_neighbour(y)
                scores.append(sc) 
                if i == 0:
                    scores.append(sc) 
        output.append( np.mean(scores) )
    return output

def get_harmonic_similarity_score_extra_notes(df):
    def fn(x):
        a = [int(y) for y in x*2]
        return np.mean(INTERVAL_SCORE[[l-1 for i in range(len(x)) for l in list(np.cumsum(a[i:i+len(x)]))]])
    df['score_en'] = df.str_fmt.apply(lambda x: fn(x))
    return df

def calculate_most_harmonic_neighbour(int_cents, sim_only=False, CENT_DIFF_MAX=11.):
    best_ratio = [1,1]
    max_similarity = 0.0
    cents = 0.0
    for x in np.arange(1,75, dtype=float):
#       cent_diff = 1200.*np.log10((x+1.)/x)/np.log10(2.) - int_cents
#       if cent_diff > CENT_DIFF_MAX:
#           continue
        for y in np.arange(x, 99., dtype=float):
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

def get_most_harmonic_ratios_equal_temp():
    real_num = [2.**(x/1200.) for x in  np.arange(100, 1300, 100, dtype=float)]
    CENT_DIFF_MAX = 22.0
    harmonic_similarity = []
    for i, num in enumerate(real_num):
        max_similarity, best_ratio, cents = calculate_most_harmonic_neighbour(int_cents)
        harmonic_similarity.append(max_similarity)
    return np.array(harmonic_similarity)

# Takes as input the scale given in cents from 0 to 1200
def get_similarity_rating_any_scale(scale_cents):
    all_ints = [scale_cents[j] - scale_cents[i] for i in range(len(scale_cents)) for j in range(i+1, len(scale_cents))]
    real_num = [2.**(x/1200.) for x in  all_ints]
    harmonic_similarity = []
    for i, num  in enumerate(real_num):
        int_cents = 1200.*np.log10(num)/np.log10(2.)
        max_similarity, best_ratio, cents = calculate_most_harmonic_neighbour(int_cents)
        harmonic_similarity.append(max_similarity)
    return np.array(harmonic_similarity)
 
def get_harmonic_similarity_score_equal_temp(df):
    INTERVAL_SCORE = get_most_harmonic_ratios_equal_temp()
    def fn(x):
        a = [int(y) for y in x]
        return np.mean(INTERVAL_SCORE[[l-1 for i in range(len(a)) for l in list(np.cumsum(a[i:]))]])
    df['score_eq'] = df.str_fmt.apply(lambda x: fn(x))
    return df

def get_harmonic_similarity_score_equal_temp_extra_notes(df):
    INTERVAL_SCORE = get_most_harmonic_ratios_equal_temp()
    def fn(x):
        a = [int(y) for y in x*2]
        return np.mean(INTERVAL_SCORE[[l-1 for i in range(len(x)) for l in list(np.cumsum(a[i:i+len(x)]))]])
    df['score_eq_en'] = df.str_fmt.apply(lambda x: fn(x))
    return df

def dataframe_possible_scales(df):
    sc_df = pd.DataFrame(columns=['n_notes', 'code', 'str_fmt', 'family'])
    for row in df.itertuples():
        for fams in row.families.split(';'):
            sp = fams.split('-')
            fam = int(sp[0])
            for scale in sp[1].split(','):
                sc_df.loc[len(sc_df)] = [row.notes_in_scale, row.code, scale, fam]
    return sc_df

def plot_score_histograms(df):
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    ax = ax.reshape(4)
    lbls = ['score' + s for s in ['', '_en', '_eq', '_eq_en']]
    for i, a in enumerate(ax):
        for n in df.n_notes.unique():
            sns.distplot(df.loc[df.n_notes==n, lbls[i]], label=str(n), kde=True, ax=a)
        a.legend(loc='best')
        a.set_ylim(0,2)
    plt.show()
    

def get_intervals_as_list(df, i):
    ints = [int(x) for x in df.loc[i,['1','2','3','4']]]
    return ''.join([ j for i in range(4) for j in ints[i]*str(i+1) ])

def get_translational_invariants(non_ident):
    trans_inv = []
    variants = set()
    families = {}
    for ni in non_ident:
        if ni in variants:
            continue
        var_set = set(''.join(np.roll([x for x in ni],i)) for i in range(len(ni)) )
        [variants.add(x) for x in var_set]
        trans_inv.append(ni)
        families.update({ni:var_set})
    return trans_inv, families

def get_unique_scales(df):
    for row in df.itertuples():
        if np.isnan(row.possible_arrangements):
            ll = get_intervals_as_list(df, row[0])
            non_identical = [''.join(x) for x in set(permutations(ll))]
            df.loc[row[0],'n_ni'] = len(non_identical)

            trans_inv, families = get_translational_invariants(non_identical)
            df.loc[row[0],'n_ti'] = len(trans_inv)

            ti_str = ';'.join([str(i)+'-'+ trans_inv[i] for i in range(len(trans_inv))])
            fam_str = ';'.join([str(i)+'-'+ ','.join(families[trans_inv[i]]) for i in range(len(trans_inv))])

            
            df.loc[row[0],'trans_inv'] = ti_str
            df.loc[row[0],'families'] = fam_str
            df.loc[row[0],'possible_arrangements'] = ti_str
    return df

def match_scale_with_instrument(scale):
    in_mask = np.where(INST)[0]
    sc_mask = np.array([int(x) for x in scale], dtype=bool)
    notes_per_key = []
    for key in range(12):
        notes_per_key.append(sum([1 for x in np.where(sc_mask[key:])[0] if x in in_mask ]))
    return ';'.join([str(x) for x in notes_per_key])

def count_those_over_75p(mix):
    counts, n_max = mix.split('-')
    return sum([1 for x in counts.split(';') if float(x) >= 0.75*(float(n_max)*2.+1.)])

def count_those_over_85p(mix):
    counts, n_max = mix.split('-')
    return sum([1 for x in counts.split(';') if float(x) >= 0.85*(float(n_max)*2.+1.)])

def new_score(mix):
    counts, n_notes = mix.split('-')
    n_max = int(n_notes) * 2 + 1
    points = {n_max-i: max(5-i,0) for i in range(n_max)}
    return sum([points[int(x)] for x in counts.split(';')])

def a_n_u(scale):
    in_mask = np.where(INST)[0]
    sc_mask = np.array([int(x) for x in scale], dtype=bool)
    notes_in_scale = len(in_mask)
    notes_per_key = []
    for key in range(12):
        notes_per_key.append(sum([1 for x in in_mask if x in np.where(sc_mask[key:])[0] ])) 
    return sum([ 1 for x in notes_per_key if x == notes_in_scale ])


def all_notes_used(df):
    df['all_notes'] = df['mask'].apply(lambda x: a_n_u(x))
    return df

def get_inst_var_score(df):
    df['inst_keys'] = df['mask'].apply(lambda x: match_scale_with_instrument(x))
    df['inst_score'] = df.inst_keys.apply(lambda x: sum([int(y) for y in x.split(';')]))
    df['inst_norm'] = df.inst_score.astype(float) / (df.n_notes.astype(float) * 2. + 1.) / 12.
    df['inst_std'] = df.inst_keys.apply(lambda x: np.std([float(y) for y in x.split(';')])) /  (df.n_notes.astype(float) * 2. + 1.) / 12.
    tmp_df = df.inst_keys + '-' + df.n_notes.astype(str)
    df['inst_75p'] = tmp_df.apply(lambda x: count_those_over_75p(x))
    df['inst_85p'] = tmp_df.apply(lambda x: count_those_over_85p(x))
    df['inst_new_score'] = tmp_df.apply(lambda x: new_score(x))
    return df

def df_cols_as_int(df):
    cols = ['1','2','3','4','possible_arrangements','notes_in_scale','n_ni','n_ti']
    df.loc[:, cols] = df.loc[:, cols].astype(int)
    return df

def get_codes(df):
    for row in df.itertuples():
        df.loc[row[0],'code'] = ''.join([str(row[i]) for i in range(1,5)])
    return df

#ef reformat_scales_as_mask(df):
#   st = '000000000001'
#   fn = lambda x: '1' + ''.join([st[-int(i):] for i in x])
#   idx = df.loc[df.Tuning.apply(lambda x: x not in ['Unique', 'Turkish'])].index
#   df.loc[idx, 'mask'] = df.loc[idx, 'Intervals'].apply(fn)
#   return df
def reformat_scales_as_mask(df):
    st = '000000000000001'
    fn = lambda x: '1' + ''.join([st[-int(i):] for i in x])
    idx = df.loc[df.Tuning.apply(lambda x: x not in ['Unique', 'Turkish', '53-tet'])].index
    df.loc[idx, 'mask'] = df.loc[idx, 'Intervals'].apply(fn)

    fn = lambda x: '1' + ''.join([st[-int(i):] for i in x.split(';')])
    idx = df.loc[df.Tuning=='53-tet'].index
    df.loc[idx, 'mask'] = df.loc[idx, 'Intervals'].apply(fn)
    return df

def extract_scales_and_ints_from_scales(df):
    names = []
    scales = []
    all_ints = []
    pair_ints = []
    cultures = []
    tunings = []
    conts = []
    for row in df.itertuples():
        try:
            idx = np.where(np.array([int(x) for x in row.mask]))[0]
        except:
            pass

        if row.Tuning in ['12-tet', 'Just']:
            scale = EQ12_INTS[idx]
            names.append(row.Name)
            scales.append(scale)
            all_ints.append([scale[i] - scale[j] for j in range(len(scale)) for i in range(j+1,len(scale))])
            pair_ints.append([scale[j+1] - scale[j] for j in range(len(scale)-1)])
            cultures.append(row.Culture)
            tunings.append(row.Tuning)
            conts.append(row.Continent)
#       elif row.Tuning == 'Just' or '12-tet':
            scale = JI_INTS[idx]
        elif row.Tuning == 'Perfect fifths/fourths':
            scale = PYT_INTS[idx]
        elif row.Tuning == 'Arabian':
            scale = EQ24_INTS[idx]
        elif row.Tuning == 'Dastgah-ha':
            scale = DASTGAH[idx]
        elif row.Tuning == 'Vietnamese':
            scale = VIET[idx]
        elif row.Tuning == 'Turkish':
            scale = np.cumsum([0.0] + [TURKISH[a] for a in row.Intervals])
        elif row.Tuning == 'Khmer':
            for KHM in [KHMER_1, KHMER_2]:
                base = KHM[[i-1 for i in idx[1:]]]
                for i in range(len(base)):
                    scale = np.cumsum([0.] + np.roll(KHM,i))
                    names.append(row.Name)
                    scales.append(scale)
                    all_ints.append([scale[i] - scale[j] for j in range(len(scale)) for i in range(j+1,len(scale))])
                    pair_ints.append([scale[j+1] - scale[j] for j in range(len(scale)-1)])
                    cultures.append(row.Culture)
                    tunings.append(row.Tuning)
                    conts.append(row.Continent)
            continue
        elif row.Tuning == 'Unique':
            scale = np.cumsum([0.] + [float(x) for x in row.Intervals.split(';')])
        else:
          continue

        names.append(row.Name)
        scales.append(scale)
        all_ints.append([scale[i] - scale[j] for j in range(len(scale)) for i in range(j+1,len(scale))])
        pair_ints.append([scale[j+1] - scale[j] for j in range(len(scale)-1)])
        cultures.append(row.Culture)
        tunings.append(row.Tuning)
        conts.append(row.Continent)

    return cultures, tunings, conts, names, scales, all_ints, pair_ints

# This will not work in all instances!!!
# A proper clustering algorithm is needed
def get_dist_order(pair_ints):
    order = []
    for ints in pair_ints:
        uniq = np.array(list(set(ints)))
        idx = [sum([1 for j in range(i+1, len(uniq)) if abs(uniq[i] - uniq[j]) < 45.]) for i in range(len(uniq))]
        order.append(np.where(np.array(idx)==0)[0].size)
    return order


def reformat_real_scales_as_strings(df):
    for row in df.itertuples():
        df.loc[row[0],'str_fmt'] = ''.join([str(row[i]) for i in range(1,13) if row[i]])
    return df

def match_scale_with_family(df, rs):
    for code in rs.code.unique():
        dfIdx = df.loc[df.code==code].index[0]
        fam_dict = {}
        fams = [{z:x.split('-')[0] for z in x.split('-')[1].split(',')}  for x in df.loc[dfIdx,'families'].split(';')]
        for f in fams:
            fam_dict.update(f)

        associated_scales = []
        for row in rs.loc[rs.code==code].itertuples():
            famIdx = fam_dict[row.str_fmt]
            associated_scales.append(famIdx + '-' + row.Names)
        if len(associated_scales):
            df.loc[dfIdx, 'real_scales'] = ';'.join(associated_scales)
            df.loc[dfIdx, 'n_rs'] = len(associated_scales)

    return df

def get_2grams_dist(df, dI=10, imin=0, imax=620):
    int_bins = np.arange(imin, imax+dI, dI)
    nI = int_bins.size
    dist = np.zeros((nI, nI), dtype=float)
    for p_int in df.pair_ints:
        pi = [int(x) for x in p_int.split(';')]
        for i in range(len(pi)-1):
            x = int(pi[i] / float(dI))
            y = int(pi[i+1] / float(dI))
            dist[x,y] += 1.0
            dist[y,x] += 1.0
    return dist
    
def plot_2gram_dist_by_n_notes(df, dI=40, cond=False, prune=0):
    fig, ax = plt.subplots(2,3)
    ax = ax.reshape(ax.size)
    for i, n in enumerate([4,5,6,7,8,9]):
        dist = get_2grams_dist(df.loc[df.n_notes==n], dI=dI)
        if cond:
            for j in range(dist.shape[0]):
                tot = sum(dist[j])
                if tot < prune:
                    dist[j] = 0
                if tot:
                    dist[j] = dist[j] / tot
        else:
            if prune:
               dist[dist<=prune] = 0
        sns.heatmap(np.log(dist+0.1), label=str(n), ax=ax[i])
        ax[i].invert_yaxis()
        ax[i].set_title(str(n))
    plt.show()

def plot_pair_ints_by_n_notes(df):
    fig, ax = plt.subplots(4,2)
    ax = ax.reshape(ax.size)
    n_notes = sorted(df['n_notes'].unique())
    for i, t in enumerate(n_notes):
        cultures, tunings, names, scales, all_ints, pair_ints = extract_scales_and_ints_from_scales(df.loc[df['n_notes']==t])
        sns.distplot([y for x in pair_ints for y in x], bins=120, label=str(t), ax=ax[i])
        ax[i].legend(loc='best')
        sns.distplot([y for x in pair_ints for y in x], bins=120, label=str(t), ax=ax[-1])
    plt.show()

def plot_pair_ints_by_n_notes_one_graph(df):
    fig, ax = plt.subplots()
    n_notes = np.arange(4,10) 
    for i, t in enumerate(n_notes):
        cultures, tunings, names, scales, all_ints, pair_ints = extract_scales_and_ints_from_scales(df.loc[df['n_notes']==t])
#       sns.distplot([y for x in pair_ints for y in x], bins=120, label=str(t), ax=ax)
        sns.kdeplot([y for x in pair_ints for y in x], label=str(t), ax=ax)
    ax.legend(loc='best')
         
    plt.show()

def plot_pair_ints_by_tuning(df):
    fig, ax = plt.subplots(6,2)
    ax = ax.reshape(12)
    tunings = df.Tuning.unique()
    for i, t in enumerate(tunings):
        cultures, tunings, names, scales, all_ints, pair_ints = extract_scales_and_ints_from_scales(df.loc[df['n_notes']==t])
        sns.distplot([y for x in pair_ints for y in x], bins=120, label=t, ax=ax[i])
        ax[i].legend(loc='best')
    plt.show()

def plot_order_vs_n_notes_distplot(order, pair_ints):
    len_scales = [len(x) for x in pair_ints]
    n_notes = np.unique(len_scales)
    fig, ax = plt.subplots(2,1)
    order = np.array(order)
    for i in range(len(n_notes)):
        idx = np.where(len_scales==n_notes[i])[0]
        sns.distplot(order[idx], ax=ax[0], label=str(n_notes[i]))
        sns.kdeplot(order[idx], ax=ax[1], label=str(n_notes[i]))
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    plt.show()

def plot_order_vs_n_notes_heatmap(order, pair_ints):
    len_scales = [len(x) for x in pair_ints]
    x1 = np.unique(order)
    y1 = np.unique(len_scales)
    z1 = np.zeros((x1.size, y1.size))
    for i in range(len(order)):        
            x = np.where(x1==order[i])[0]
            y = np.where(y1==len_scales[i])[0]
            z1[x,y] = z1[x,y] + 1.0
    fig, ax = plt.subplots()
    sns.heatmap(np.log10(z1+1), ax=ax, cmap='Greys')
    ax.set_xticklabels(y1)
    ax.set_yticklabels(x1)
    plt.show()

def plot_score_by_cat(df, cat='Tuning', score='s1'):
    uni = df.loc[:,cat].unique()
    if uni.size <=12:
        fig, ax = plt.subplots(4,3)
    elif uni.size <=24:
        fig, ax = plt.subplots(6,4)
    ax = ax.reshape(ax.size)
    for i, u in enumerate(uni):
        if df.loc[(df.loc[:,cat]==u)&(df.loc[:,score].notnull()), score].size ==0:
            sns.distplot(df.loc[(df.loc[:,score].notnull()), score], ax=ax[i], bins=100, label='all')
            continue
        sns.distplot(df.loc[(df.loc[:,cat]==u)&(df.loc[:,score].notnull()), score], ax=ax[i], bins=100, label=u)
        ax[i].legend(loc='best')
    plt.show()

def calculate_energy_harmonic(ints, base):
    return np.mean([(round(i/base) - i/base)**2 for i in ints])

def calculate_energy(ints, base):
    return np.mean([(round(i/base) - i/base)**2 * float(round(i/base)) for i in ints])

def get_min_energy_integer_ratios(pair_ints):
    ratios = []
    base_o = []
    base_ints = np.arange(35., 155., 5.)
    for pi in pair_ints:
        energy = np.zeros(base_ints.size, dtype=float)
        for i, base in enumerate(base_ints):
            energy[i] = calculate_energy(pi, base)
        ratios.append([x / base_ints[np.argmin(energy)] for x in pi ])
        base_o.append(base_ints[np.argmin(energy)]) 
        print(base_ints[np.argmin(energy)], pi)
        print(energy)
    return ratios, base_o

def reformat_original_csv_data(df):
    new_df = pd.DataFrame(columns=['Name', 'Intervals', 'Culture', 'Continent', 'Tuning'])
    for i, col in enumerate(df.columns):
        tuning  = df.loc[0, col]
        culture = df.loc[1, col]
        cont  = df.loc[2, col]
        name = '_'.join([culture, col])
        ints = ';'.join([str(int(round(float(x)))) for x in df.loc[3:, col] if not str(x)=='nan'])
        new_df.loc[i] = [name, ints, culture, cont, tuning]
    return new_df

def extract_scales_and_ints_from_unique(df):
    names = []
    scales = []
    all_ints = []
    pair_ints = []
    cultures = []
    tunings = []
    conts = []
    for row in df.itertuples():
        ints = [int(x) for x in row.Intervals.split(';')]
        if sum(ints) < (1200 - OCT_CUT):
            continue

        start_from = 0
        for i in range(len(ints)):
            if i < start_from:
                continue
            sum_ints = np.cumsum(ints[i:], dtype=int)
            if sum_ints[-1] < (1200 - OCT_CUT):
                break
            # Find acceptable octave and call everything
            # in between a scale
            idx_oct = np.argmin(np.abs(sum_ints-1200))
            oct_val = sum_ints[idx_oct]
            if abs(oct_val - 1200) > OCT_CUT:
                continue
            
            scale = [0.] + list(sum_ints[:idx_oct+1])
            names.append(row.Name)
            scales.append(scale)
            all_ints.append([scale[i] - scale[j] for j in range(len(scale)) for i in range(j+1,len(scale))])
            pair_ints.append([scale[j+1] - scale[j] for j in range(len(scale)-1)])
            cultures.append(row.Culture)
            tunings.append(row.Tuning)
            conts.append(row.Continent)

            start_from = idx_oct

    return cultures, tunings, conts, names, scales, all_ints, pair_ints

def weighted_mean(x1, x2, n1, n2):
    return (x1 * float(n1) + x2 * float(n2)) / float(n1+n2)

def amalgamate_dataframes(real_df, df1, df2):
    out = []
    n_n5 = real_df.loc[real_df.n_notes==5].shape[0]
    n_n7 = real_df.loc[real_df.n_notes==7].shape[0]
    for i, row in enumerate(df1.itertuples()):
        idx = df2.loc[(df2.cat=='pair_ints')&(df2.min_int==row.min_int)&(df2.max_int==row.max_int)&(df2.bias==row.bias)&(df2['quantile']==row.quantile)].index
        if len(idx):
            JSD   = weighted_mean(row.JSD, df2.loc[idx, 'JSD'].values[0], n_n5, n_n7)
            fr_20 = weighted_mean(row.fr_20, df2.loc[idx, 'fr_20'].values[0], n_n5, n_n7)
            out.append(['pair_ints', row.min_int, row.max_int, row.bias, row.quantile, row.bias_group, JSD, fr_20])
    return pd.DataFrame(data=out, columns=['cat', 'min_int', 'max_int', 'bias', 'quantile', 'bias_group', 'JSD', 'fr_20'])

#ef match_model(row, df):
#       idx = df.loc[(df.n_notes==5)&(df.min_int==row.min_int)&(df.max_int==row.max_int)&(df.bias==row.bias)&(df.beta==row.beta)].index
#       if len(idx):
#           JSD   = weighted_mean(df.loc[row[0],X], df.loc[idx, X].values[0], n_n5, n_n7)
#           fr_20 = weighted_mean(df.loc[row[0],Y], df.loc[idx, Y].values[0], n_n5, n_n7)
#           out.append(['pair_ints', row.min_int, row.max_int, row.bias, row.beta, row.bias_group, JSD, fr_20])

def amalgamate_model_results(df, real_df, X='met1', Y='met2'):
    out = []
    n_n5 = real_df.loc[real_df.n_notes==5].shape[0]
    n_n7 = real_df.loc[real_df.n_notes==7].shape[0]
    for row in df.loc[df.n_notes==7].itertuples():
        idx = df.loc[(df.n_notes==5)&(df.min_int==row.min_int)&(df.max_int==row.max_int)&(df.bias==row.bias)&(df.beta==row.beta)].index
        if len(idx):
#           JSD   = weighted_mean(df.loc[row[0],X], df.loc[idx, X].values[0], n_n5, n_n7)
#           fr_20 = weighted_mean(df.loc[row[0],Y], df.loc[idx, Y].values[0], n_n5, n_n7)
            JSD   = (df.loc[row[0],X] * df.loc[idx, X].values[0])**0.5
            fr_20 = (df.loc[row[0],Y] * df.loc[idx, Y].values[0])**0.5
            mfr_10 = (df.loc[row[0],'mfr_10'] * df.loc[idx, 'mfr_10'].values[0])**0.5
            out.append(['pair_ints', row.min_int, row.max_int, row.bias, row.beta, row[7], row.bias_group, idx[0], row[0], JSD, fr_20, mfr_10])
    return pd.DataFrame(data=out, columns=['cat', 'min_int', 'max_int', 'bias', 'beta', 'quantile', 'bias_group', 'idx_5', 'idx_7', X, Y, 'mfr_10'])

def test_sample_size(df, cat):
#   fig, ax = plt.subplots()
    Y = []
    X = [30, 100, 300, 1000, 3000, 10000]
    for n in X:
        idx = df.index[np.random.rand(len(df)).argsort()[:n]]
#       sns.kdeplot(extract_floats_from_string(df.loc[idx, 'pair_ints']), ax=ax, label=str(n))
        Y.append(float(len(set([int(x) for y in df.loc[idx,cat] for x in y.split(';') if len(y)]))))
    plt.plot(X, Y)

def find_min_pair_int_dist(b, c):
    dist = 0.0
    for i in range(len(b)):
        dist += np.min(np.abs(c-b[i]))
    return dist

def pair_int_distance(pair_ints):
    pair_dist = np.zeros((len(pair_ints), len(pair_ints)), dtype=float)
    for i in range(len(pair_ints)):
        for j in range(len(pair_ints)):
            dist1 = find_min_pair_int_dist(pair_ints[i], pair_ints[j])
            dist2 = find_min_pair_int_dist(pair_ints[j], pair_ints[i])
            pair_dist[i,j] = (dist1 + dist2) * 0.5
    return pair_dist

def cluster_pair_ints(df, n_clusters):
    pair_ints = np.array([np.array([float(x) for x in y.split(';')]) for y in df.pair_ints])
    pair_dist = pair_int_distance(pair_ints)
    li = linkage(pdist(pair_dist), 'ward')
    return fcluster(li, li[-n_clusters,2], criterion='distance')

def get_links_from_pair_dist(pair_dist):
    return linkage(pdist(pair_dist), 'ward')

def get_cluster_from_links(li, n_clusters):
    return fcluster(li, li[-n_clusters,2], criterion='distance')

def cluster_dendrogram(df, n_clusters, pair_dist='None'):
    if pair_dist == 'None':
        pair_ints = np.array([np.array([float(x) for x in y.split(';')]) for y in df.pair_ints])
        pair_dist = pair_int_distance(pair_ints)
    li = linkage(pdist(pair_dist), 'ward')
    dendrogram(li)

def plot_cluster_size_distributions(li):
    fig, ax = plt.subplots(5,4)
    ax = ax.reshape(ax.size)
    for i, n in enumerate(range(2,22)):
        cut = li[-n+1,2]
        nc = fcluster(li, cut, criterion='distance')
        print(f'{n} clusters:   distance between this cluster and the previous = {cut - li[-n,2]}')
        sns.countplot(nc, ax=ax[i])

def label_scales_by_cluster(df, n=16):
    idx = df.loc[(df.n_notes>3)&(df.n_notes<10)].index
    nc = cluster_pair_ints(df.loc[idx], n)
    df.loc[idx, f"cl_{n:02d}"] = nc
    return df

def mixing_cost(ints):
#   [(ints[(i-1)] + ints[i%len(ints)] - 2400./float(len(ints)))**2 for i in range(1,len(ints)+1)]
    return np.mean([np.abs(ints[(i-1)] + ints[i%len(ints)] - 2400./float(len(ints)))**2 for i in range(1,len(ints)+1)])**0.5

def mixing_cost_arr(arr):
    return [np.mean([np.abs(ints[(i-1)] + ints[i%len(ints)] - 2400./float(len(ints)))**2 for i in range(1,len(ints)+1)])**0.5 for ints in arr] 

def get_probability_from_costs(costs):
    costs = np.array(costs)
    return [np.exp(1./c) / np.sum(np.exp(1./costs)) for c in costs]

def mix_pair_ints(ints):
    np.random.seed()
    ints = np.array([float(x) for x in ints.split(';')])
    arrangements = np.array(list(set(permutations(ints))))
    costs = np.array([mixing_cost(a) for a in arrangements])
    costs = costs / costs.max()
    probabilities = np.cumsum([np.exp(1./c) / np.sum(np.exp(1./costs)) for c in costs])
    return arrangements[np.where(probabilities>np.random.rand())[0][0]]


def mix_pair_ints_df(df):
    df['pair_ints'] = df.pair_ints.apply(mix_pair_ints)
    return df

def permute_scale(int_str):
    ints = np.array([int(x) for x in int_str.split(';')])
    return np.array(list(set(permutations(ints))))

def get_real_scale_permutations(df_real):
    pool = mp.Pool(28)
    perm = pool.map(permute_scale, df_real.pair_ints)
    df_real['perm'] = perm
    pool.close()
    return df_real

def metrics(df_real, df,  test_cases):
    metric_cols = ['type', 'geo_norm', 'err_sq', 'bp_geo_norm',
                   'bp_err_sq', 'cum_geo_norm', 'cum_err_sq', 'peak_ratio', 'peak_dist',
                   'deriv_gn', 'deriv_es']
    out_df = pd.DataFrame(columns=metric_cols + list(test_cases[0].keys()))
    for tc in test_cases:
        df_model = pd.read_feather(df.loc[tc['i'],'fName'])
        grid, y1, xxx, hist1 = smooth_dist_kde(df_real.loc[df_real.n_notes==tc['n']], cat='pair_ints', hist=True)
        grid, y2, xxx, hist2 = smooth_dist_kde(df_model, cat='pair_ints', hist=True)

        out = try_out_metrics(y1, y2)
        out_df.loc[len(out_df), metric_cols] = ['dist'] + out
        for k, v in tc.items():
            out_df.loc[len(out_df)-1, k] = v

        out = try_out_metrics(hist1, hist2)
        out_df.loc[len(out_df), metric_cols] = ['hist'] + out
        for k, v in tc.items():
            out_df.loc[len(out_df)-1, k] = v
        
    return out_df

def try_out_metrics(y1, y2):
    y1 = y1.reshape(y1.size)
    y2 = y2.reshape(y2.size)

    geo_norm = np.sqrt(np.dot(y1, y2))
    err_sq = np.sqrt(np.dot(y1-y2, y1-y2))
    
    balpeak_geo_norm = np.sqrt(np.dot(y1, y2/y2.max()*y1.max()))
    balpeak_err_sq = np.sqrt(np.dot(y1-y2/y2.max()*y1.max(), y1-y2/y2.max()*y1.max()))

    cum_geo_norm = np.sqrt(np.dot(np.cumsum(y1), np.cumsum(y2)))
    cum_err_sq = np.sqrt(np.dot(np.cumsum(y1) - np.cumsum(y2), np.cumsum(y1) - np.cumsum(y2)))

    peak1 = argrelextrema(y1, np.greater)[0]
    peak2 = argrelextrema(y2, np.greater)[0]
    peak_ratio = float(len(peak1)) / float(len(peak2))
    peak_dist = 0.0
    for p1 in peak1:
        peak_dist += np.min(np.abs(peak2-p1))

    d1 = y1[1:] - y1[:-1]
    d2 = y2[1:] - y2[:-1]
    deriv_gn = np.sqrt(np.dot(d1, d2))
    deriv_es = np.sqrt(np.dot(d1-d2, d1-d2))

    output = [geo_norm, err_sq, balpeak_geo_norm, balpeak_err_sq, cum_geo_norm,
              cum_err_sq, peak_ratio, peak_dist, deriv_gn, deriv_es]

    return output

def scale_metrics(y1, y2):
    y1 = y1.reshape(y1.size)
    y2 = y2.reshape(y2.size)
    err_sq = np.sqrt(np.dot(y1-y2, y1-y2))
    d1 = y1[1:] - y1[:-1]
    d2 = y2[1:] - y2[:-1]
    deriv_es = np.sqrt(np.dot(d1-d2, d1-d2))
    return [err_sq, deriv_es]
    
#   print(y1.shape, y2.shape)
#   print('Geometric norm')
#   print(f'kde:   {np.sqrt(np.dot(y1, y2))}')
#   print(f'hist:  {np.sqrt(np.dot(hist1, hist2))}')
#   print('Root_sum_error squared')
#   print(f'kde:   {np.sqrt(np.dot(y1-y2, y1-y2))}')
#   print(f'hist:  {np.sqrt(np.dot(hist1-hist2, hist1-hist2))}')

#   print('Balanced peaks geometric norm')
#   print(f'kde:   {np.sqrt(np.dot(y1, y2/y2.max()*y1.max()))}')
#   print(f'hist:   {np.sqrt(np.dot(hist1, hist2/hist2.max()*hist1.max()))}')
#   print('Balanced peaks RSES')
#   print(f'kde:   {np.sqrt(np.dot(y1-y2/y2.max()*y1.max(), y1-y2/y2.max()*y1.max()))}')
#   print(f'hist:   {np.sqrt(np.dot(hist1-hist2/hist2.max()*hist1.max(), hist1-hist2/hist2.max()*hist1.max()))}')

#   print('Cumulative distribution:')
    
def get_ss_set(fName, X='ss_w20'):
    df = pd.read_feather(fName)
    return set([int(x) for y in df[X] for x in y.split(';') if len(y)])

def get_ss_list(fName, X='ss_w20'):
    df = pd.read_feather(fName)
    return [int(x) for y in df[X] for x in y.split(';') if len(y)]

def update_best_df(df_best, df, df_real):
    try:
        df_best = df_best.drop(columns=['idx_5', 'idx_7'])
    except:
        pass

    note_count = df_real.loc[(df_real.n_notes>=4)&(df_real.n_notes<=9), 'n_notes'].value_counts()
    note_frac = note_count / sum(note_count)

    for i in df_best.index:
        mi, ma, bias, beta = df_best.loc[i, ['min_int', 'max_int', 'bias', 'beta']]
        lbl = df_best.loc[i, 'bias']
        met1  = []
        fr_10 = []
        met1_ave  = []
        fr_10_ave = []
        fm_10_ave = []
        mfr_10_ave = []
        fr_20_ave = []
        fm_20_ave = []
        n_total = 0.0
        n_totalmix = 0.0
        ss_10 = set()
        ss_20 = set()

        pi_10_ave = []
        pm_10_ave = []
        pi_20_ave = []
        pm_20_ave = []
        pi_10 = set()
        pi_20 = set()
        
        quant = []
        for n in range(4,10):
            idx = df.loc[(df.min_int==mi)&(df.max_int==ma)&(df.bias==bias)&(df.beta==beta)&(df.n_notes==n)].index
            try:
#           if len(idx):
                idx = idx[0]
                met1.append(df.loc[idx, 'met1'])
                fr_10.append(df.loc[idx, 'fr_10'])

                df_best.loc[i, f'met1_{n}'] = met1[-1]
                df_best.loc[i, f'fr10_{n}'] = fr_10[-1]

                quant.append(df.loc[idx, 'quantile'])

                met1_ave.append(met1[-1]**note_frac[n])
                fr_10_ave.append(fr_10[-1]*note_frac[n])
                fm_10_ave.append(df.loc[idx, 'fm_10']*note_frac[n])
#               fr_20_ave.append(df.loc[idx, 'fr_20']*note_frac[n])
#               fm_20_ave.append(df.loc[idx, 'fm_20']*note_frac[n])
#               pi_10_ave.append(df.loc[idx, 'pi_10']*note_frac[n])
#               pi_20_ave.append(df.loc[idx, 'pi_20']*note_frac[n])
#               pm_10_ave.append(df.loc[idx, 'pm_10']*note_frac[n])
#               pm_20_ave.append(df.loc[idx, 'pm_20']*note_frac[n])
                n_total += note_frac[n]
#               if not np.isnan(df.loc[idx, 'mfr_10']):
#                   mfr_10_ave.append(df.loc[idx, 'mfr_10']*note_frac[n])
#                   n_totalmix += note_frac[n]

                df_m = pd.read_feather(df.loc[(df.min_int==mi)&(df.max_int==ma)&(df.bias==bias)&(df.beta==beta)&(df.n_notes==n), 'fName'].values[0])
                [ss_10.add(z) for z in set([y for x in df_m.ss_w10 for y in x.split(';') if len(x)])]
                [ss_20.add(z) for z in set([y for x in df_m.ss_w20 for y in x.split(';') if len(x)])]
#               [pi_10.add(z) for z in set([y for x in df_m.pi_w10 for y in x.split(';') if len(x)])]
#               [pi_20.add(z) for z in set([y for x in df_m.pi_w20 for y in x.split(';') if len(x)])]
                
#           else:
            except Exception as e:
                print(e)
                print(n, mi, ma, bias, beta)
#               print(df.loc[(df.min_int==mi)&(df.max_int==ma)&(df.bias==bias)&(df.beta==beta)&(df.n_notes==n), 'fName'].values)
                continue

        df_best.loc[i, 'met1_ave'] = np.product(met1_ave) ** n_total
        df_best.loc[i, 'fr10_ave'] = np.sum(fr_10_ave) ** n_total
        df_best.loc[i, 'fm10_ave'] = np.sum(fm_10_ave) ** n_total
#       df_best.loc[i, 'mfr10_ave'] = np.sum(mfr_10_ave) ** n_totalmix
#       df_best.loc[i, 'fr20_ave'] = np.sum(fr_20_ave) ** n_total
#       df_best.loc[i, 'fm20_ave'] = np.sum(fm_20_ave) ** n_total

#       df_best.loc[i, 'pi10_ave'] = np.sum(pi_10_ave) ** n_total
#       df_best.loc[i, 'pi20_ave'] = np.sum(pi_20_ave) ** n_total
#       df_best.loc[i, 'pm10_ave'] = np.sum(pm_10_ave) ** n_total
#       df_best.loc[i, 'pm20_ave'] = np.sum(pm_20_ave) ** n_total

        df_best.loc[i, 'ss_10'] = ';'.join(sorted(list(ss_10)))
        df_best.loc[i, 'ss_20'] = ';'.join(sorted(list(ss_20)))
#       df_best.loc[i, 'pi_10'] = ';'.join(sorted(list(pi_10)))
#       df_best.loc[i, 'pi_20'] = ';'.join(sorted(list(pi_20)))

        df_best.loc[i, 'quantile'] = np.mean(quant)
        df_best.loc[i, 'fin'] = sum([1 for n in range(4,10) if not np.isnan(df_best.loc[i, f"met1_{n}"])])

    return df_best

def get_df_idx_from_best(df_best, df, i, n):
    mi, ma, bias, beta = df_best.loc[i, ['min_int', 'max_int', 'bias', 'beta']]
    try:
        return df.loc[(df.min_int==mi)&(df.max_int==ma)&(df.bias==bias)&(df.beta==beta)&(df.n_notes==n)].index[0]
    except:
        return []
        
def load_df_from_best(df_best, df, i, n):
    mi, ma, bias, beta = df_best.loc[i, ['min_int', 'max_int', 'bias', 'beta']]
    return pd.read_feather(df.loc[(df.min_int==mi)&(df.max_int==ma)&(df.bias==bias)&(df.beta==beta)&(df.n_notes==n), 'fName'].values[0])

def get_ss_set_from_model(df_best, df, i, X='ss_w20'):
    all_idx = []
    for n in range(4,10):
        try:
            [all_idx.append(x) for x in get_ss_set(df.loc[get_df_idx_from_best(df_best, df, i, n), 'fName'], X)]
        except:
            pass
    return sorted(list(set(all_idx)))

def remove_matched_edge(con1, con2, edge):
    con1 = {i:np.delete(con1[i], np.where(con1[i]==edge[1])[0]) for i in range(len(con1))}
    con2 = {i:np.delete(con2[i], np.where(con2[i]==edge[0])[0]) for i in range(len(con2))}
    con1[edge[0]] = []
    con2[edge[1]] = []
    return con1, con2

def check_if_pair_ints_equivalent(pi1, pi2, d=10):
    N = len(pi1)
    con1 = {i:np.where(abs(pi2-pi1[i])<=d)[0] for i in range(N)}
    if not all([len(c) for c in con1.values()]):
        return False
    elif not len(set([a for c in con1.values() for a in c])) == N:
        return False
    else:
        con2 = {i:np.where(abs(pi1-pi2[i])<=d)[0] for i in range(N)}
        matched1 = set()
        matched2 = set()
        count = 0

        sys.setrecursionlimit(100000)

        def match_nodes(con1, con2, matched1, matched2, n=1, count=0):
#           print(n, count, con1)
            if len(matched1) == len(matched2) == len(con1):
#               print(matched1)
#               print(matched2)
                return [True]
            elif sum([len(v) for v in con1.values()]) == 0:
                return [False]
            elif sum([len(v) for v in con2.values()]) == 0:
                return [False]
            count += 1
            if n == 1:
                for i in range(N):
                    if len(con1[i]) == n:
#                       print(con1)
#                       print(con2)
#                       print(matched1)
#                       print(matched2)
                        if con1[i][0] in matched2:
                            return con1, con2, matched1, matched2, n, count, False
                        else:
                            matched1.add(i)
                            matched2.add(con1[i][0])
#                           print('Matched: ', i, con1[i][0])
                            con1, con2 = remove_matched_edge(con1, con2, [i, con1[i][0]])
                            out = match_nodes(con1, con2, matched1, matched2, n=1, count=count)
#                           print(out)
                            if out[-1] != '':
                                return [out[-1]]
                            con1, con2, matched1, matched2, n, count, truth = out
                    elif len(con2[i]) == n:
#                       print(con1)
#                       print(con2)
#                       print(matched1)
#                       print(matched2)
                        if con2[i][0] in matched1:
                            return con1, con2, matched1, matched2, n, count, False
                        else:
                            matched1.add(con2[i][0])
                            matched2.add(i)
#                           print('Matched: ', i, con2[i][0])
                            con2, con1 = remove_matched_edge(con2, con1, [i, con2[i][0]])
                            out = match_nodes(con1, con2, matched1, matched2, n=1, count=count)
#                           print(out)
                            if out[-1] != '':
                                return [out[-1]]
                            con1, con2, matched1, matched2, n, count, truth = out
            if n > 1:
                con1_n = {i:con1[i] for i in range(len(con1)) if len(con1[i]) == n}
                n1_n = len(con1_n)
                con2_n = {i:con2[i] for i in range(len(con2)) if len(con2[i]) == n}
                n2_n = len(con2_n)
                if n1_n == n2_n > 0:
                    i = list(con1_n.keys())[0]
                    matched1.add(i)
                    matched2.add(con1_n[i][0])
#                   print('Matched: ', i, con1[i][0])
                    con1, con2 = remove_matched_edge(con1, con2, [i, con1_n[i][0]])
                    out = match_nodes(con1, con2, matched1, matched2, n=1, count=count)
#                   print(out)
                    if out[-1] != '':
                        return [out[-1]]
                    con1, con2, matched1, matched2, n, count, truth = out
                    
                        
            if count >100:
                print('Counted out...')
                return [None]

            if n <= len(con1):
                out = match_nodes(con1, con2, matched1, matched2, n=n+1, count=count)
                if out[-1] != '':
                    return [out[-1]]
                con1, con2, matched1, matched2, n, count, truth = out
                return con1, con2, matched1, matched2, n, count, truth
            else:
                return con1, con2, matched1, matched2, n, count, ''

        tmp = match_nodes(con1, con2, matched1, matched2)
        return tmp[-1]
        
def how_many_pair_ints_similar(pi_str, df_real, d=10):
    pi1 = np.array([int(x) for x in pi_str.split(';')])
    n = len(pi1)
    idx = df_real.loc[df_real.n_notes==n].index
#   pi_10 = [i for i in idx if check_if_pair_ints_equivalent(pi1, np.array([int(x) for x in df_real.loc[i,'pair_ints'].split(';')]), d=10)]
    pi_10 = []
    for i in idx:
        out = check_if_pair_ints_equivalent(pi1, np.array([int(x) for x in df_real.loc[i,'pair_ints'].split(';')]), d=d)
        if out:
            pi_10.append(i)
        elif out == None:
            print(pi_str, i, df_real.loc[i, 'pair_ints'])
    return ';'.join([str(x) for x in pi_10])

def update_df_real_ss(df_real, df_best, df, idx1, idx2, idx3):
    for w in [10, 20]:
        set1 = set()
        for i in idx1:
            [set1.add(x) for x in get_ss_set_from_model(df_best, df, i, X=f'ss_w{w}')]
        df_real[f'ss_di_{w}'] = [True if x in set1 else False for x in df_real.index]

        set2 = set()
        for i in idx2:
            [set2.add(x) for x in get_ss_set_from_model(df_best, df, i, X=f'ss_w{w}')]
        df_real[f'ss_hs_{w}'] = [True if x in set2 else False for x in df_real.index]

        set3 = set()
        for i in idx3:
            [set3.add(x) for x in get_ss_set_from_model(df_best, df, i, X=f'ss_w{w}')]
        df_real[f'ss_no_{w}'] = [True if x in set3 else False for x in df_real.index]
    df_real.loc[(df_real.ss_di_10)|(df_real.ss_hs_10),'is_f'] = True
    df_real.loc[(df_real.ss_di_10)|(df_real.ss_hs_10),'not_f'] = False

    df_real.loc[(df_real.ss_di_10==False)&(df_real.ss_hs_10==False),'not_f'] = True
    df_real.loc[(df_real.ss_di_10==False)&(df_real.ss_hs_10==False),'is_f'] = False

    return df_real

def calculate_base_probability_ints(df_real, mi, ma):
    X = np.arange(0, 1201)
    n_arr = np.arange(4,10)
    prob = {}
    for n in n_arr:
        Y = np.load(os.path.join(DIST_DIR, f"n{n}_none_MI{mi:d}_MA{ma:d}_BETA_000.000.npy"))
#       X, Y = smooth_dist_kde(pd.read_feather(df.loc[(df.min_int==mi)&(df.max_int==ma)&(df.bias=='none')&(df.n_notes==n), 'fName'].values[0]))
        Y[X<mi] = 0
        Y[X>ma] = 0
        Y = Y / np.sum(Y)
        prob.update({n:Y})

    for i in df_real.index:
        n = df_real.loc[i, 'n_notes']
        if n not in n_arr:
            continue
        ints = [int(x) for x in df_real.loc[i, 'pair_ints'].split(';')]
        df_real.loc[i, f'p_{mi:d}_{ma:d}'] = np.product([prob[n][X==i] for i in ints]) ** (1./n)

    return df_real

def create_base_probability_database(df_real):
    df_real = df_real.loc[(df_real.n_notes>=4)&(df_real.n_notes<=9),['Name','n_notes','pair_ints', 'scale', 'cl_16']]
    for mi, ma in product(np.arange(0,130,10), [400,450,500,550,600,700,800,900,1000,1100,1200]):
        if sum([1 for n in range(4,10) if os.path.exists(os.path.join(DIST_DIR, f"n{n}_none_MI{mi:d}_MA{ma:d}_BETA_000.000.npy"))==False]):
            continue
        df_real = calculate_base_probability_ints(df_real, mi, ma)
    return df_real

def calculate_base_prob(*inputs):
    scale = np.array(inputs[:-1])
    prob = inputs[-1]
    ints = scale[1:] - scale[:-1]
    return np.product([prob[int(i)] for i in ints])

def create_base_probability_database_2(df_real):
    df_real = df_real.loc[(df_real.n_notes>=4)&(df_real.n_notes<=9),['Name','n_notes','pair_ints', 'scale', 'cl_16']]
    ma = 1200
    n_arr = np.arange(4,10)
    pool = mp.Pool(N_PROC)
    for mi in np.arange(70,100,10):
        for n in n_arr:
            print(mi, n)
            prob = np.load(os.path.join(DIST_DIR, f"n{n}_none_MI{mi:d}_MA{ma:d}_BETA_000.000.npy"))
            for i in df_real.loc[df_real.n_notes==n].index:
                print(i) 
                scale = np.cumsum([0] + [int(x) for x in df_real.loc[i, 'pair_ints'].split(';')])
                variable_notes = [range(scale[i]-10, scale[i]+11) for i in range(1, len(scale)-1)]
                df_real.loc[i, f'p_{mi:d}_{ma:d}'] = sum(list(pool.starmap(calculate_base_prob, product([0], *variable_notes, [1200], [prob]))))
    return df_real

def str_to_ints(st, delim=';'):
    return [int(s) for s in st.split(delim) if len(s)]

def ints_to_str(i):
    return ';'.join([str(x) for x in i])

def get_all_ints(df, old='pair_ints', new='all_ints2'):
    def fn(pi):
        ints = np.array(str_to_ints(pi))
        return ints_to_str([x for i in range(len(ints)) for x in np.cumsum(np.roll(ints,i))[:-1]])
    df[new] = df[old].apply(fn)
    return df

def calculate_fifths_bias(df, w=10):
    if 'all_ints2' not in df.columns:
        df = get_all_ints(df)
#   df[f"Nim5_r0.0_w{w}"] = [float(1./(1 + (100 / 3. * 2. * len([z for z in y.split(';') if abs(702-int(z)) <= w])) / len(y.split(';')))) for y in df.all_ints2]
    df[f"Nim5_r0.0_w{w}"] = [float(len([z for z in y.split(';') if abs(702-int(z)) <= w]) / len(y.split(';'))) for y in df.all_ints2]
    return df

def calculate_fifths_bias_all_w(df, w=10):
    if 'all_ints2' not in df.columns:
        df = get_all_ints(df)
#   df[f"Nim5_r0.0_w{w}"] = [float(1./(1 + (100 / 3. * 2. * len([z for z in y.split(';') if abs(702-int(z)) <= w])) / len(y.split(';')))) for y in df.all_ints2]
    for w in [5,10,15,20]:
        df[f"Nim5_r0.0_w{w:02d}"] = [float(len([z for z in y.split(';') if abs(702-int(z)) <= w]) / len(y.split(';'))) for y in df.all_ints2]
    return df

def update_fifths_bias(df_list):
    if isinstance(df_list, list):
        idx = range(len(df_list))
    elif isinstance(df_list, dict):
        idx = df_list.keys()
    for i in idx:
        for w in range(5,25,5):
            df_list[i] = calculate_fifths_bias(df_list[i], w=w)
    return df_list

def old_hss_prob(df_list, beta=1):
    if isinstance(df_list, list):
        idx = range(len(df_list))
    elif isinstance(df_list, dict):
        idx = df_list.keys()
    for i in idx:
        for w in range(5,25,5):
            df_list[i][f"p_har_w{w}"] = np.exp(-beta * 1. / df_list[i][f"hs_n1_w{w}"])
    return df_list


def new_hss_prob(df_list, b, w, beta=1):
    if isinstance(df_list, list):
        idx = range(len(df_list))
    elif isinstance(df_list, dict):
        idx = df_list.keys()
    for i in idx:
        N = df_list[i].loc[:,'n_notes'].values[0]
        min_score, max_score = NORM_CONST.loc[(NORM_CONST.bias==b)&(NORM_CONST.N==N)&(NORM_CONST.W==w),  ['Min', 'Max']].values[0]
        for w in range(5,25,5):
            df_list[i][f"p2_har_w{w}"] = np.exp(-beta * (1.0 - (df_list[i][f"hs_n1_w{w}"] - min_score)/(max_score-min_score)))
    return df_list


def plot_hss_cost_dist(df, prob=True, beta=1):
#   fig, ax = plt.subplots()
#   for n, w in product(range(4,10), range( 5,25,5)):
    for n, w, m in product(range(4,10), range( 5,25,5), range(1,4)):
        min_score = round(df[n][f"hs_n{m}_w{w}"].min(),2)
        max_score = round(df[n][f"hs_n{m}_w{w}"].max(),2)
        print(n, w, m, min_score, max_score)
        if prob:
            sns.distplot(np.exp(- beta *(1 - (df[n][f"hs_n{m}_w{w}"] - min_score) / (max_score - min_score))))
        else:
            sns.distplot(1 - (df[n][f"hs_n{m}_w{w}"] - min_score) / (max_score - min_score))


def plot_fif_cost_dist(df, prob=True, beta=1):
    fig, ax = plt.subplots()
    for n, w in product(range(4,10), range(10,15,5)):
        if prob:
            sns.distplot(np.exp(- beta *(1 - df[n][f"Nim5_r0.0_w{w}"])))
        else:
            sns.distplot(1 - df[n][f"Nim5_r0.0_w{w}"])


def plot_trans_cost_dist(df, prob=True, beta=1):
    fig, ax = plt.subplots()
    for n, w in product(range(4,10), range(1,4)):
        min_score = df[n][f"distI_{w}_0"].min()
        max_score = df[n][f"distI_{w}_0"].max()
        print(n, w, min_score, max_score)
        if prob:
            sns.distplot(np.exp(- beta *(1 - (df[n][f"distI_{w}_0"] - min_score) / (max_score - min_score))))
        else:
            sns.distplot(1 - (df[n][f"distI_{w}_0"] - min_score) / (max_score - min_score))


def plot_transB_cost_dist(df, prob=True, beta=1):
    fig, ax = plt.subplots()
    for n, w in product(range(4,10), range(1,4)):
        min_score = df[n][f"TRANSB_{w}_0"].min()
        max_score = df[n][f"TRANSB_{w}_0"].max()
        print(n, w, min_score, max_score)
        if prob:
            sns.distplot(np.exp(- beta *(1 - (df[n][f"TRANSB_{w}_0"] - min_score) / (max_score - min_score))))
        else:
            sns.distplot(1 - (df[n][f"TRANSB_{w}_0"] - min_score) / (max_score - min_score))


def calc_jensen_shannon_distance(pk, qk):
    mk = 0.5 * (pk + qk) 
    return (0.5 * (calc_relative_entropy(pk, mk) + calc_relative_entropy(qk, mk))) ** 0.5 


def calculate_bias_selectivity(df, df_real, bias, nbin=100, plot=False):
    bin_min = min(df[bias].min(), df_real[bias].min())
    bin_max = max(df[bias].max(), df_real[bias].max())
    bins = np.linspace(bin_min, bin_max, nbin)
    hist1, bins = np.histogram(df[bias], bins=bins, normed=True)
    hist2, bins = np.histogram(df_real[bias], bins=bins, normed=True)
    hist1 /= np.sum(hist1)
    hist2 /= np.sum(hist2)
    if plot:
        plt.plot(bins[:-1], hist1)
        plt.plot(bins[:-1], hist2)
    return calc_jensen_shannon_distance(hist1, hist2)
    

def get_paper_results_paths(df_best, df):
    lbls = ['RAN', 'MIN', 'HAR', 'TRANS', 'FIF']
    idx = [64, 56, 62, 23, 152]
    path = []
    for i in idx:
        mi, ma, bias, beta = df_best.loc[i, ['min_int', 'max_int', 'bias', 'beta']]
        for n in range(4,10):
            path.append(df.loc[(df.min_int==mi)&(df.max_int==ma)&(df.bias==bias)&(df.beta==beta)&(df.n_notes==n), 'fName'].values[0])
    return path


def get_paper_results_paths_2():
    groups = ['FIF', 'HAR3', 'HAR2', 'HAR', 'TRANS', 'MIN', 'RAN']
    lbls = ['FIF', r'$\text{HAR}^{3}$', r'$\text{HAR}^{2}$', r'$\text{HAR}$', 'TRANS', 'MIN', 'RAN']

    for i, bg in enumerate(groups):
        ax.scatter(df1.loc[(df1.bias_group==bg)&(df1.method=='best'), 'met1']*1000., df1.loc[(df1.bias_group==bg)&(df1.method=='best'), 'fr_10'], color=col_s[i],  s=60, edgecolor='k', alpha=0.7, label=lbls[i])
    for i, bg in enumerate(groups):
        ax.scatter(df2.loc[(df2.bias_group==bg)&(df1.method=='best'), 'met1']*1000., df2.loc[(df2.bias_group==bg)&(df1.method=='best'), 'fr_10'], color=col_s[i],  s=60, edgecolor='k', alpha=0.7)


def calculate_goodness_of_fit(paths, df_real, X='pair_ints'):
    idx = [64, 56, 62, 23, 152]
    labels = ['RAN', 'MIN', 'HAR', 'TRANS', 'FIF']

    out_df = pd.DataFrame(columns=['bias', 'N', 'S', 'r2', 'RMSD', 'd_RMSD', 'met1'])

    for i, n in enumerate(range(4,10)):
        if os.path.exists(os.path.join(STATS_PATH, f"real_{n}")):
            X1, Y1 = np.load(os.path.join(STATS_PATH, f"real_{n}"))
        else:
            X1, Y1 = smooth_dist_kde(df_real.loc[df_real.n_notes==n])
            np.save(os.path.join(STATS_PATH, f"real_{n}"), np.array([X1, Y1]))
        S = len(df_real.loc[df_real.n_notes==n])
    
        for j, l in enumerate(labels):
            if os.path.exists(os.path.join(STATS_PATH, f"{labels[j]}_{n}")):
                X2, Y2 = np.load(os.path.join(STATS_PATH, f"{labels[j]}_{n}"))
            else:
                X2, Y2 = smooth_dist_kde(pd.read_feather(paths[l][n]))
                np.save(os.path.join(STATS_PATH, f"{labels[j]}_{n}"), np.array([X2, Y2]))

            SStot = np.sum((Y1 - np.mean(Y1))**2)
            SSres = np.sum((Y1 - Y2)**2)
            r2 = 1 - SSres/SStot
            RMSD = np.sqrt(np.dot(Y1-Y2, Y1-Y2))
            D1 = Y1[1:] - Y1[:-1]
            D2 = Y2[1:] - Y2[:-1]
            deriv_RMSD = np.sqrt(np.dot(D1-D2, D1-D2))
            met1 = (RMSD * deriv_RMSD)**0.5
            out_df.loc[len(out_df)] = [labels[j], n, S, r2, RMSD, deriv_RMSD, met1]
    return out_df

def average_goodness_of_fit(df, X='r2', ave='euc'):
    labels = ['RAN', 'MIN', 'HAR', 'TRANS', 'FIF']
    for i, l in enumerate(labels):
        if ave=='euc':
            score = np.sum([df.loc[j, X] * df.loc[j, 'S'] / df.loc[df.bias==l, 'S'].sum() for j in df.loc[df.bias==l].index])
        elif ave=='geo':
            score = np.product([df.loc[j, X] ** (df.loc[j, 'S'] / df.loc[df.bias==l, 'S'].sum()) for j in df.loc[df.bias==l].index])
        yield score

def get_primes(n):
    numbers = set(range(n, 1, -1))
    primes = []
    while numbers:
        p = numbers.pop()
        primes.append(p)
        numbers.difference_update(set(range(p*2, n+1, p)))
    return primes


def harmonic_series_intervals(n1=10, n2=10, att=0.8):
    primes = get_primes(max(n1,n2))
    ratios = []
    weights = []
    for i in range(1,n1):
        for j in range(i,n2):
            top, bottom = j, i
            ratio = top / bottom
            while ratio > 2.0:
                bottom *= 2
                ratio = top / bottom

            for p in primes:
                while True:
                    if (int(top/p) == top/p) and (int(bottom/p) == bottom/p):
                        top /= p
                        bottom /= p
                    else:
                        break
            ratios.append(f"{int(top)}/{int(bottom)}")
            weights.append(att**(i-1) * att**(j-1))

    count = {r:0 for r in list(set(ratios))}
    for r, w in zip(ratios, weights):
        count[r] += w
    count = sorted(count.items(), key=lambda kv :kv[1])[::-1]
    return count


def reformat_old_df(df):
#   df = df.copy()
    for w in range(5,25,5):
        df.loc[df.bias==f"Nim5_r0.0_w{w:02d}", 'bias_group'] = "FIF"
        df.loc[df.bias==f"Nim5_r0.0_w{w:02d}", 'bias'] = f"FIF_{w}"
        df.loc[df.bias==f"Nhs_n1_w{w:02d}", 'bias_group'] = "HAR"
        df.loc[df.bias==f"Nhs_n2_w{w:02d}", 'bias_group'] = "HAR2"
        df.loc[df.bias==f"Nhs_n3_w{w:02d}", 'bias_group'] = "HAR3"
        df.loc[df.bias==f"hs_n1_w{w:02d}", 'bias_group'] = "HAR"
        df.loc[df.bias==f"hs_n2_w{w:02d}", 'bias_group'] = "HAR2"
        df.loc[df.bias==f"hs_n3_w{w:02d}", 'bias_group'] = "HAR3"
        for i in range(1,4):
           df.loc[df.bias==f"Nhs_n{i}_w{w:02d}", 'bias'] = f"HAR_{w}_{i}"
           df.loc[df.bias==f"hs_n{i}_w{w:02d}", 'bias'] = f"HAR_{w}_{i}"

    df.loc[(df.bias=="none")&(df.min_int==0)&(df.max_int==1200), 'bias_group'] = "RAN"
    df.loc[(df.bias=="none")&(df.min_int==0)&(df.max_int==1200), 'bias'] = "RAN"
    df.loc[(df.bias=="none")&(df.min_int!=0)&(df.max_int==1200), 'bias_group'] = "MIN"
    df.loc[(df.bias=="none")&(df.min_int!=0)&(df.max_int==1200), 'bias'] = "MIN"

    for i in range(1,4):
       df.loc[df.bias==f"distI_n{i}", 'bias_group'] = f"TRANS"
       df.loc[df.bias==f"distI_n{i}", 'bias'] = f"TRANS_{i}"

    groups = ['FIF', 'HAR', 'HAR2', 'HAR3', 'RAN', 'MIN', 'TRANS']
    return df.drop(index=df.loc[df.bias_group.apply(lambda x: x not in groups)].index).reset_index(drop=True)
        

def normalise_metrics(df, X='met1', Y='fr_10'):
    for b in df.bias.unique():
        for mi in df.min_int.unique():
            for n in range(4,10):
                idx = df.loc[(df.n_notes==n)&(df.bias==b)&(df.min_int==mi)&(df.max_int==1200)].index
                df.loc[idx, 'X'] = np.abs(df.loc[idx,X].min() - df.loc[idx,X]) / df.loc[idx,X].min()
                if df.loc[idx,Y].max() == 0:
                    df.loc[idx, 'Y'] = 0
                else:
                    df.loc[idx, 'Y'] = np.abs(df.loc[idx,Y].max() - df.loc[idx,Y]) / df.loc[idx,Y].max()
                df.loc[idx, 'Z'] = df.loc[idx, 'X'] + df.loc[idx, 'Y']
    return df


def normalise_metrics_2(df, X='met1', Y='fr_10'):
    for b in df.bias_group.unique():
        idx = df.loc[(df.bias_group==b)].index
        df.loc[idx, 'X'] = np.abs(df.loc[idx,X].min() - df.loc[idx,X]) / df.loc[idx,X].min()
        df.loc[idx, 'Y'] = np.abs(df.loc[idx,Y].max() - df.loc[idx,Y]) / df.loc[idx,Y].max()
        df.loc[idx, 'Z'] = df.loc[idx, 'X'] + df.loc[idx, 'Y']
    return df


def find_best_beta(df, f_real):
    X, Y, Z, Q = [], [], [], []
    beta = df.beta.unique()
    print(df)
    for b in beta:
        print(1, b)
        try:
            idx = [df.loc[(df.beta==b)&(df.n_notes==n)].index[0] for n in range(4,10)]
        except IndexError:
            continue
        met1, fr10, Za, q = df.loc[idx, ['met1', 'fr_10', 'Z', 'logq']].values.T
        X.append(np.product(met1 ** f_real))
        Y.append(np.sum(fr10 * f_real))
        Z.append(np.sum(Za * f_real))
        Q.append(np.sum(q * f_real))
        print(2, b)
    idx = np.argmin(Z)
    return [Q[idx], X[idx], Y[idx], beta[idx]]


def pick_best_models(df, df_real, plot=False):
    biases = df.bias.unique()
    n_real = np.array([len(df_real.loc[df_real.n_notes==n]) for n in range(4,10)])
    f_real = n_real / n_real.sum()
    cols = ['min_int', 'bias', 'bias_group', 'beta_mean', 'beta_std', 'logq_mean', 'met1', 'fr_10', 'method']
    col1 = RdYlGn_11.hex_colors
    col2 = Paired_12.mpl_colors
    col3 = RdYlGn_6.hex_colors
    col = {'HAR':col2[3], 'HS':col2[3], 'FIF':col1[0], 'im5':col1[0], 'none':col2[7],
           'HAR2':col1[6], 'HAR3':col1[4], 'MIN':col3[1], 'RAN':'k', 'TRANS':col2[1], 'distI':col2[1], 'TRANSB':'purple'}
    df_best = pd.DataFrame(columns=cols)
    if plot:
        fig, ax = plt.subplots()
    for b in biases:
        for mi in [0, 70, 80, 90]:
            try:
                idx = [df.loc[(df.n_notes==n)&(df.bias==b)&(df.min_int==mi)&(df.max_int==1200), 'Z'].idxmin() for n in range(4,10)]
            except ValueError:
                continue
            out1 = [np.product(df.loc[idx, 'met1'].values ** f_real)]
            out2 = [np.sum(x * f_real) for x in df.loc[idx, ['fr_10', 'logq']].values.T]

            bg = df.loc[idx[0], 'bias_group']

            df_best.loc[len(df_best)] = [mi, b, bg] + [df.loc[idx, 'beta'].mean(), df.loc[idx, 'beta'].std()] + \
                                        [out2[1]] + out1 + [out2[0]] + ['best']

#           out3 = find_best_beta(df.loc[(df.bias==b)&(df.min_int==mi)&(df.max_int==1200)], f_real)
#           df_best.loc[len(df_best)] = [mi, b, bg] + [out3[3], 0] + out3[:3] + ['bias']

            if plot:
                plt.plot(out1, [out2[0]], 'o', color=col[bg], fillstyle='none')
                plt.plot([out3[1]], [out3[2]], 'o', color=col[bg])
                plt.plot(out1+[out3[1]], [out2[0]]+[out3[2]], '-', color=col[bg])

    df_best = normalise_metrics_2(df_best)

    PATHS = {}
    for bg in df_best.bias_group.unique():
        if bg == 'HAR':
            mi, b, met1, fr_10 = df_best.loc[df_best.loc[df_best.bias=="HAR_10_1", 'Z'].idxmin(), ['min_int', 'bias', 'met1', 'fr_10']].values
        else:
            mi, b, met1, fr_10 = df_best.loc[df_best.loc[df_best.bias_group==bg, 'Z'].idxmin(), ['min_int', 'bias', 'met1', 'fr_10']].values
        mi = 80
        if bg == 'RAN':
            idx = [df.loc[(df.n_notes==n)&(df.bias==b)&(df.min_int==0)&(df.max_int==1200), 'Z'].idxmin() for n in range(4,10)]
        else:
            idx = [df.loc[(df.n_notes==n)&(df.bias==b)&(df.min_int==mi)&(df.max_int==1200), 'Z'].idxmin() for n in range(4,10)]
        PATHS.update({bg:{**{n:df.loc[i, 'fName'] for i, n in zip(idx, range(4,10))},
                      **{'met1':{**{n:df.loc[i, 'met1'] for i, n in zip(idx, range(4,10))}, **{0:met1}}},
                      **{'fr_10':{**{n:df.loc[i, 'fr_10'] for i, n in zip(idx, range(4,10))}, **{0:fr_10}}}}})

    return df_best, PATHS

def amalgamate_paths(paths1, paths3):
    paths = paths3.copy()
    paths['TRANS'] = paths1['TRANS']
    paths['MIN'] = paths1['MIN']
    paths['FIF'] = paths1['FIF']
    paths['RAN'] = paths1['RAN']
#   paths['RAN'] = {n:v for n, v in zip(range(4,10), df.loc[(df.bias=="none")&(df.min_int==0)&(df.max_int==1200), 'fName'].values)}
#   paths['TRANS'] = {n:df.loc[(df.n_notes==n)&(df.bias=="distI_n2")&(df.min_int==80)&(df.max_int==1200)].sort_values(by='Z')['fName'].values[0] for n in range(4,10)}
    return paths


def fifths_bias_old(all_ints2, w=10):
    return 1.0 / (1.0 + np.mean([66.7 if abs(702-int(z)) <= w else 0. for z in all_ints2.split(';')]))


def probability_of_finding_scales(paths, df):
    if os.path.exists(os.path.join(DIST_DIR, 'MIN80.npy')):
        min_prob = np.load(os.path.join(DIST_DIR, 'MIN80.npy'))
    else:
        bins = np.linspace(0, 1200, num=1201)
        files = [f"/home/johnmcbride/projects/Scales/Toy_model/Data/Processed3/n{n}_none_MI80_MA1200_BETA_001.000.feather" for n in range(4,10)]
        min_prob = np.array([np.histogram(extract_floats_from_string(pd.read_feather(f).pair_ints), bins=bins, normed=True)[0] for f in files])
        np.save(os.path.join(DIST_DIR, 'MIN80.npy'), min_prob)

    models = ["TRANS", "HAR", "FIF"]

    if os.path.exists(os.path.join(SRC_DIR, 'n_trials.npy')):
        n_trials = np.load(os.path.join(SRC_DIR, 'n_trials.npy'))
    else:
        n_trials = np.array([[pd.read_feather(paths[m][n])["n_att"].sum() for n in range(4,10)] for m in models])
        np.save(os.path.join(SRC_DIR, 'n_trials.npy'), n_trials)

    beta = {k:[float(v[n].split('_')[-1].strip('.feather')) for n in range(4,10)] for k, v in paths.items()}
    scores = {"FIF":"Nim5_r0.0_w10", "HAR":"hs_n1_w10", "TRANS":"distI_2_0"}
    min_har = [10, 11, 13, 13, 14, 14]
    max_har = [42.69, 37.54, 35.3, 32.23, 30.86, 28.91]

    for i in df.loc[(df.n_notes>3)&(df.n_notes<10)].index:
        n = df.loc[i, 'n_notes'] - 4
#       pmin = np.product([min_prob[n, k] for k in str_to_ints(df.loc[i, 'pair_ints'])])
        pmin = df.loc[i, "p_80_1200"]

        df.loc[i, "pMIN"] = pmin
        df.loc[i, f'pTRANS'] = pmin * n_trials[0][n] * np.exp( - beta["TRANS"][n] * df.loc[i, scores["TRANS"]]) 
        df.loc[i, f'pHAR'] = pmin * n_trials[1][n] * np.exp( - beta["HAR"][n] * (1.0 - ((df.loc[i, scores["HAR"]] - min_har[n])/(max_har[n]-min_har[n]))**2))
        df.loc[i, f'pFIF'] = pmin * n_trials[2][n] * np.exp( - beta["FIF"][n] * fifths_bias_old(df.loc[i, 'all_ints2']))
        df.loc[i, 'pALL'] = np.sum(df.loc[i, ['pTRANS', 'pFIF', 'pHAR']])
#       print(i, n_trials[2][n], beta["TRANS"][n], fifths_bias_old(df.loc[i, 'all_ints2']), df.loc[i, f'pFIF'])

    return df

    
def round_sf(number, significant):
    if number > 1:
        return round(number, significant-1)
    else:
        return round(number*10**-int(np.log10(number)), significant)/10**-int(np.log10(number))


def check_bin_size_effect(paths, df_real, X='scale'):

    labels = ['MIN','TRANS', 'HAR', 'FIF']
    num = np.arange(10, 110, 10)
    bin_list = [np.arange(10, 1200, n) for n in num]

    out = np.zeros((2,len(labels),num.size), dtype=float)

    for i, n in enumerate([5,7]):
        n_data = len(df_real.loc[df_real.n_notes==n, X])
        for j, bins in enumerate(bin_list):
            histD, bins = np.histogram(extract_floats_from_string(df_real.loc[df_real.n_notes==n, X]), bins=bins, normed=True)
            for count, lbl in enumerate(labels):
                xxx = bins[:-1] + 0.5 * (bins[1] - bins[0])
                histM, bins = np.histogram(extract_floats_from_string(pd.read_feather(paths[lbl][n])[X]), bins=bins, normed=True)
                df_hist = pd.DataFrame(data={'bin':xxx, lbl:histM})
                
                SStot = np.sum((histD - np.mean(histD))**2)
                SSres = np.sum((histD - histM)**2)
                r2 = 1 - SSres/SStot

                out[i,count,j] = r2
#               print(f"N={n}\tnum={num[j]}\tmodel={j+1}\nfit={r2}\n")
#   return out

    fig, ax = plt.subplots(2,1)
    for i in range(4):
        ax[0].plot(num, out[0,i]/out[0].max(axis=0), label=labels[i])
        ax[1].plot(num, out[1,i]/out[1].max(axis=0), label=labels[i])
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')


def attenuated_hss_score(i, j, attenuate, n=100):
    score = 0.0
    for n1 in np.arange(1.,n+1):
        for n2 in np.arange(1.,n+1):
            if float(n2*j/i) == n1:
                score += 1.0*(attenuate**(n1-1)) * 1.0*(attenuate**(n2-1))
            elif float(n2*j/i)>n1:
                break
    return score

def ratio_scores(attenuate=0.9):
    scores = []
    cents = []
    for i in np.arange(1.,202.):
        for j in np.arange(i, 400.):
            scores.append(attenuated_hss_score(i,j, attenuate))
            cents.append(get_cents_from_ratio(float(j/i)))
    return np.array(cents), np.array(scores)


def modified_hss(attenuate):
    X = np.arange(0,1201, dtype=float)
    Y = []
    cents, scores = ratio_scores(attenuate=attenuate)
    for i, x in enumerate(X):
        Y.append(max(scores[np.where(np.abs(cents-x)<=CENT_DIFF_MAX)[0]]))
    return np.array(Y)


def modified_hss_by_attenuation():
    X = np.arange(0,1201, dtype=float)
    with mp.Pool(28) as pool:
        mhar = np.array(list(pool.map(modified_hss, np.arange(0.01, 1.01, 0.01))))
    return X, np.array(mhar)


def plot_distributions_by_fifths(X, Y1, Y2):
    fig, ax = plt.subplots()
    plt.plot(X, Y1/Y1[np.where(np.abs(X-702)<10)[0]].max())
    plt.plot(X, Y2/Y2[np.where(np.abs(X-702)<10)[0]].max(), 'o')
    plt.ylim(0, 1.1)


def match_attenuation(df, mhar):
    X = np.arange(0,1201, dtype=float)
    att = np.arange(0.01, 1.01, 0.01)
    har = np.copy(mhar)
    for i in range(har.shape[0]):
        har[i] /= har[i][np.where(np.abs(X-702)<10)[0]].max()

    har1 = np.copy(har[-1])
    har2 = np.copy(har[-1]*10)**2
    har3 = np.copy(har[-1]*10)**3
    har4 = np.zeros(1201)
    har4[np.abs(X-702)<=20] = 1

    match = []
    r_vals = []

    ints = [str_to_ints(x) for x in df.all_ints2]

    for h1 in [har1, har2, har3, har4]:
        h1 /= h1[np.where(np.abs(X-702)<10)[0]].max()
        score1 = [np.mean([h1[i] for j in i]) for i in ints]
        r_vals.append([pearsonr(score1, [np.mean([h2[i] for j in i]) for i in ints])[0] for h2 in har])
        match.append(att[np.argmax(r_vals[-1])])
#       r = [np.abs(np.sum(h1[20:-20] - h2[20:-20])) for h2 in har]
#       match.append(att[np.argmin(r)])
#       print(r)
#       print(np.argmax(r))

    return match, r_vals


def return_beta_below_optimum(df, bias, n):
    idx = []
    for mi, b in product([70,80,90], bias):
        idx1 = df.loc[(df.n_notes==n)&(df.bias==b)&(df.min_int==mi)&(df.max_int==1200)&(df.mfr_10.notnull())].index
        if len(idx1):
            betamax = df.loc[df.loc[idx1, 'Z'].idxmin(), 'beta']
            idx += list(df.loc[([True if i in idx1 else False for i in df.index])&(df.beta<=betamax)].index)
    return df.loc[idx].reset_index(drop=True)

def save_min_model_costs(df_min):
    biases = ['hs_n1_w10', 'distI_2_0', 'Nim5_r0.0_w10']
    labels = ['HAR', 'TRANS', 'FIF']
    bins = [np.linspace(0, x, 101) for x in [50, 0.1, 0.25]]
    for n in range(4,10):
        for b, l, bn in zip(biases, labels, bins):
            hist, bn = np.histogram(df_min[n][b], bins=bn)
            hist = hist / hist.sum()
            X = bn[:-1] + 0.5 * (bn[1] - bn[0])
            np.save(BASE_DIR + f"/Processed/Cleaned/MIN_cost_func/MIN_{n}_{l}_stats", np.array([X, hist]))




