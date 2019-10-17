import argparse
import glob
import os
import sys
import time

from itertools import product, permutations
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.nonparametric.api as smnp

import swifter

N_PROC = 1
CHUNK  = 25

MIX = False

BASE_DIR = '/home/jmcbride/Scales/Compared_data'
RAW_DIR  = '/home/jmcbride/Scales/Toy_model/Data/Raw/'
PRO_DIR  = '/home/jmcbride/Scales/Toy_model/Data/Processed/'
DIST_DIR  = '/home/jmcbride/Scales/Toy_model/Data/None_dist/'
REAL_DIR = '/home/jmcbride/Scales/Real_scales'

TEMP_MIN = 50.
TEMP_MAX = 300.
TEMP_LOW_MARGIN = 0.50
TEMP_HI_MARGIN = 1.50
N_TRIALS = 50

ALPHA_W = 0.1


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--partabase', action='store', default='None', type=str)
    parser.add_argument('-f', action='store', default='None', dest='fName', type=str)
    parser.add_argument('--sample', action='store_true', default=False, dest='sample',)
    return parser.parse_args()

args = parse_arguments()



def get_scale_from_pair_ints(pair_ints):
    ints = [int(y) for y in pair_ints.split(';')]
    return ';'.join(['0'] + [str(y) for y in np.cumsum(ints)])


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
            simils.append(max_similarity**n / 100.**(n-1))
    return sc_i, np.array(attract), ratios, simils


def get_similarity_of_nearest_attractor(x, sc_f, simil):
    minIdx = np.argmin(np.abs(sc_f - x)) 
    return simil[minIdx]


def get_harmonic_similarity_score_series(series, diff, n):
    sc_i, sc_f, ratios, simil  = get_attractors(n, diff=diff)
    return series.swifter.apply(lambda x: np.mean([get_similarity_of_nearest_attractor(float(y), sc_f, simil) for y in x.split(';')]))


def calculate_optimum_window_size(df):
    def fn(x):
        w_arr = np.arange(15,61)
        d_arr = []
        n_arr = []
        for w in w_arr:
            dists =  calculate_distance_between_windows(x, w)
            d_arr.append(min([int(y) for y in dists.split(';')]) if len(dists) else 0)
            n_arr.append(len(dists.split(';'))+1 if len(dists) else 1)
        d_arr = np.array(d_arr)
        cost = np.zeros(len(w_arr), dtype=float)
        idx = np.where(d_arr)[0]
        cost[idx] = w_arr[idx] / d_arr[idx]
#       idx = [i for i in range(cost.size) if 0< d_arr[i] < 20]
#       if len(idx):
#           cost[idx] = cost[idx] + 10.
        idxMin = np.argmin(cost)
        return ';'.join([str(y) for y in [round(cost[idxMin],3)+ALPHA_W, w_arr[idxMin], d_arr[idxMin], n_arr[idxMin]]])
    df['tmp'] = df.pair_ints.swifter.apply(fn)
    df['opt_c'] = df.tmp.swifter.apply(lambda x: float(x.split(';')[0]))
    df['opt_w'] = df.tmp.swifter.apply(lambda x: int(x.split(';')[1]))
    df['opt_d'] = df.tmp.swifter.apply(lambda x: int(x.split(';')[2]))
    df['opt_n'] = df.tmp.swifter.apply(lambda x: int(x.split(';')[3]))
    return df.drop(columns=['tmp'])


def calculate_highest_minimum(df):
    cols = [c for c in df.columns if len(c.split('_')) == 3 and c[-3:] == 'min']
    df['best_sep'] = df.swifter.apply(lambda x: max([x[c] for c in cols]))
    return df


def calculate_distance_between_windows(x, w): 
    ints = sorted([int(y) for y in x.split(';')])
    windows = [[ints[0]]]
    for i in ints[1:]:
        if i - windows[-1][0] < w:
            windows[-1].append(i)
        else:
            windows.append([i])
    if len(windows) == 1:
        return ''
    else:
        dist = [windows[i+1][0] - windows[i][-1] for i in range(len(windows)-1)]
        return ';'.join([str(d) for d in dist]) if len(dist) > 1 else str(dist[0])


def get_distance_between_windows(df, w, X='pair_ints'):
    df[f"d_w{w}"] = df.loc[:,X].swifter.apply(lambda x: calculate_distance_between_windows(x, w))
    df[f"d_w{w}_min"] = df.loc[:,f"d_w{w}"].swifter.apply(lambda x: min([int(y) for y in x.split(';')]) if len(x) else 0)
    df[f"d_w{w}_mean"] = df.loc[:,f"d_w{w}"].swifter.apply(lambda x: np.mean([int(y) for y in x.split(';')]) if len(x) else 0)
    return df


def calc_relative_entropy(pk, qk):
    RE = 0.0
    for i in range(len(pk)):
        if pk[i] <= 0 or qk[i] <= 0:
            pass
        else:
            RE += pk[i] * np.log2(pk[i] / qk[i])
    return RE


def calc_jensen_shannon_distance(pk, qk):
    mk = 0.5 * (pk + qk)
    return (0.5 * (calc_relative_entropy(pk, mk) + calc_relative_entropy(qk, mk))) ** 0.5


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
    grid, y = kde.support, kde.density
    if hist:    
        hist, edges = np.histogram(X, bins=grid, normed=True)
        xxx = grid[:-1] + (grid[1] - grid[0]) * 0.5    
        return grid, y, xxx, hist
    else:
        return grid, y


def get_KDE(df, cat):
    xKDE, yKDE = smooth_dist_kde(df, cat=cat)
    xKDE, yKDE = convert_grid(xKDE, yKDE)
    return yKDE / np.trapz(yKDE)


def get_real_scales_dists(n, df_real):
    fHist = os.path.join(REAL_DIR, f"n_{n}_hist.npy")
    fKDE  = os.path.join(REAL_DIR, f"n_{n}_kde.npy")
    if os.path.exists(fHist):
        data = np.load(fHist)
        xHist, yHist = data[:,0], data[:,1]
        data = np.load(fKDE)
        new_grid, new_y = data[:,0], data[:,1]
    else:
        xKDE, yKDE, xHist, yHist = smooth_dist_kde(df_real.loc[df_real.n_notes==n], cat='pair_ints', hist=True)
        new_grid, new_y = convert_grid(xKDE, yKDE)
        np.save(fHist, np.array([xHist, yHist]).T)
        np.save(fKDE, np.array([new_grid, new_y]).T)
    return new_grid, new_y, xHist, yHist 


def calculate_energy_from_intervals(ints, base, m, n): 
    return np.mean([abs(round(i/base) - i/base)**m * float(round(i/base))**n for i in ints])


def template_function(ints, m, n): 
    ints = [float(x) for x in ints.split(';')]
    temp_min = max(TEMP_MIN, min(ints)*TEMP_LOW_MARGIN)
    temp_max = min(TEMP_MAX, min(ints)*TEMP_HI_MARGIN)
    baseArr = np.linspace(temp_min, temp_max, num=N_TRIALS)
    energies = np.zeros(baseArr.size, dtype=float)
    for i, base in enumerate(baseArr):
        energies[i] = calculate_energy_from_intervals(ints, base, m, n)
    if len(np.where(energies==0)[0]) > 1:
        idxMin = np.where(energies==0)[0][-1]
    else:
        idxMin = np.argmin(energies)
    return energies[idxMin]


def test_distinguishability_integer_multiples(df):
    for n in range(3):
        for m in range(3):
            if N_PROC > 1:
                pool = mp.Pool(N_PROC)
                df[f"distI_{m}_{n}"] = pool.starmap(template_function, product(df.pair_ints, [m], [n]))
                pool.close()
            else:
                df[f"distI_{m}_{n}"] = df.pair_ints.swifter.apply(lambda x: template_function(x, m, n))
    return df


def test_distinguishability_window_distance(df):
    return calculate_optimum_window_size(df)


def test_harmonic_series_similarity(df, n):
    for i in range(5,25,5):
        df[f"hs_n{n}_w{i:02d}"] = get_harmonic_similarity_score_series(df.all_ints2, i, n)
    return df


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


def calculate_fifths_bias(df):
    for w in [5,10,15,20]:
        df[f"Nim5_r0.0_w{w}"] = [float(len([z for z in y.split(';') if abs(702-int(z)) <= w]) / len(y.split(';'))) for y in df.all_ints2]
    return df


def process_df(df, grid):
    timeS = time.time()
    if grid and MIX:
#       df['scale'] = df.scale.swifter.apply(lambda x: '0;' + x)
        if N_PROC > 1:
            pool = mp.Pool(N_PROC)
            df['mix_ints'] = pool.map(choose_permutation, df.pair_ints, CHUNK)
            df['mix_scale'] = pool.map(get_scale_from_pair_ints, df.mix_ints, CHUNK)
            pool.close()
        else:
            df['mix_ints'] = df.pair_ints.swifter.apply(choose_permutation)
            df['mix_scale'] = df.mix_ints.swifter.apply(get_scale_from_pair_ints)
#   print(f"Mixing: {(time.time()-timeS)/60.} minutes")
    df['min_int'] = df.pair_ints.apply(lambda x: min([int(y) for y in x.split(';')]))
    df = df.drop(index=df.loc[df.min_int==0].index).reset_index(drop=True)
    df['max_int'] = df.pair_ints.apply(lambda x: max([int(y) for y in x.split(';')]))
    print(f"Min/max: {(time.time()-timeS)/60.} minutes")
    df = get_all_ints(df)
    print(f"All_ints2: {(time.time()-timeS)/60.} minutes")
    df = test_distinguishability_integer_multiples(df)
    print(f"DistI: {(time.time()-timeS)/60.} minutes")
    df = calculate_fifths_bias(df)
    print(f"Fifths score: {(time.time()-timeS)/60.} minutes")
#   df = test_distinguishability_window_distance(df)
#   print(f"DistW: {(time.time()-timeS)/60.} minutes")
#   df['opt_c_I1'] = df.opt_c * df.distI_0_1
#   df['opt_c_I2'] = df.opt_c * df.distI_0_2
#   print(f"DistW_S1: {(time.time()-timeS)/60.} minutes")
#   def small_int_bias(x, n):
#       ints = np.array([int(y) for y in x.split(';')])
#       return np.sum(ints**n) / 1200.**n
#   df['opt_c_s2'] = df.opt_c * df.pair_ints.swifter.apply(lambda x: small_int_bias(x, 2))
#   df['opt_c_s3'] = df.opt_c * df.pair_ints.swifter.apply(lambda x: small_int_bias(x, 3))
#   print(f"DistW_S2: {(time.time()-timeS)/60.} minutes")
    df = test_harmonic_series_similarity(df, 1)
    df = test_harmonic_series_similarity(df, 2)
    df = test_harmonic_series_similarity(df, 3)
    print(f"HS: {(time.time()-timeS)/60.} minutes")
    return df

def ss_fn(x, df_real, idx, w):
    return ';'.join([str(i) for i in idx if is_scale_similar(x, df_real.loc[i, 'scale'], w)])

def process_grid_similar_scales(df_grid, df_real, n):
    timeS = time.time()
    if args.sample:
        samples = ['theory', 'instrument'] + [f"sample_f{frac:3.1f}_{i:02d}" for frac in [0.4, 0.6, 0.8] for i in range(10)]
        for w in [10, 20]:
            for i, s in enumerate(samples):
                idx = df_real[i].loc[df_real[i].n_notes==n].index
                if N_PROC > 1:
                    pool = mp.Pool(N_PROC)
                    df_grid[f'{s}_ss_w{w:02d}'] = pool.starmap(ss_fn, product(df_grid.scale, [df_real[i]], [idx], [w]))
                    pool.close()
                else:
                    df_grid[f'{s}_ss_w{w:02d}'] = df_grid.scale.apply(lambda x: ss_fn(x, df_real[i], idx, w))
                print(f"ss_w{w:02d}: {(time.time()-timeS)/60.} minutes")
    else:
        idx = df_real.loc[df_real.n_notes==n].index
        for w in [10, 20]:
            if N_PROC > 1:
                pool = mp.Pool(N_PROC)
                df_grid[f'ss_w{w:02d}'] = pool.starmap(ss_fn, product(df_grid.scale, [df_real], [idx], [w]))
                if MIX:
                    df_grid[f'mss_w{w:02d}'] = pool.starmap(ss_fn, product(df_grid.mix_scale, [df_real], [idx], [w]))
                pool.close()
            else:
                df_grid[f'ss_w{w:02d}'] = df_grid.scale.swifter.apply(lambda x: ss_fn(x, df_real, idx, w))
                if MIX:
                    df_grid[f'mss_w{w:02d}'] = df_grid.mix_scale.swifter.apply(lambda x: ss_fn(x, df_real, idx, w))
            print(f"ss_w{w:02d}: {(time.time()-timeS)/60.} minutes")
    return df_grid

def is_scale_similar(x, y, w):
    xint = [int(a) for a in x.split(';')]
    yint = [int(a) for a in y.split(';')]
    return np.allclose(xint, yint, atol=w)

def how_much_real_scales_predicted(df, n_real, w):
    return float(len(set([int(x) for y in df[f"ss_w{w:02d}"] for x in y.split(';') if len(y)]))) / float(n_real)

def mixing_cost_arr(arr):
    return np.array([np.mean([np.abs(ints[(i-1)] + ints[i%len(ints)] - 2400./float(len(ints)))**2 for i in range(1,len(ints)+1)])**0.5 for ints in arr])

def get_probability_from_costs(costs):
    return np.array([np.exp(1./c) / np.sum(np.exp(1./costs)) for c in costs])

def permute_scale(int_str):
    ints = np.array([int(x) for x in int_str.split(';')])
    return np.array(list(set(permutations(ints))))

def choose_permutation(int_str):
    perm  = permute_scale(int_str)
    costs = mixing_cost_arr(perm)
    np.random.seed()
    if np.any(costs==0):
        return ';'.join([str(int(round(x))) for x in perm[np.random.randint(len(perm))]])
    else:
        prob  = get_probability_from_costs(costs/costs.max())
        ran = np.random.rand()
        cumprob = np.cumsum(prob)
        return ';'.join([str(int(round(x))) for x in perm[np.where(cumprob>ran)[0][0]]])

def get_metrics(grid, y1, y2):
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

if __name__ == "__main__":

    categories = ['pair_ints']
    n = int(args.fName.split('_')[0].strip('n'))
    n_arr = np.array([n])

    if args.sample:
        df_real = [pd.read_feather(os.path.join(REAL_DIR, 'Samples', f"{f}.feather")) for f in ['theory', 'instrument'] + \
                                   [f"sample_f{frac:3.1f}_{i:02d}" for frac in [0.4, 0.6, 0.8] for i in range(10)]]
    else:
        if os.path.exists(os.path.join(REAL_DIR, 'theories_real_scales.feather')):
            df_real = pd.read_feather(os.path.join(REAL_DIR, 'theories_real_scales.feather'))
        else:
            df_real = pd.read_feather(os.path.join(REAL_DIR, 'real_scales.feather'))
            df_real = process_df(df_real, 0)
            df_real.to_feather(os.path.join(REAL_DIR, 'theories_real_scales.feather'))


    def read_model_results(path):
        print(path)
        fName = os.path.split(path)[1]
        if args.sample:
            pro_name = os.path.join(PRO_DIR, 'sample_'+fName)
        else:
            pro_name = os.path.join(PRO_DIR, fName)
        tmp_df = pd.read_feather(path)
        tmp_df['beta'] = float(fName.split('_')[-1].strip('.feather'))
        tmp_df = process_df(tmp_df, 1)
        n = int(fName.split('_')[0].strip('n'))
        tmp_df = process_grid_similar_scales(tmp_df, df_real, n)
        tmp_df.to_feather(pro_name)
        return tmp_df


    df_grid = read_model_results(os.path.join(RAW_DIR, args.fName))
#   df_grid = read_model_results(os.path.join(PRO_DIR, args.fName))

