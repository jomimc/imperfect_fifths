import argparse
import glob
import os
import sys
import time

from itertools import product
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.nonparametric.api as smnp
import swifter

import utils
import graphs

N_PROC = 28

BASE_DIR = '/home/johnmcbride/projects/Scales/Data_compare/Processed/'
RAW_DIR  = '/home/johnmcbride/projects/Scales/Toy_model/Data/Raw/'
PRO_DIR  = '/home/johnmcbride/projects/Scales/Toy_model/Data/Processed/'
DIST_DIR  = '/home/johnmcbride/projects/Scales/Toy_model/Data/None_dist/'
REAL_DIR = os.path.join(BASE_DIR, 'Real')

#BASE_DIR = '/home/johnmcbride/projects/Scales/Data_compare/TMP/'
#RAW_DIR  = '/home/johnmcbride/projects/Scales/Data_compare/TMP/Raw/'
#PRO_DIR  = '/home/johnmcbride/projects/Scales/Data_compare/TMP/Processed/'

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
        return ';'.join([str(y) for y in [round(cost[idxMin],3), w_arr[idxMin], d_arr[idxMin], n_arr[idxMin]]])
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
#   grid = np.linspace(0, 1200, num=1201)
#   y = np.array([kde.evaluate(x) for x in grid]).reshape(1201)
    if hist:    
        grid = np.linspace(0, 1200, num=1201)
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

def calculate_energy_from_intervals(scale, m, n):
    int_ratios = sorted([s/min(scale) for s in scale])[1:]
    energy = 0.0 
    for i in int_ratios:
        energy += (abs(float(round(i)) - i))**m * float(round(i))**n
    return energy / float(len(scale))

def test_distinguishability_integer_multiples(df):
    def fn(x, m, n):
        return calculate_energy_from_intervals([int(y) for y in x.split(';')], m, n)
    for n in range(3):
        for m in range(3):
            df[f"distI_{m}_{n}"] = df.pair_ints.swifter.apply(lambda x: fn(x, m, n))
    return df

def test_distinguishability_window_distance(df):
    return calculate_optimum_window_size(df)

def test_harmonic_series_similarity(df, n):
    for i in range(5,25,5):
        df[f"hs_n{n}_w{i:02d}"] = get_harmonic_similarity_score_series(df.all_ints, i, n)
    return df

def process_df(df, grid):
#   if grid:
#       df['scale'] = df.scale.swifter.apply(lambda x: '0;' + x)
    df['min_int'] = df.pair_ints.swifter.apply(lambda x: min([int(y) for y in x.split(';')]))
    df = df.drop(index=df.loc[df.min_int==0].index).reset_index(drop=True)
    df['max_int'] = df.pair_ints.swifter.apply(lambda x: max([int(y) for y in x.split(';')]))
    df = test_distinguishability_integer_multiples(df)
    df = test_distinguishability_window_distance(df)
    df['opt_c_I1'] = df.opt_c * df.distI_0_1
    df['opt_c_I2'] = df.opt_c * df.distI_0_2
    def small_int_bias(x, n):
        ints = np.array([int(y) for y in x.split(';')])
        return np.sum(ints**n) / 1200.
    df['opt_c_s2'] = df.opt_c * df.pair_ints.swifter.apply(lambda x: small_int_bias(x, 2))
    df['opt_c_s3'] = df.opt_c * df.pair_ints.swifter.apply(lambda x: small_int_bias(x, 3))
    df = test_harmonic_series_similarity(df, 1)
    df = test_harmonic_series_similarity(df, 2)
    return df

def process_grid_similar_scales(df_grid, df_real, n):
    def fn(x, df_real, idx, w):
        return ';'.join([str(i) for i in idx if is_scale_similar(x, df_real.loc[i, 'scale'], w)])
    idx = df_real.loc[df_real.n_notes==n].index
    for w in [10, 20]:
        df_grid[f'ss_w{w:02d}'] = df_grid.scale.swifter.apply(lambda x: fn(x, df_real, idx, w))
    return df_grid

def is_scale_similar(x, y, w):
    xint = [int(a) for a in x.split(';')]
    yint = [int(a) for a in y.split(';')]
    return np.allclose(xint, yint, atol=w)

def how_much_real_scales_predicted(df, n_real, w, s):
    try:
        return float(len(set([int(x) for y in df[f"{s}_w{w:02d}"] for x in y.split(';') if len(y)]))) / float(n_real)
    except:
        return None

if __name__ == "__main__":

    timeS = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--partabase', action='store', default='None', type=str)
    args = parser.parse_args()

    categories = ['pair_ints', 'scale']
    n_arr = np.arange(4,10,dtype=int)

    if os.path.exists(os.path.join(REAL_DIR, 'theories_real_scales.feather')):
        df_real = pd.read_feather(os.path.join(REAL_DIR, 'theories_real_scales.feather'))
    else:
        df_real = pd.read_feather(os.path.join(REAL_DIR, 'real_scales.feather'))
        df_real = process_df(df_real, 0)
        df_real.to_feather(os.path.join(REAL_DIR, 'theories_real_scales.feather'))

    if args.partabase == 'theory':
        df_real = df_real.drop(labels=df_real.loc[df_real.Continent.apply(lambda x: x in ['Western', 'East Asia', 'South Asia', 'Middle East'])].index, axis=0)
    elif args.partabase == 'instrument':
        df_real = df_real.drop(labels=df_real.loc[df_real.Continent.apply(lambda x: x not in ['Western', 'East Asia', 'South Asia', 'Middle East'])].index, axis=0)
    
    real_KDE  = {cat:{n: get_KDE(df_real.loc[df_real.n_notes==n], cat) for n in n_arr} for cat in categories}
    real_HIST = {cat:{n: smooth_dist_kde(df_real.loc[df_real.n_notes==n], cat, hist=True) for n in n_arr} for cat in categories}

    print(f"Real scales loaded after {(time.time()-timeS)/60.} minutes")


#   raw_files = glob.glob(RAW_DIR+'*feather')
    pro_files = [f for f in sorted(glob.glob(PRO_DIR+'*feather')) if 'sample' not in f]

#   [print(f) for f in pro_files if 'n9' in f]
#   sys.exit()

#   print(raw_files)

    def read_model_results(path):
        fName = os.path.split(path)[1]
        pro_name = os.path.join(PRO_DIR, fName)
        if os.path.exists(pro_name):
            return pd.read_feather(pro_name)
        else:
            path = os.path.join(RAW_DIR, fName)
            tmp_df = pd.read_feather(path)
            tmp_df['beta'] = float(fName.split('_')[-1].strip('.feather'))
            tmp_df = process_df(tmp_df, 1)
            n = int(fName.split('_')[0].strip('n'))
            tmp_df = process_grid_similar_scales(tmp_df, df_real, n)
            tmp_df.to_feather(pro_name)
            return tmp_df

#   pool = mp.Pool(N_PROC)

#   df_grids = list(pool.map(read_model_results, raw_files))

#   df_grids = []
#   for f in raw_files:
#       df_grids.append(read_model_results(f))
#   df_grid = pd.concat(df_grids, ignore_index=True)

#   df_grids = []
#   pool.close()
#   pool.join()

    print(f"MC scales loaded after {(time.time()-timeS)/60.} minutes")

    def extract_stats_each_model(fName):
        df_g = read_model_results(fName)
        bits = os.path.split(fName)[1].split('_')
        n = int(bits[0].strip('n'))
        idx = [i for i in range(len(bits)) if bits[i][0]=='M'][0]
        bias = '_'.join(bits[1:idx])
        mi = int(bits[idx].strip('MI'))
        ma = int(bits[idx+1].strip('MA'))
        beta = float(bits[-1].strip('.feather'))
        cat = 'pair_ints'

#       if bias == 'none':
#           df_g = df_grid.loc[(df_grid.n_notes==n)&(df_grid.min_int>mi)&(df_grid.max_int<ma)]
#       else:
#           df_g = df_grid.loc[(df_grid.n_notes==n)&(df_grid.min_int>mi)&(df_grid.max_int<ma)&(df_grid.beta==beta)]

        n_sample = df_g.n_att.sum()
        q = float(len(df_g))/float(n_sample)

        yKDE = get_KDE(df_g, cat)

        RE_pq = calc_relative_entropy(real_KDE[cat][n], yKDE)
        RE_qp = calc_relative_entropy(yKDE, real_KDE[cat][n])
        JSD = calc_jensen_shannon_distance(real_KDE[cat][n], yKDE)

#       packing_KDE = np.load(os.path.join(DIST_DIR, f"n{n}_none_MI{mi}_MA{ma}_BETA_000.000.npy"))
#       packing_KDE = packing_KDE.reshape(packing_KDE.size)
#       JSDex = calc_jensen_shannon_distance(packing_KDE, yKDE)

#       norm = np.sqrt(np.dot(real_KDE[cat][n], yKDE))

        n_real = len(df_real.loc[df_real.n_notes==n])
        frac_real = [how_much_real_scales_predicted(df_g, n_real, w, s) for s in ['ss', 'mss', 'pi'] for w in [10, 20]]
        frac_model =  [float(len(df_g.loc[df_g[f"{s}_w{w}"].str.len()>0])) / float(len(df_g)) if f"{s}_w{w}" in df_g.columns else None for s in ['ss', 'mss', 'pi'] for w in [10, 20]]

        metrics = utils.try_out_metrics(real_KDE[cat][n], yKDE)

        _, _, _, yHIST = smooth_dist_kde(df_g, 'scale', hist=True)
        scale_met = utils.scale_metrics(real_HIST['scale'][n][3], yHIST)

        return [cat, n, mi, ma, bias, beta, q, n_sample, RE_pq, RE_qp, JSD] + frac_real + frac_model + metrics + scale_met + [fName]

    biases = ['none',
             'distI_1_0', 'distI_2_0', 'distI_3_0', 'distI_0_1', 'distI_0_2',
             'distI_1_1', 'distI_2_1', 'distI_1_2', 'distI_2_2',
             'opt_c', 'opt_c_I1', 'opt_c_I2', 'opt_c_s2', 'opt_c_s3'] + \
             [f"hs_n{i}_w{w:02d}" for i in range(1,4) for w in [5,10,15,20]] + \
             [f"hs_r3_w{w:02d}" for w in [5,10,15,20]] + \
             [f"ahs{i:02d}_w{w:02d}" for i in range(1,11) for w in [5,10,15,20]] + \
             [f"im5_r{r:3.1f}_w{w:02d}" for r in [0, 0.5, 1, 2] for w in [5,10,15,20]] + \
             [f"Nhs_n1_w{w:02d}" for w in [5,10,15,20]] + \
             [f"Nhs_n2_w{w:02d}" for w in [5,10,15,20]] + \
             [f"Nhs_n3_w{w:02d}" for w in [5,10,15,20]] + \
             [f"Nim5_r0.0_w{w:02d}" for w in [5,10,15,20]] + \
             [f"TRANSB_{i}" for i in [1,2,3]]

#            ['hs_r3_w05', 'hs_r3_w10', 'hs_r3_w15', 'hs_r3_w20'] + \
#            [f"im5_r0.75_w{w:02d}" for w in [5,10,15,20] + 

    groups = ['none'] + ['distI']*3 + ['S#1']*2 + ['distI_S#1']*4 + \
             ['distW'] + ['distW_S#1']*2 + ['distW_S#2']*2 + ['HS']*12 + ['im5']*4 + ['AHS']*40 + ['im5']*16 + \
             ['HS']*12 + ['im5']*4 + ['TRANSB']*3
    bias_groups = {biases[i]:groups[i] for i in range(len(biases))} 
    
    min_int_arr = np.array([0.] + list(np.arange(50., 110., 10.)))
    max_int_arr = np.array(list(np.arange(400., 600., 50.)) + [1200.])

    pool = mp.Pool(N_PROC)

    results = list(pool.imap_unordered(extract_stats_each_model, pro_files))
    print(f"Model comparison finished after {(time.time()-timeS)/60.} minutes")
    df = pd.DataFrame(columns=['cat', 'n_notes', 'min_int', 'max_int', 'bias', 'beta', 'quantile', 'n_sample',
                               'RE_pq', 'RE_qp', 'JSD', 'fr_10', 'fr_20', 'mfr_10', 'mfr_20', 'pi_10', 'pi_20', 
                               'fm_10', 'fm_20', 'mfm_10', 'mfm_20', 'pm_10', 'pm_20',
                               'gn', 'es', 'bp_gn', 'bp_es', 'cum_gn', 'cum_es',
                               'peak_ratio', 'peak_dist', 'deriv_gn', 'deriv_es', 'sc_es', 'sc_des', 'fName'], data=results)


#   df = df.drop(df.loc[df.isnull().any(axis=1)].index)
#   df = df.reset_index(drop=True) 

    df['bias_group'] = df.bias.apply(lambda x: bias_groups[x])
    df['met1'] = (df.es * df.deriv_es)**0.5
    df['logq'] = np.log10(df['quantile'])
    df = graphs.rename_bias_groups(df)
    df = graphs.rename_biases(df)
    

    print(f"DataFrame compiled after {(time.time()-timeS)/60.} minutes")

    if args.partabase == 'None':
        df.to_feather(os.path.join(BASE_DIR, 'monte_carlo_comparison.feather'))
    elif args.partabase == 'theory':
        df.to_feather(os.path.join(BASE_DIR, 'monte_carlo_comparison_theory.feather'))
    elif args.partabase == 'instrument':
        df.to_feather(os.path.join(BASE_DIR, 'monte_carlo_comparison_instrument.feather'))

#   real_KDE  = []
#   real_hist = []
#   for n in range(4,10):
#       new_grid, new_y, xHist, yHist = get_real_scales_dists(n, df_real)
#       real_KDE.append(new_y)
#       real_hist.append(yHist)

#   df = pd.DataFrame(columns=['n_notes', 'min_int', 'max_int', 'beta', 'RE_pq', 'RE_qp', 'JSD', 'kde_path', 'hist_path'])

#   for root, dirs, files in os.walk(MC_DIR):
#       for fName in files:
#           base, ext = os.path.splitext(fName)
#           if ext != '.feather':
#               continue
#           
#           df_mc = pd.read_feather(fName)
#           base_split = base.split('_')
#           int_min = int(base_split[2])
#           int_max = int(base_split[4].strip('max'))
#           beta    = int(base_split[6])

#           for i, n in enumerate(range(4,10)):
#               path = os.path.join(root, base + f'_n_{n}_kde.npy')
#               hist_path = os.path.join(root, base + f'_n_{n}_hist.npy')
#               dataKDE  = np.load(path)
#               dataHIST = np.load(os.path.join(root, base + f'_n_{n}_hist.npy'))

#               RE_pq = calc_relative_entropy(real_KDE[i], dataKDE[:,1])
#               RE_qp = calc_relative_entropy(dataKDE[:,1], real_KDE[i])
#               JSD = calc_jensen_shannon_distance(real_KDE[i], dataKDE[:,1])

#               df.loc[len(df)] = [n, int_min, int_max, beta, RE_pq, RE_qp, JSD, path, hist_path]



