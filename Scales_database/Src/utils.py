
import re
import sys
import time

import matplotlib.pyplot as plt
from itertools import permutations
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
import statsmodels.nonparametric.api as smnp
import swifter

# Interval harmonic similarity ratings as detailed in:
# doi:10.1371/journal.pone.0008144.t001

INST = np.array([1,0,1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,0,1,0,1,1,1,1], dtype=bool)
CENT_DIFF_MAX = 11.0

BETA = 50.

PYT_INTS = np.array([0., 90.2, 203.9, 294.1, 407.8, 498.1, 611.7, 702., 792.2, 905., 996.1, 1109.8, 1200.])
EQ5_INTS = np.linspace(0, 1200, num=6, endpoint=True, dtype=float)
EQ7_INTS = np.linspace(0, 1200, num=8, endpoint=True, dtype=float)
EQ9_INTS = np.linspace(0, 1200, num=10, endpoint=True, dtype=float)
EQ10_INTS = np.linspace(0, 1200, num=11, endpoint=True, dtype=float)
EQ12_INTS = np.linspace(0, 1200, num=13, endpoint=True, dtype=float)
EQ24_INTS = np.linspace(0, 1200, num=25, endpoint=True, dtype=float)
EQ53_INTS = np.linspace(0, 1200, num=54, endpoint=True, dtype=float)
JI_INTS = np.array([0., 111.7, 203.9, 315.6, 386.3, 498.1, 590.2, 702., 813.7, 884.4, 1017.6, 1088.3, 1200.])
SLENDRO = np.array([263., 223., 253., 236., 225.])
PELOG   = np.array([167., 245., 125., 146., 252., 165., 100.])
DASTGAH = np.array([0., 90., 133.23, 204., 294.14, 337.14, 407.82, 498., 568.72, 631.28, 702., 792.18, 835.2, 906., 996., 1039.1, 1109.77, 1200.])
TURKISH = {'T':203.8, 'K':181.1, 'S':113.2, 'B':90.6, 'F':22.6, 'A':271, 'E':67.9}
KHMER_1 = np.array([185., 195., 105., 195., 195., 185., 140.])
KHMER_2 = np.array([190., 190., 130., 190., 190., 190., 120.])
VIET    = np.array([0., 175., 200., 300., 338., 375., 500., 520., 700., 869., 900., 1000., 1020., 1200.])
CHINA   = np.array([0., 113.67291609,  203.91000173,  317.73848174,  407.83554758, 520.68758457,  611.71791523,  701.95500087,  815.62791696, 905.8650026 , 1019.47514332, 1109.76982292, 1201.27828039])


OCT_CUT = 50

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
        return ';'.join([str(d) for d in dist])

def get_distance_between_windows(df, w, X='pair_ints'):
    df[f"d_w{w}"] = df.loc[:,X].swifter.apply(lambda x: calculate_distance_between_windows(x))
    df[f"d_w{w}_min"] = df.loc[:,X].swifter.apply(lambda x: min([int(y) for y in x.split(';')]))
    df[f"d_w{w}_mean"] = df.loc[:,X].swifter.apply(lambda x: min([int(y) for y in x.split(';')]))
    return df

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

def smooth_dist_kde(df):
    X = [float(x) for y in df.pair_ints for x in y.split(';')]
    kde = smnp.KDEUnivariate(np.array(X))
    kde.fit('gau', 'scott', 1, gridsize=10000, cut=20)
    grid, y = kde.support, kde.density
    return grid, y

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
        
def get_attractors(dI=5.):
    sc_i = np.arange(dI, 1200.+dI, dI)
    sc_f = set()
    attract = []
    ratios = []
    simils = []
    for s in sc_i:
        max_similarity, best_ratio, cents = calculate_most_harmonic_neighbour(s)
        if max_similarity == 0.0:
            continue
        if round(cents,2) not in sc_f:
            sc_f.add(round(cents,2))
            attract.append(round(cents,2))
            ratios.append(best_ratio)
            simils.append(max_similarity)
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

def get_harmonic_similarity_score_df(df):
    sc_i, sc_f, ratios, simil  = get_attractors()
    df['harm_sim'] =  df.all_ints.apply(lambda x: np.mean([get_similarity_of_nearest_attractor(float(y), sc_f, simil) for y in x.split(';')]))
    return df

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

def calculate_most_harmonic_neighbour(int_cents, sim_only=False):
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
    ref = []
    theory = []
    for row in df.itertuples():
        try:
            idx = np.where(np.array([int(x) for x in row.mask]))[0]
        except:
            pass

        for tun in row.Tuning.split(';'):

            if tun == '12-tet':
                scale = EQ12_INTS[idx]
            elif tun == '53-tet':
                scale = EQ53_INTS[idx]
            elif tun == 'Just':
                scale = JI_INTS[idx]
            elif tun == 'Pythagorean':
                scale = PYT_INTS[idx]
            elif tun == 'Arabian':
                scale = EQ24_INTS[idx]
            elif tun == 'Dastgah-ha':
                scale = DASTGAH[idx]
            elif tun == 'Vietnamese':
                scale = VIET[idx]
            elif tun == 'Chinese':
                scale = CHINA[idx]
            elif tun == 'Turkish':
                scale = np.cumsum([0.0] + [TURKISH[a] for a in row.Intervals])
            elif tun == 'Khmer':
                for KHM in [KHMER_1, KHMER_2]:
                    base = KHM[[i-1 for i in idx[1:]]]
                    for i in range(len(base)):
                        scale = np.cumsum([0.] + np.roll(KHM,i))
                        names.append(row.Name)
                        scales.append(scale)
                        all_ints.append([scale[i] - scale[j] for j in range(len(scale)) for i in range(j+1,len(scale))])
                        pair_ints.append([scale[j+1] - scale[j] for j in range(len(scale)-1)])
                        cultures.append(row.Culture)
                        tunings.append(tun)
                        conts.append(row.Continent)
                        ref.append(row.Reference)
                        theory.append(row.Theory)
                continue
            elif tun == 'Unique':
                scale = np.cumsum([0.] + [float(x) for x in row.Intervals.split(';')])
            else:
                print(row.Name, tun, tun=='12-tet')
                continue

            names.append(row.Name)
            scales.append(scale)
            all_ints.append([scale[i] - scale[j] for j in range(len(scale)) for i in range(j+1,len(scale))])
            pair_ints.append([scale[j+1] - scale[j] for j in range(len(scale)-1)])
            cultures.append(row.Culture)
            tunings.append(tun)
            conts.append(row.Continent)
            ref.append(row.Reference)
            theory.append(row.Theory)

    return cultures, tunings, conts, names, scales, all_ints, pair_ints, ref, theory

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
    return dist
    
def plot_2gram_dist_by_n_notes(df, dI=10):
    fig, ax = plt.subplots(2,3)
    ax = ax.reshape(ax.size)
    for i, n in enumerate([4,5,6,7,8,9]):
        dist = get_2grams_dist(df.loc[df.n_notes==n], dI=dI)
        sns.heatmap(np.log(dist[::-1]+0.1), label=str(n), ax=ax[i])
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

def reformat_surjodiningrat(df):
    for row in df.itertuples():
        ints = [get_cents_from_ratio(float(row[i+3])/float(row[i+2])) for i in range(7) if row[i+3] != 0]
        df.loc[row[0], 'pair_ints'] = ';'.join([str(int(round(x))) for x in ints])
    df['Reference'] = 'Surjodiningrat'
    df['Theory'] = 'N'
    df = df.drop(columns=[str(x) for x in range(1,9)])
    return df

def reformat_original_csv_data(df):
    new_df = pd.DataFrame(columns=['Name', 'Intervals', 'Culture', 'Continent', 'Tuning', 'Reference', 'Theory'])
    for i, col in enumerate(df.columns):
        tuning  = df.loc[0, col]
        culture = df.loc[1, col]
        cont    = df.loc[2, col]
        ref     = df.loc[3, col]
        theory  = df.loc[4, col]
        try:
            int(col)
            name = '_'.join([culture, col])
        except:
            name = col
        ints = ';'.join([str(int(round(float(x)))) for x in df.loc[5:, col] if not str(x)=='nan'])
        new_df.loc[i] = [name, ints, culture, cont, tuning, ref, theory]
    return new_df

def extract_scales_and_ints_from_unique(df):
    names = []
    scales = []
    all_ints = []
    pair_ints = []
    cultures = []
    tunings = []
    conts = []
    ref = []
    theory = []
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
            ref.append(row.Reference)
            theory.append('N')

            start_from = idx_oct + i

    return cultures, tunings, conts, names, scales, all_ints, pair_ints, ref, theory

