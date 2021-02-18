from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import seaborn as sns

import octave as OC
import utils

N_PROC = 8

PATH_FIG = Path("../Figures")


def scale_degree(df, norm=False, nmax=9):
    fig, ax = plt.subplots()
    bins = np.arange(3.5, nmax+1, 1)
    X = bins[:-1] + 0.5 * np.diff(bins[:2])

    width = 0.25
    hist = [np.histogram(df.n_notes, bins=bins)[0]] + \
           [np.histogram(df.loc[df.Theory==s, "n_notes"], bins=bins)[0] for s in 'YN']

    if norm:
        hist = [h / sum(h) for h in hist]

    col = sns.color_palette()
    lbls = ['All', 'Theory', 'Measured']
    for i, h in enumerate(hist):
        ax.bar(X + (i-1)*width, h/h.sum(), width, color=col[i], ec='k', label=lbls[i])
    ax.legend(loc='best', frameon=False)
    ax.set_xlabel("Scale degree")
    ax.set_ylabel("Normalised frequency")

    fig.savefig(PATH_FIG.joinpath("scale_degree.pdf"), bbox_inches='tight')


def scale_dist(df):
    fig, ax = plt.subplots()

    n_arr = [5, 7]
    df_list = [df, df.loc[df.Theory=='Y'], df.loc[df.Theory=='N']]
    bins = np.arange(-10, 1220, 20)
    X = bins[:-1] + 0.5 * np.diff(bins[:2])
    lbls = ['All', 'Theory', 'Measured']


    for i, df in enumerate(df_list):
#       for j in range(3):
#           if not j:
        hist = np.histogram([x for y in df.scale for x in utils.str_to_ints(y)], bins=bins)[0]
#           else:
#               hist = np.histogram([x for y in df.loc[df.n_notes==n_arr[j-1], "scale"] for x in utils.str_to_ints(y)], bins=bins)[0]
        hist = hist / hist.sum()
        ax.plot(X, hist, label=lbls[i])
#       if not i:
#           ax[i].set_xticks([])
    ax.legend(loc='best', frameon=False)
    ax.set_xlabel("Scale note")
    ax.set_ylabel("Normalised frequency")

    fig.savefig(PATH_FIG.joinpath("scale_dist.pdf"), bbox_inches='tight')


def int_dist(df):
    fig, ax = plt.subplots()

    n_arr = [5, 7]
    df_list = [df, df.loc[df.Theory=='Y'], df.loc[df.Theory=='N']]
    bins = np.arange(-10, 520, 10)
    X = bins[:-1] + 0.5 * np.diff(bins[:2])
    lbls = ['All', 'Theory', 'Measured']


    for i, df in enumerate(df_list):
#       for j in range(3):
#           if not j:
        hist = np.histogram([x for y in df.pair_ints for x in utils.str_to_ints(y)], bins=bins)[0]
#           else:
#               hist = np.histogram([x for y in df.loc[df.n_notes==n_arr[j-1], "pair_ints"] for x in utils.str_to_ints(y)], bins=bins)[0]
        hist = hist / hist.sum()
        ax.plot(X, hist, label=lbls[i])
    ax.legend(loc='best', frameon=False)

    fig.savefig(PATH_FIG.joinpath("int_dist.pdf"), bbox_inches='tight')


def octave_equiv(df, octave=1200, n_rep=10):
    res = OC.octave_chance_individual(df, octave=octave, n_rep=n_rep)
    fig, ax = plt.subplots(2,2)
    ax = ax.reshape(ax.size)
    sns.scatterplot(x='f_real', y='f_shuf', data=res, hue='sig', ax=ax[0])
    mx = max(res.f_real.max(), res.f_shuf.max())
    ax[0].plot([0, mx], [0, mx], '-k')

    sns.scatterplot(x='mean_real', y='mean_shuf', data=res, hue='sig', ax=ax[1])
    mx = max(res.mean_real.max(), res.mean_shuf.max())
    ax[1].plot([0, mx], [0, mx], '-k')

    ax[0].set_xlabel("Fraction of all real intervals within w2 of octave")
    ax[1].set_xlabel("Deviation of all real intervals within (w1 of octave) from the octave")
    ax[0].set_ylabel("Fraction of all shuffled intervals within w2 of octave")
    ax[1].set_ylabel("Deviation of all shuffled intervals within (w1 of octave) from the octave")

    ax[0].legend(loc='best', frameon=False)
    ax[1].legend(loc='best', frameon=False)

    w_arr = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    n_greater = []
    n_less = []
    n_nonsig = []
    for w in w_arr:
        for i in range(5):
            res = OC.octave_chance_individual(df, octave=octave, n_rep=n_rep, w2=w)
            n_greater.append(len(res.loc[(res.MWU<0.05)&(res.f_real>res.f_shuf)]))
            n_less.append(len(res.loc[(res.MWU<0.05)&(res.f_real<res.f_shuf)]))
            n_nonsig.append(len(res.loc[(res.MWU>=0.05)]))
    n_greater = np.array(n_greater).reshape(len(w_arr), 5).mean(axis=1)
    n_less = np.array(n_less).reshape(len(w_arr), 5).mean(axis=1)
    n_nonsig = np.array(n_nonsig).reshape(len(w_arr), 5).mean(axis=1)
    total = np.sum([n_greater, n_less, n_nonsig], axis=0)

    ax[2].plot(w_arr, n_greater / total, label='support')
    ax[2].plot(w_arr, n_less / total, label='against')
    ax[2].plot(w_arr, n_nonsig / total, label='non_sig')

    ax[2].set_xlabel("window_1")
    ax[2].set_ylabel("Normalised frequency")
    ax[2].legend(loc='best', frameon=False)

    w_arr = [50, 75, 100, 125, 150, 175, 200]
    n_greater = []
    n_less = []
    n_nonsig = []
    for w in w_arr:
        print(w)
        for i in range(5):
            res = OC.octave_chance_individual(df, octave=octave, n_rep=n_rep, w1=w)
            n_greater.append(len(res.loc[(res.MWU<0.05)&(res.mean_real<res.mean_shuf)]))
            n_less.append(len(res.loc[(res.MWU<0.05)&(res.mean_real>res.mean_shuf)]))
            n_nonsig.append(len(res.loc[(res.MWU>=0.05)]))
    n_greater = np.array(n_greater).reshape(len(w_arr), 5).mean(axis=1)
    n_less = np.array(n_less).reshape(len(w_arr), 5).mean(axis=1)
    n_nonsig = np.array(n_nonsig).reshape(len(w_arr), 5).mean(axis=1)
    total = np.sum([n_greater, n_less, n_nonsig], axis=0)

    ax[3].plot(w_arr, n_greater / total, label='support')
    ax[3].plot(w_arr, n_less / total, label='against')
    ax[3].plot(w_arr, n_nonsig / total, label='non_sig')

    ax[3].set_xlabel("window_2")
    ax[3].set_ylabel("Normalised frequency")
    ax[3].legend(loc='best', frameon=False)

    fig.savefig(PATH_FIG.joinpath("octave_demo.pdf"), bbox_inches='tight')



def get_stats(df, i, n_rep=20, w=20, nrep2=10):
    out = np.zeros((3,nrep2), float)
    for j in range(nrep2):
        res = OC.octave_chance_individual(df, octave=i, n_rep=n_rep, w=20)
        out[0,j] = len(res.loc[(res.MWU<0.05)&(res.mean_real<res.mean_shuf)])
        out[1,j] = len(res.loc[(res.MWU<0.05)&(res.mean_real>res.mean_shuf)])
        out[2,j] = len(res.loc[(res.MWU>=0.05)])
    return out.mean(axis=1)


def unexpected_intervals(df, n_rep=20):
    fig, ax = plt.subplots(2,1)
    df = df.loc[:, ['Intervals', 'scale']]

    ints = np.arange(200, 2520, 10)
    with Pool(N_PROC) as pool:
        res = np.array(pool.starmap(get_stats, product([df], ints)))
    n_greater = res[:,0]
    n_less = res[:,1]
    n_nonsig = res[:,2]

    n_tot = n_greater + n_less + n_nonsig
    
    ax[0].plot(ints, n_greater / n_tot, label='support')
    ax[0].plot(ints, n_less / n_tot, label='against')
    ax[1].plot(ints, (n_greater - n_less) / n_tot, label='against')
    ax[0].plot(ints, [0]*len(ints), ':k')
    ax[1].plot(ints, [0]*len(ints), ':k')
#   ax.plot(ints, n_nonsig / n_tot, label='non_sig')


def identify_culture(cultures):
    if len(set(cultures)) == 1:
        return cultures[0]
    else:
        return 'Multiple'


### NEED TO FIX INDEXING! REF AND DF ARE NOT MATCHING
def octave_by_source(df, res):
    text = OC.load_text_summary()

    df['RefID'] = df['RefID'].astype(int)
    src_cnt_tot = df.RefID.value_counts()
    df0 = df.loc[df.scale.apply(lambda x: x[-2] >= 1200)]
    src_cnt_use = df0.RefID.value_counts()

    src_support = df0.loc[df0.index[res.loc[(res.MWU<0.05)&(res.mean_real<res.mean_shuf)].index], "RefID"].value_counts()
    src_against = df0.loc[df0.index[res.loc[(res.MWU<0.05)&(res.mean_real>res.mean_shuf)].index], "RefID"].value_counts()
    src_insig   = df0.loc[df0.index[res.loc[(res.MWU>=0.05)].index], 'RefID'].value_counts()

    src_cult = {k:c for k, c in zip(df.RefID, df.Reference)}
    
    for i in text.index:
        k = i + 1
        try:
            name = f"{int(k):4d}  {src_cult[k][:40]:40s}"
            inst_results = "  ".join([f"{d.get(k,0):4d}" for d in [src_cnt_tot, src_cnt_use, src_support, src_against, src_insig]])
            t = text.loc[i]
            culture = identify_culture(df.loc[df.RefID==k, 'Culture'].values)
            print(f"{name}  {culture[:15]:15s}  {inst_results}  {str(t):6s}")
        except:
            pass






