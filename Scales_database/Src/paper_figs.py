from itertools import product
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from multiprocessing import Pool
import numpy as np
from palettable.colorbrewer.qualitative import Paired_12
import pandas as pd
import seaborn as sns

import octave as OC
import utils

N_PROC = 8

PATH_FIG = Path("../Figures")
PATH_DATA = Path("../Figures/Data")


def set_ticks(ax, xMaj, xMin, xForm, yMaj, yMin, yForm):
    ax.xaxis.set_major_locator(MultipleLocator(xMaj))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xForm))
    ax.xaxis.set_minor_locator(MultipleLocator(xMin))

    ax.yaxis.set_major_locator(MultipleLocator(yMaj))
    ax.yaxis.set_major_formatter(FormatStrFormatter(yForm))
    ax.yaxis.set_minor_locator(MultipleLocator(yMin))


def major_ticks( ax ):
    ax.tick_params(axis='both', which='major', right='on', top='on', \
                  labelsize=12, length=6, width=2, pad=8)


def minor_ticks( ax ):
    ax.tick_params(axis='both', which='minor', right='on', top='on', \
                  labelsize=12, length=3, width=1, pad=8)


def scale_degree(df, norm=False, nmax=9):
    fig, ax = plt.subplots()
    data = pickle.load(open(PATH_DATA.joinpath("scale_degree.pickle"), 'rb'))
    X = data['X']
    width = 0.15
    lbls = ['All', 'Theory', 'Measured', 'Continent', 'Culture']
    cols = sns.color_palette()
    for i, l in enumerate(lbls):
        m, lo, hi = data[l]
        err = np.array([m - lo, hi - m])
        ax.bar(X + (i-2)*width, m, width, yerr=err, color=cols[i], label=l, ec='k')

    ax.legend(loc='best', frameon=False)
    ax.set_xlabel("Scale degree")
    ax.set_ylabel("Normalised frequency")

    fig.savefig(PATH_FIG.joinpath("scale_degree.pdf"), bbox_inches='tight')


def scale_dist():
    fig, ax = plt.subplots()
    data = pickle.load(open(PATH_DATA.joinpath("scale.pickle"), 'rb'))
    X = data['X']

    lbls = ['All', 'Theory', 'Measured', 'Continent', 'Culture']
    cols = Paired_12.hex_colors
    for i, l in enumerate(lbls):
        m, lo, hi = data[l]
        ax.plot(X, m, '-', c=cols[i*2+1], label=lbls[i])
        ax.fill_between(X, lo, hi, color=cols[i*2], alpha=0.5)

    ax.legend(loc='best', frameon=False)
    ax.set_xlabel("Scale note")
    ax.set_ylabel("Normalised frequency")

    fig.savefig(PATH_FIG.joinpath("scale_dist.pdf"), bbox_inches='tight')


def adjacent_int_dist():
    fig, ax = plt.subplots()
    data = pickle.load(open(PATH_DATA.joinpath("adjacent_int.pickle"), 'rb'))
    X = data['X']

    lbls = ['All', 'Theory', 'Measured', 'Continent', 'Culture']
    cols = Paired_12.hex_colors
    for i, l in enumerate(lbls):
        m, lo, hi = data[l]
        ax.plot(X, m, '-', c=cols[i*2+1], label=lbls[i])
        ax.fill_between(X, lo, hi, color=cols[i*2], alpha=0.5)

    ax.legend(loc='best', frameon=False)
    fig.savefig(PATH_FIG.joinpath("int_dist.pdf"), bbox_inches='tight')


def all_int_dist():
    fig, ax = plt.subplots()
    data = pickle.load(open(PATH_DATA.joinpath("all_int.pickle"), 'rb'))
    X = data['X']

    lbls = ['All', 'Theory', 'Measured', 'Continent', 'Culture']
    cols = Paired_12.hex_colors
    for i, l in enumerate(lbls):
        m, lo, hi = data[l]
        ax.plot(X, m, '-', c=cols[i*2+1], label=lbls[i])
        ax.fill_between(X, lo, hi, color=cols[i*2], alpha=0.5)

    ax.legend(loc='best', frameon=False)
    fig.savefig(PATH_FIG.joinpath("all_int_dist.pdf"), bbox_inches='tight')


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
    tots = []
    
    res_lbl = ' & '.join(['Total', 'Testable', 'Support', 'Against', 'Non-sig'])
    print(f"Name & Culture &  {res_lbl} & Text support & Culture support & Overall")
    for i in text.index:
        k = i + 1
        try:
            name = f"{int(k):4d}  {src_cult[k][:40]:40s}"
            inst_results_list = [d.get(k,0) for d in [src_cnt_tot, src_cnt_use, src_support, src_against, src_insig]]
            inst_results = " & ".join([f"{d:4d}" for d in inst_results_list])
            t1, t2 = text.loc[i]
            culture = identify_culture(df.loc[df.RefID==k, 'Culture'].values)
#           print(src_against.get(k,0), src_support.get(k,0), t1, t2)
            evidence = (src_against.get(k,0) < src_support.get(k,0)) | ('Yes' in t1) | ('Yes' in t2)
            tots.append(inst_results_list +  [('Yes' in t1), ('Yes' in t2), evidence])
            print(f"{name} & {culture[:15]:15s} & {inst_results} & {str(t1):6s} & {str(t2):10s} & {evidence:3b}")
        except Exception as e:
#           print(e)
            pass
    tots = np.sum(tots, axis=0)
    print(" "*65 + "  ".join([f"{int(t):4d}" for t in tots]))


def load_interval_data(path_list):
    data = np.array([np.load(path) for path in path_list])
    total = data.sum(axis=1)
    Y1 = data[:,0,:] / total
    Y2 = data[:,1,:] / total
    out = []
    for y in [Y1, Y2]:
        out.append(np.mean(y, axis=1))
        out.append(np.quantile(y, 0.025, axis=1))
        out.append(np.quantile(y, 0.975, axis=1))
    return out


def interval_sig():
    fig, ax = plt.subplots(3,1)
    ints = np.arange(200, 2605, 5)
    col = sns.color_palette()

    m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/0_w1100_w220_I{i:04d}.npy" for i in ints])
    ax[0].plot(ints, m1, '-', label='Support', color=col[0])
    ax[0].fill_between(ints, lo1, hi1, color='grey')
    ax[0].plot(ints, m2, ':', label='Against', color=col[1])
    ax[0].fill_between(ints, lo2, hi2, color='grey')

    for j in range(1, 3):
        m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/{j}_w1100_w220_I{i:04d}.npy" for i in ints])
        ax[1].plot(ints, m1, '-', label='Support', color=col[0])
        ax[1].fill_between(ints, lo1, hi1, color='grey')
        ax[1].plot(ints, m2, ':', label='Against', color=col[1])
        ax[1].fill_between(ints, lo2, hi2, color='grey')

    cont = ['South East Asia', 'Africa', 'Oceania', 'South Asia', 'Western',
            'South America', 'East Asia', 'Middle East']
    for c in cont[:5]:
        m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/{c}_w1100_w220_I{i:04d}.npy" for i in ints])
        ax[2].plot(ints, m1, '-', label='Support', color=col[0])
        ax[2].fill_between(ints, lo1, hi1, color='grey')
        ax[2].plot(ints, m2, ':', label='Against', color=col[1])
        ax[2].fill_between(ints, lo2, hi2, color='grey')

    for a in ax:
        a.set_xlabel("Interval size (cents)")
        a.set_ylabel("Fraction of\nsignificant results")
        a.set_ylim(0, 0.4)
        a.set_xticks(range(0, 2700, 200))
        a.xaxis.set_tick_params(which='minor', bottom=True)

        set_ticks(a, 200, 100, '%d', 0.1, 0.05, '%3.1f')

    fig.savefig(PATH_FIG.joinpath("interval_sig.pdf"), bbox_inches='tight')


def window_size():
    fig, ax = plt.subplots(7,1)
    ints = np.arange(200, 2605, 5)
    col = sns.color_palette()

    w1_list = [50, 75, 100, 125, 150, 175, 200]
    w2_list = [5, 10, 15, 20, 30, 40]
    for i, w1 in enumerate(w1_list):
        for w2 in w2_list:
            try:
                print(w1, w2)
                m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/0_w1{w1}_w2{w2}_I{i:04d}.npy" for i in ints])
                ax[i].plot(ints, m1, '-', label='Support', color=col[0])
                ax[i].fill_between(ints, lo1, hi1, color='grey')
                ax[i].plot(ints, m2, ':', label='Against', color=col[1])
                ax[i].fill_between(ints, lo2, hi2, color='grey')
            except:
                print(f"{w1}_{w2} not done yet")

    fig.savefig(PATH_FIG.joinpath("window_effect.pdf"), bbox_inches='tight')


def inst_notes(df):
    fig, ax = plt.subplots(3,1)
    OC.get_int_prob_via_sampling(df, ysamp='scale', xsamp='', ax=ax[0])
    for i in range(3):
        OC.get_int_prob_via_sampling(df, ysamp='scale', xsamp='Continent', s=6, ax=ax[1]) 
        OC.get_int_prob_via_sampling(df, ysamp='scale', xsamp='Culture', s=1, ax=ax[2]) 
    
    for a in ax:
        a.set_ylabel('Density')
        a.set_xlabel('Interval compared to lowest note')

    fig.savefig(PATH_FIG.joinpath("inst_notes.pdf"), bbox_inches='tight')



