from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import pickle

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from multiprocessing import Pool
import numpy as np
from palettable.colorbrewer.qualitative import Paired_12, Set2_8, Dark2_8, Pastel2_8
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, set_link_color_palette
from scipy.spatial.distance import pdist, cdist, jensenshannon
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

def set_xticks(ax, xMaj, xMin, xForm):
    ax.xaxis.set_major_locator(MultipleLocator(xMaj))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xForm))
    ax.xaxis.set_minor_locator(MultipleLocator(xMin))


def major_ticks( ax ):
    ax.tick_params(axis='both', which='major', right='on', top='on', \
                  labelsize=12, length=6, width=2, pad=8)


def minor_ticks( ax ):
    ax.tick_params(axis='both', which='minor', right='on', top='on', \
                  labelsize=12, length=3, width=1, pad=8)


def scale_degree():
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



def multiple_dist():
    fig, ax = plt.subplots(3,3, figsize=(10,6))
    fig.subplots_adjust(wspace=0, hspace=0)
    lblA = ['All', 'Theory', 'Measured']
    lblB = ['Continent', 'Culture']
    lblC = ['Idiophone', 'Aerophone', 'Chordophone']
    path_stem = ['adj_ints', 'scale', 'all_ints2']
    xlbls = ['Adjacent Interval / cents', 'Scale note / cents', 'Interval / cents']
    xlim = [570, 1370, 1370]
    cols = Paired_12.hex_colors

    for j, stem in enumerate(path_stem):
        data = pickle.load(open(PATH_DATA.joinpath(f"{stem}.pickle"), 'rb'))
        X = data['X']
        for i, lbl in enumerate([lblA, lblB, lblC]):
            for k, l in enumerate(lbl):
                m, lo, hi = data[l]
                ax[i,j].plot(X, m, '-', c=cols[((i*len(lblA)+k)*2+1)%12], label=l)
                ax[i,j].fill_between(X, lo, hi, color=cols[((i*len(lblA)+k)*2)%12], alpha=0.5)

            if i == 2:
                ax[i,j].set_xlabel(xlbls[j])
            if j == 0:
                set_ticks(ax[i,j], 200, 50, '%d', 1.02, 1.01, '%4.2f')
            else:
                set_ticks(ax[i,j], 400, 100, '%d', 1.002, 1.001, '%5.3f')
            ax[i,j].set_xlim(0, xlim[j])
            ax[i,0].set_ylabel("Density")

    for a in ax[:,0]:
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi)
        for x in np.arange(100, 400, 100):
            a.plot([x]*2, [lo, hi], ':k', alpha=0.3)

    for a in ax[:,1:].ravel():
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi)
        for x in np.arange(100, 1200, 100):
            a.plot([x]*2, [lo, hi], ':k', alpha=0.3)

    for a in ax[:2,:].ravel():
        a.set_xticks([])
    for a in ax[:,:].ravel():
        a.set_yticks([])
    ax[0,0].legend(loc='upper right', frameon=False)
    ax[1,0].legend(loc='upper right', frameon=False)
    ax[2,0].legend(loc='upper right', frameon=False)

    fig.savefig(PATH_FIG.joinpath("multi_dist.pdf"), bbox_inches='tight')


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
    fig, ax = plt.subplots(2)
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

#   w_arr = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#   n_greater = []
#   n_less = []
#   n_nonsig = []
#   for w in w_arr:
#       for i in range(5):
#           res = OC.octave_chance_individual(df, octave=octave, n_rep=n_rep, w2=w)
#           n_greater.append(len(res.loc[(res.MWU<0.05)&(res.f_real>res.f_shuf)]))
#           n_less.append(len(res.loc[(res.MWU<0.05)&(res.f_real<res.f_shuf)]))
#           n_nonsig.append(len(res.loc[(res.MWU>=0.05)]))
#   n_greater = np.array(n_greater).reshape(len(w_arr), 5).mean(axis=1)
#   n_less = np.array(n_less).reshape(len(w_arr), 5).mean(axis=1)
#   n_nonsig = np.array(n_nonsig).reshape(len(w_arr), 5).mean(axis=1)
#   total = np.sum([n_greater, n_less, n_nonsig], axis=0)

#   ax[2].plot(w_arr, n_greater / total, label='support')
#   ax[2].plot(w_arr, n_less / total, label='against')
#   ax[2].plot(w_arr, n_nonsig / total, label='non_sig')

#   ax[2].set_xlabel("window_1")
#   ax[2].set_ylabel("Normalised frequency")
#   ax[2].legend(loc='best', frameon=False)

#   w_arr = [50, 75, 100, 125, 150, 175, 200]
#   n_greater = []
#   n_less = []
#   n_nonsig = []
#   for w in w_arr:
#       print(w)
#       for i in range(5):
#           res = OC.octave_chance_individual(df, octave=octave, n_rep=n_rep, w1=w)
#           n_greater.append(len(res.loc[(res.MWU<0.05)&(res.mean_real<res.mean_shuf)]))
#           n_less.append(len(res.loc[(res.MWU<0.05)&(res.mean_real>res.mean_shuf)]))
#           n_nonsig.append(len(res.loc[(res.MWU>=0.05)]))
#   n_greater = np.array(n_greater).reshape(len(w_arr), 5).mean(axis=1)
#   n_less = np.array(n_less).reshape(len(w_arr), 5).mean(axis=1)
#   n_nonsig = np.array(n_nonsig).reshape(len(w_arr), 5).mean(axis=1)
#   total = np.sum([n_greater, n_less, n_nonsig], axis=0)

#   ax[3].plot(w_arr, n_greater / total, label='support')
#   ax[3].plot(w_arr, n_less / total, label='against')
#   ax[3].plot(w_arr, n_nonsig / total, label='non_sig')

#   ax[3].set_xlabel("window_2")
#   ax[3].set_ylabel("Normalised frequency")
#   ax[3].legend(loc='best', frameon=False)

#   fig.savefig(PATH_FIG.joinpath("octave_demo.pdf"), bbox_inches='tight')



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
    print(f"Ref & Culture &  {res_lbl} & Text support & Culture support & Overall \\\\")
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
            print(f"{name} & {culture[:15]:15s} & {inst_results} & {str(t1):6s} & {str(t2):10s} & {evidence:3b} \\\\")
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


def load_sampled_interval_data(ints, xsamp, n):
    data = np.array([[np.load(f"../IntStats/{xsamp}samp{j}_w1100_w220_I{i:04d}.npy") for i in ints] for j in range(n)])
#   data = data.mean(axis=0)
    data = np.concatenate([d for d in data], axis=2)
    total = data.sum(axis=1)
    Y1 = data[:,0,:] / total
    Y2 = data[:,1,:] / total
    out = []
    for y in [Y1, Y2]:
        out.append(np.mean(y, axis=1))
        out.append(np.quantile(y, 0.025, axis=1))
        out.append(np.quantile(y, 0.975, axis=1))
    return out


def octave_sig(ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    sigma = np.arange(0, 55, 5)
    data = np.array([np.load(f"../IntStats/sigma{s}_w1100_w220_I1200.npy") for s in sigma])
    lbls = ['Greater freq\nthan chance', 'Less freq\nthan chance']
    for i in range(2):
        frac = data[:,i] / data.sum(axis=1)
        m = np.mean(frac, axis=1)
        lo = np.quantile(frac, 0.025, axis=1)
        hi = np.quantile(frac, 0.975, axis=1)
        ax.plot(sigma, m, label=lbls[i])
        ax.fill_between(sigma, hi, lo, color='grey', alpha=0.5)

    ax.set_xlabel("Tuning deviation (cents)")
    ax.set_ylabel("Fraction of\nsignificant results")
    ax.legend(loc='best', frameon=False)


def interval_sig(res):
    fig = plt.figure(figsize=(10,5))
    gs = GridSpec(12,3, width_ratios=[1, 0.4, 2])
    ax = [fig.add_subplot(gs[i*3:(i+1)*3,2]) for i in range(4)] + \
         [fig.add_subplot(gs[:5,0]), fig.add_subplot(gs[8:,0])]
#   fig, ax = plt.subplots(3,1)
    fig.subplots_adjust(hspace=0, wspace=0)
    ints = np.arange(200, 2605, 5)
    col = sns.color_palette()

    lbls = ['Greater freq than chance', 'Less freq than chance']
    m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/0_w1100_w220_I{i:04d}.npy" for i in ints])
    ax[0].plot(ints, m1, '-', label=lbls[0], color=col[0])
    ax[0].fill_between(ints, lo1, hi1, color='grey')
    ax[0].plot(ints, m2, ':', label=lbls[1], color=col[1])
    ax[0].fill_between(ints, lo2, hi2, color='grey')

    for j in range(1, 3):
        m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/{j}_w1100_w220_I{i:04d}.npy" for i in ints])
        ax[1].plot(ints, m1, '-', label='Support', color=col[0])
        ax[1].fill_between(ints, lo1, hi1, color='grey')
        ax[1].plot(ints, m2, ':', label='Against', color=col[1])
        ax[1].fill_between(ints, lo2, hi2, color='grey')

#   cont = ['South East Asia', 'Africa', 'Oceania', 'South Asia', 'Western',
#           'Latin America', 'East Asia', 'Middle East']
#   for c in cont[:5]:
#       m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/{c}_w1100_w220_I{i:04d}.npy" for i in ints])
#   for j in range(3):
    m1, lo1, hi1, m2, lo2, hi2 = load_sampled_interval_data(ints, 'cont', 3)
    ax[2].plot(ints, m1, '-', label='Support', color=col[0])
    ax[2].fill_between(ints, lo1, hi1, color='grey')
    ax[2].plot(ints, m2, ':', label='Against', color=col[1])
    ax[2].fill_between(ints, lo2, hi2, color='grey')


    m1, lo1, hi1, m2, lo2, hi2 = load_sampled_interval_data(ints, 'cult', 3)
    ax[3].plot(ints, m1, '-', label='Support', color=col[0])
    ax[3].fill_between(ints, lo1, hi1, color='grey')
    ax[3].plot(ints, m2, ':', label='Against', color=col[1])
    ax[3].fill_between(ints, lo2, hi2, color='grey')


    ax[1].set_ylabel("Fraction of significant results")
    ax[3].set_xticks(range(0, 2700, 200))
    ax[3].xaxis.set_tick_params(which='minor', bottom=True)
    ax[3].set_xlabel("Interval size (cents)")

    txt = ['All', 'Null?', 'Cont-Samp', 'Cult-Samp']
    for i, a in enumerate(ax[:4]):
        a.set_xlim(0, 2620)
        a.set_ylim(0, 0.47)
        set_ticks(a, 400, 100, '%d', 0.2, 0.1, '%3.1f')
        a.annotate(txt[i], (0.04, 0.82), xycoords='axes fraction')
    for a in ax[:3]:
        a.set_xticks([])

    ax[0].legend(loc='upper left', bbox_to_anchor=(0.00, 1.35), frameon=False, ncol=2)

    sns.scatterplot(x='mean_real', y='mean_shuf', data=res.sort_values(by='sig'), hue='sig', ax=ax[4])
    mx = max(res.mean_real.max(), res.mean_shuf.max())
    ax[4].plot([0, mx], [0, mx], '-k')
    ax[4].set_xlabel("Deviation of real intervals\nfrom the octave")
    ax[4].set_ylabel("Deviation of shuffled intervals\nfrom the octave")
    ax[4].legend(loc='best', frameon=False)

    octave_sig(ax[5])

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


def inst_notes(df, ysamp='scale'):
    df = df.loc[df.Reduced_scale=='N']
    fig, ax = plt.subplots(3,2, figsize=(9,3))
    fig.subplots_adjust(hspace=0, wspace=0.3)
    OC.get_int_prob_via_sampling(df, ysamp=ysamp, xsamp='', ax=ax[0,0])

    labels = ['Data', 'Lognormal fit', '99% CI']
    handles = [Line2D([], [], linestyle=ls, color=m, label=l) for ls, m, l in zip('-:', 'bk', labels[:2])] + \
              [Patch(facecolor='pink', label=labels[-1])]
    ax[0,0].legend(handles=handles, bbox_to_anchor=(1.70, 1.50), frameon=False, ncol=3)
    
    for i in range(3):
        OC.get_int_prob_via_sampling(df, ysamp=ysamp, xsamp='Continent', s=6, ax=ax[1,0]) 
        OC.get_int_prob_via_sampling(df, ysamp=ysamp, xsamp='Culture', s=1, ax=ax[2,0]) 

    lbls2 = ['Idiophone', 'Aerophone', 'Chordophone']
    for i, l in enumerate(lbls2):
        OC.get_int_prob_via_sampling(df.loc[df.Inst_type==l], ysamp=ysamp, xsamp='', ax=ax[i,1])
        ax[i,1].annotate(l, (0.70, 0.70), xycoords='axes fraction')
        
    txt = ["All", "Cont-Samp", "Cult-Samp"]
    for a in ax.ravel():
#       a.set_ylim(0, 0.002)
        a.set_xlim(0, 3000)
        set_ticks(a, 600, 200, '%d', 0.001, 0.0005, '%5.3f')
        a.set_ylabel('Density')
        a.tick_params(axis='both', which='major', direction='in', length=6, width=2)
        a.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
        a.set_yticks([0, 0.001])

    for i, a in enumerate(ax[:,0]):
        a.annotate(txt[i], (0.70, 0.70), xycoords='axes fraction')

    for a in ax[:2,:].ravel():
        a.set_xticks([])
    ax[2,0].set_xlabel('Interval from the lowest note / cents')
    ax[2,1].set_xlabel('Interval from the lowest note / cents')

    fig.savefig(PATH_FIG.joinpath(f"inst_notes_{ysamp}.pdf"), bbox_inches='tight')


def clustering(df):
    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(6,5, width_ratios=[0.7, 0.7, 1, 0.1, 1])
    ax = [fig.add_subplot(gs[:,0])] + \
         [fig.add_subplot(gs[j,i]) for i in [2, 4] for j in range(6)]


    df7 = df.loc[df.n_notes==7].reset_index(drop=True)
    s7 = np.array([[float(x) for x in y] for y in df7.scale])[:,1:-1]
    li = linkage(s7, method='ward')
    nc = fcluster(li, li[-6,2], criterion='distance')
    df7['nc6'] = nc

    ## Dendrogram
    thresh = li[-5, 2]
    cols = Paired_12.hex_colors[:6]
    set_link_color_palette(cols)
    idx_label = [137, 375,
                 22, 93, 357,
                 0, 209, 255, 429, 492,
                 98, 230,
                 323, 459,
                 2, 381, 439, 542]
#   print(df7.loc[idx_label, ['Name', 'nc6']])
    x_lbls = [name if i in idx_label else '' for i, name in enumerate(df7.Name)]
    dendrogram(li, labels=x_lbls, color_threshold=thresh, above_threshold_color='k', leaf_rotation=00, ax=ax[0], orientation='left')


    ## Scale and Continent dist

    cont = ['South East Asia', 'Africa', 'Oceania', 'South Asia', 'Western',
            'Latin America', 'East Asia', 'Middle East']
    cont_tot = Counter(df7['Continent'])
    Y_tot = np.array([cont_tot.get(c2,0) for c2 in cont])

    bins = np.arange(15, 1290, 30)
    X = bins[:-1] + np.diff(bins[:2])
    xbar = np.arange(len(cont))
    width = 0.3

    for i in range(6):
        c = 6 - i
        sns.distplot([x for y in df7.loc[nc==c, 'scale'] for x in y], bins=bins, ax=ax[i+1], kde=False, norm_hist=True, color=cols[c-1])
        count = Counter(df7.loc[nc==c, 'Continent'])
        Y = np.array([count.get(c2,0) for c2 in cont])
        ax[i+7].bar(xbar - width/2, Y, width, color=Dark2_8.hex_colors, ec='k')
        ax.append(ax[i+7].twinx())
        ax[i+13].bar(xbar + width/2, Y/Y_tot, width, color=Pastel2_8.hex_colors, ec='k')
        

    ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=10)
#   ax[0].spines['bottom'].set_visible(False)
    ax[0].set_xlabel("Distance between scale clusters")
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    for a in ax:
        a.spines['top'].set_visible(False)

    for a in ax[1:]:
        a.tick_params(axis='x', which='major', direction='in', length=6, width=2)
        a.tick_params(axis='x', which='minor', direction='in', length=4, width=1)

    for a in ax[1:7]:
        a.set_yticks([])
        a.set_ylabel("Density")
        a.spines['right'].set_visible(False)
        set_xticks(a, 200, 100, '%d')

    for a in ax[7:13]:
        a.set_ylabel("Frequency")

    for a in ax[1:6] + ax[13:18]:
        a.set_xticklabels([])

    for a in ax[13:]:
        a.set_ylabel("Relative\nFrequency")

    ax[6].set_xlabel("Scale note")
    ax[12].set_xlabel("Continent")
    ax[12].set_xticks(xbar)
    ax[12].set_xticklabels(cont, rotation=90)

    fig.savefig(PATH_FIG.joinpath(f"cluster7.pdf"), bbox_inches='tight')


def real_poss_dist(s7, ax):
    sdist = cdist(s7, s7)
    np.fill_diagonal(sdist, sdist.max())
    min_dist = np.load("../possible_7_20_60_320_md1.npy")
    sns.distplot(sdist.min(axis=0), kde=False, norm_hist=True, label='Real-Real', ax=ax, color='b')
    sns.distplot(min_dist, kde=False, norm_hist=True, label='Real-Grid', ax=ax, color='pink')
    yhi = ax.get_ylim()[1]
    ax.plot([50]*2, [0, yhi*0.8], ':k', alpha=0.5)
    ax.plot([100]*2, [0, yhi*0.8], ':k', alpha=0.5)


### Input: ascending list of notes in scale, minus the tonic and octave (0 and 1200 cents)
def get_mean_equid(scale):
    d = 0.
    n = len(scale) + 1
    for i, s in enumerate(scale):
        d += abs(s - (i + 1) * 1200 / n)
    return d / (n - 1)


def dist_equihept(scales):
    equi = np.arange(1, 7) * 1200 / 7.
    return cdist(scales, equi.reshape(1,6)).ravel()


def equidistance_diff(s7, ax):
    bins = np.arange(0, 610, 10)
    X = bins[:-1] + np.diff(bins[:2])
    hist = np.histogram(dist_equihept(s7), bins=bins, density=True)[0]
    ax.plot(X, hist, label='Real')

    path_list = ["possible_7_20_60_320_close50", "possible_7_20_60_320_far100", "possible_7_20_60_320"]
    lbls = ['Poss-Close', 'Poss-Far', 'Poss-All']
    for i, path in enumerate(path_list):
        path_hist = PATH_DATA.joinpath(f"{path}_hist.npy")
        if path_hist.exists() and 1:
            hist = np.load(path_hist)
        else:
            if not i:
                data = np.cumsum(np.load(f"../{path}.npy"), axis=1)[:,:-1]
            else:
                data = np.load(f"../{path}.npy")
            hist = np.histogram(dist_equihept(data), bins=bins, density=True)[0]
            np.save(path_hist, hist)

        ax.plot(X, hist, label=lbls[i])


def non_scales_diff(df, dx=20):
    fig = plt.figure(figsize=(12,5))
    gs = GridSpec(3,6, height_ratios=[1, 0.2, 1])
    ax = [fig.add_subplot(gs[2,j]) for j in range(6)] + \
         [fig.add_subplot(gs[0,i*2:(i+1)*2]) for i in [0,2,1]]

#   fig, ax = plt.subplots(1,6, figsize=(16,3))
    df7 = df.loc[df.n_notes==7].reset_index(drop=True)
    s7 = np.array([[float(x) for x in y] for y in df7.scale])[:,1:-1]

    # Distance between real and grid scales
    real_poss_dist(s7, ax[6])
    ax[6].set_xlim(-20, 330)


    # Distance between equidistant scales, and  real / grid scales
    equidistance_diff(s7, ax[7])
    ax[6].set_xlim(-20, 330)


    # Distributions for real scales, similar scales, and different scales
    far = np.load('../possible_7_20_60_320_far100.npy')
    close = np.load('../possible_7_20_60_320_close50.npy')

    lbls = ['Real', 'Grid-similar', 'Grid-different']
    xlbls = [[f"{a} {b}" for b in ['All', 'Min', 'Max']] for a in ["adj", "2nd", "3rd"]]
    for i, (d, l) in enumerate(zip([s7, close, far], lbls)):

        # Distributions of adjacent intervals,
        ints = np.diff(d, axis=1)
        xlo, xhi = np.min(ints), np.max(ints)
        xlo = dx * ((xlo // dx) - 1) - int(dx/2)
        xhi = dx * (3 + xhi // dx)
        bins = np.arange(xlo, xhi, dx)
        X = bins[:-1] + np.diff(bins[:2])
        ax[8].plot(X, np.histogram(ints, bins=bins, density=True)[0], label=lbls[i])

        # Distributions of each scale note,
        for j, s in enumerate(d.T):
            xlo, xhi = np.min(s), np.max(s)
            bins = np.arange(dx * ((xlo // dx) - 1), dx * (2 + xhi // dx), dx)
            X = bins[:-1] + np.diff(bins[:2])
            hist = np.histogram(s, bins=bins, density=True)[0]
            ax[j].plot(X, hist, label=lbls[i])
            ax[j].plot([(j+1)*1200/7]*2, [0, hist.max()], ':k', alpha=0.5)

            ax[j].set_xlabel(f"Note {j+2} / cents")
            ax[j].set_ylim(0, ax[j].get_ylim()[1])

    ax[0].set_ylabel("Density")
    ax[6].set_ylabel("Density")
    ax[7].set_ylabel("Density")
    ax[8].set_ylabel("Density")

    ax[6].set_xlabel("Distance to nearest scale / cents")
    ax[7].set_xlabel("Mean Note Distance from\nEquiheptatonic Scale / cents")
    ax[8].set_xlabel("Adjacent Interval / cents")

    ax[6].legend(loc='upper right', frameon=False)
    ax[7].legend(loc='upper right', frameon=False)
    ax[8].legend(loc='upper right', frameon=False)

    ax[7].set_xlim(0, 600)

    for a in ax[1:6]:
        a.spines['left'].set_visible(False)

    xmaj = [200, 200, 250, 350, 200, 200, 200, 100]
    for x, a in zip(xmaj, ax[:6] + ax[7:]):
        set_xticks(a, x, x/2, '%d')

    ax[2].set_xticklabels(['', 250, 500, 750, ''])

    for a in ax[:]:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_yticks([])

    fig.savefig(PATH_FIG.joinpath(f"nonscale.pdf"), bbox_inches='tight')

    


    




