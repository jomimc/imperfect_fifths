import os
import re
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
from itertools import permutations, product
import numpy as np
import pandas as pd
from palettable.colorbrewer.qualitative import Paired_12
import seaborn as sns
from scipy.optimize import curve_fit

import utils

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

DATA_DIR = '/home/johnmcbride/projects/Scales/Data_compare/Data_for_figs/'
FIGS_DIR = '/home/johnmcbride/projects/Scales/Data_compare/Figs/'
REAL_DIR = '/home/johnmcbride/projects/Scales/Data_compare/Processed/Real'

BIASES = ['none', 'S#1_n1', 'S#1_n2',#'',
          'distI_n1', 'distI_n2', 'distI_n3', 'distW',#'',
          'distI_n1_S#1_n1', 'distI_n1_S#1_n2', 'distI_n2_S#1_n1', 'distI_n2_S#1_n2',
          'distW_S#1_n1', 'distW_S#1_n2', 'distW_S#2_n2', 'distW_S#2_n3',
          'hs_n1_w05', 'hs_n1_w10', 'hs_n1_w15', 'hs_n1_w20',
          'hs_n2_w05', 'hs_n2_w10', 'hs_n2_w15', 'hs_n2_w20',
          'hs_n3_w05', 'hs_n3_w10', 'hs_n3_w15', 'hs_n3_w20',
          'hs_r3_w05', 'hs_r3_w10', 'hs_r3_w15', 'hs_r3_w20'] + \
         [f"im5_r{r:3.1f}_w{w:02d}" for r in [0, 0.5, 1, 2] for w in [5,10,15,20]] + \
         [f"Nim5_r0.0_w{w:02d}"  for w in [5,10,15,20]] + \
         [f"Nhs_n1_w{w:02d}"  for w in [5,10,15,20]] + \
         [f"Nhs_n2_w{w:02d}"  for w in [10,20]] + \
         [f"Nhs_n3_w{w:02d}"  for w in [10,20]]
          
BIAS_GROUPS = ['none', 'S#1', 'HS',
               'distW', 'distW_S#1', 'distW_S#2',
               'distI', 'distI_S#1']

BIAS_GROUPS = ['none', 'HS',
               'S#1', 'distW',
               'distW_S#1', 'distW_S#2',
               'distI', 'distI_S#1', 'im5', 'AHS']

groups = ['none'] + ['S#1']*2 + ['distI']*3 + ['distW'] + ['distI_S#1']*4 + \
         ['distW_S#1']*2 + ['distW_S#2']*2 + ['HS']*12 + ['im5']*24 + ['HS']*8
BIAS_KEY = {BIASES[i]:groups[i] for i in range(len(BIASES))}


def plot_MC_dist(fName, X='pair_ints', out=False, f=False, cum=False):
    df = pd.read_feather(fName)
    if f:
        sns.distplot(df[X], bins=100)
    else:
        if cum:
            sns.distplot(utils.extract_floats_from_string(df[X]), bins=100, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
        else:
            sns.distplot(utils.extract_floats_from_string(df[X]), bins=100)
    if out:
        return df

def plot_MC_kde(fName, X='pair_ints', out=False, f=False, ax='None'):
    df = pd.read_feather(fName)
    if f:
        sns.kdeplot(df[X])
    else:
        sns.kdeplot(utils.extract_floats_from_string(df[X]))
    if out:
        return df

def rename_biases(df):
    df.loc[df.bias=='distI_1_0', 'bias'] = 'distI_n1'
    df.loc[df.bias=='distI_2_0', 'bias'] = 'distI_n2'
    df.loc[df.bias=='distI_3_0', 'bias'] = 'distI_n3'
    df.loc[df.bias=='distI_0_1', 'bias'] = 'S#1_n1'
    df.loc[df.bias=='distI_0_2', 'bias'] = 'S#1_n2'
    df.loc[df.bias=='distI_1_1', 'bias'] = 'distI_n1_S#1_n1'
    df.loc[df.bias=='distI_1_2', 'bias'] = 'distI_n1_S#1_n2'
    df.loc[df.bias=='distI_2_1', 'bias'] = 'distI_n2_S#1_n1'
    df.loc[df.bias=='distI_2_2', 'bias'] = 'distI_n2_S#1_n2'
    df.loc[df.bias=='opt_c', 'bias'] = 'distW'
    df.loc[df.bias=='opt_c_I1', 'bias'] = 'distW_S#1_n1'
    df.loc[df.bias=='opt_c_I2', 'bias'] = 'distW_S#1_n2'
    df.loc[df.bias=='opt_c_s2', 'bias'] = 'distW_S#2_n2'
    df.loc[df.bias=='opt_c_s3', 'bias'] = 'distW_S#2_n3'
    return df

def rename_bias_groups(df):
    df.loc[df.bias_group=='distI+small', 'bias_group'] = 'distI_S#1'
    df.loc[df.bias_group=='distW+I', 'bias_group'] = 'distW_S#1'
    df.loc[df.bias_group=='distW+S', 'bias_group'] = 'distW_S#2'
    df.loc[df.bias_group=='small', 'bias_group'] = 'S#1'
    df.loc[df.bias_group=='hs', 'bias_group'] = 'HS'
    return df

def plot_violin(df, cat='pair_ints', X='bias_group', Y='JSD', kind='violin'):
    df = rename_bias_groups(df)
    violin_order = get_violin_order(df, X, Y)
    sns.catplot(x=X, y=Y, data=df.loc[df.cat==cat], kind=kind, order=violin_order)
#   sns.catplot(x=X, y=Y, data=df.loc[df.cat==cat], kind='violin', order=[0.0, 50.0, 60., 70., 80., 90., 100.])
#   sns.catplot(x=X, y=Y, data=df.loc[df.cat==cat], kind='boxen', order=[0.0, 50.0, 60., 70., 80., 90., 100.])
#   sns.catplot(x=X, y=Y, data=df.loc[df.cat==cat], kind='boxen', order=[400., 450., 500., 550., 1200.])

def get_violin_order(df, X, Y):
    groups = np.array(df[X].unique())
    min_J = [df.loc[(df[X]==g)&(df.cat=='pair_ints'),Y].min() for g in groups]
    if 'fr' in Y:
        violin_order = groups[np.argsort(min_J)[::-1]]
    else:
        violin_order = groups[np.argsort(min_J)]
    return violin_order


def df_distplot_with_constraints(df, bias, MI, MA, q, cat='pair_ints', ret=0):
    if 'hs' in bias:
        cut = df.loc[(df.min_int>MI)&(df.max_int<MA), bias].quantile(1.-q)
        print(cut)
        tmp_df = df.loc[(df[bias]>cut)&(df.min_int>MI)&(df.max_int<MA)]
        sns.distplot(utils.extract_floats_from_string(tmp_df.loc[:,cat]), bins=100, label=bias)
    else:
        cut = df.loc[(df.min_int>MI)&(df.max_int<MA), bias].quantile(q)
        tmp_df = df.loc[(df[bias]<cut)&(df.min_int>MI)&(df.max_int<MA)]
        sns.distplot(utils.extract_floats_from_string(tmp_df.loc[:,cat]), bins=100, label=bias)
    plt.legend(loc='best')
    if ret:
        return tmp_df

def get_files_and_labels_from_idx(df, idx, kde=True, hist=False):
    fNames = []
    labels = []
    for i in idx:
        if kde:
            fNames.append(df.loc[i, 'kde_path'])
            labels.append("kde: int=[{0[0]}-{0[1]}]; beta={0[2]}".format(df.loc[i, ['min_int', 'max_int', 'beta']]))
        if hist:
            fNames.append(df.loc[i, 'hist_path'])
            labels.append("hist: int=[{0[0]}-{0[1]}]; beta={0[2]}".format(df.loc[i, ['min_int', 'max_int', 'beta']]))
    return fNames, labels

def plot_harmonic_similarity_distributions(df_grid, df_real, cat='Continent', leg=True, n=5, m=1):
    fig, ax = plt.subplots(4,1)
    ax = ax.reshape(ax.size)
#   fig, ax = plt.subplots(1,1)
#   ax = [ax]

#   plt.subplots_adjust(hspace=0.8)
    for i, lbl in enumerate([f'hs_n{m}_w{x:02d}' for x in range(5,25,5)]):
#   for i, lbl in enumerate([f'hs_n{m}_w{x:02d}' for x in range(10,15,5)]):
        sns.distplot(df_grid[lbl], label='no_constraint', ax=ax[i], color='k')
        for c in df_real[cat].unique():
            sns.kdeplot(df_real.loc[(df_real[cat]==c)&(df_real.n_notes==n), lbl], ax=ax[i], label=c)
        if leg and not i:
#           ax[i].legend(loc='best', frameon=False)
#           ax[i].legend(bbox_to_anchor=(0.39, 1.3), frameon=False, ncol=7)
            handles, labels = ax[i].get_legend_handles_labels()
            ax[i].get_legend().set_visible(False)
        else:
            ax[i].get_legend().set_visible(False)
        ax[i].set_ylabel('Prob')
    if leg:
        fig.legend(handles, labels, loc='upper center', frameon=False, ncol=4)
    fig.savefig(f"{FIGS_DIR}/harm_sim_dist_notes{n}_ver{m}.png")
    fig.savefig(f"{FIGS_DIR}/harm_sim_dist_notes{n}_ver{m}.pdf")

def plot_dists_by_npy_file(files, labels, real=True, kde=True, hist=False, n=7):
    fig, ax = plt.subplots()
    if hist or sum([1 for f in files if 'hist' in f]):
        ax2 = ax.twinx()
    if real:
        if kde:
            data = np.load(os.path.join(REAL_DIR, f"n_{n}_kde.npy"))
            ax.plot(data[:,0], data[:,1], label='real_kde')
        if hist:
            data = np.load(os.path.join(REAL_DIR, f"n_{n}_hist.npy"))
            ax2.plot(data[:,0], data[:,1], label='real_hist')

    for i, f in enumerate(files):
        data = np.load(f)
        if 'hist' in f:
            ax2.plot(data[:,0], data[:,1], label=labels[i])
        else:
            ax.plot(data[:,0], data[:,1], label=labels[i])
    ax.legend(loc='best')
    plt.show()

def set_xticks(ax, xMaj, xMin, xForm):
  ax.xaxis.set_major_locator(MultipleLocator(xMaj))
  ax.xaxis.set_major_formatter(FormatStrFormatter(xForm))
  ax.xaxis.set_minor_locator(MultipleLocator(xMin))

def plot_dist_by_cat(df, X='scale', cat='Continent', lim=(-5,1250), bins=120):
    uni_cat = np.array(sorted(df.loc[:,cat].unique()))
    if cat=='n_notes':
        uni_cat = np.array([4,5,6,7,8,9])

    n_cat = uni_cat.size
    if n_cat <=6:
        fig, ax = plt.subplots(3,2, sharex=True)
    elif n_cat <=12:
        fig, ax = plt.subplots(4,3, sharex=True)
    else:
        print(n_cat, ' too large')
    fig2, ax2 = plt.subplots()
    ax = ax.reshape(ax.size)
    for i, uni in enumerate(uni_cat):
        idx = df.loc[df.loc[:,cat]==uni,:].index
        if not isinstance(df.loc[idx[0],X], str):#
            Xarr = df.loc[idx,X]
            Xarr2 = [a for a in df.loc[idx,X] if 0<a<1200]
        else:
            Xarr = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a]
            Xarr2 = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a if 0<b<1200]
        sns.distplot(Xarr, bins=bins, label=str(uni), ax=ax[i])
        sns.kdeplot(Xarr2, label=str(uni), ax=ax2)
        ax[i].legend(loc='best')
    ax2.legend(loc='best')

    ticks = np.arange(0, (int(lim[1]/100)+1)*100, 100)
    ax[-1].set_xlim(lim)
#   ax[-1].xaxis.set_ticks(np.arange(0,1300,100))
    ax2.set_xlim(lim)
#   ax2.xaxis.set_ticks(np.arange(0,1300,100))
    plt.show()

def plot_score_histograms(df):
    fig, ax = plt.subplots()
    uni_cat = np.array([4,5,6,7,8,9])
    for n in uni_cat:
        sns.distplot(df.loc[df.n_notes==n, 'harm_sim'], label=str(n), kde=True, bins=40, ax=ax)
    ax.legend(loc='best')
    plt.show()
    

# This was used for creating a figure for my CSLM seminar
def plot_similar_cultures(df, X='scale', cat='Continent', lim=(-5,1250)):
    groups = [  ['Western', 'East Asia', 'South Asia'],
                ['Western', 'Oceania'],
                ['Western', 'South America'],
                ['South East Asia', 'Africa'],
                ['Western', 'Middle East']]

    fig, ax = plt.subplots(3,2, sharex=True)
    plt.subplots_adjust(wspace=0.3, hspace=0.2)
    ax = ax.reshape(ax.size)
    extra_ax = []
    for i, group in enumerate(groups):
#       idx = df.loc[df.loc[:,cat].apply(lambda x: x in uni),:].index
        for j, uni in enumerate(group):
            idx = df.loc[df.loc[:,cat]==uni,:].index
            Xarr = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a]
            sns.distplot(Xarr, bins=120, ax=ax[i], label=str(uni), kde=False, norm_hist=True)
        ax[i].legend(loc='best', frameon=False)

    ax[0].set_ylabel('Probability')
    ax[2].set_ylabel('Probability')
    ax[4].set_ylabel('Probability')

    ax[4].set_xlabel('Notes in scale (cents)')
    ax[5].set_xlabel('Notes in scale (cents)')

#   ticks = np.arange(0, (int(lim[1]/100)+1)*100, 400)
#   ax[-1].xaxis.set_ticks(ticks)
    ax[-1].set_xlim(lim)
    set_xticks(ax[-1], 200, 100, '%d')
#   plt.savefig('Figs/culture_scale_comparison.png')
    plt.show()

# This was used for creating a figure for my paper
def plot_similar_cultures_2(df, X='scale', cat='Continent', lim=(-5,1250)):
    groups = [  [], ['Western', 'East Asia', 'South Asia', 'Middle East'],
                ['Oceania', 'South America', 'South East Asia', 'Africa']]

    fig, ax = plt.subplots(3,1, sharex=True)
    fig2, ax2 = plt.subplots(8,1, sharex=True)
    plt.subplots_adjust(wspace=0.3, hspace=0.2)
    ax = ax.reshape(ax.size)
    ax2 = ax2.reshape(ax2.size)
    extra_ax = []
    lbls = ['All', 'Theory', 'Instrument']
    cols = sns.color_palette('colorblind')
    for i, group in enumerate(groups):
#       idx = df.loc[df.loc[:,cat].apply(lambda x: x in uni),:].index
        if i:
            for j, uni in enumerate(group):
                idx = df.loc[df.loc[:,cat]==uni,:].index
                Xarr = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a]
                sns.distplot(Xarr, bins=120, ax=ax2[j+(i-1)*4], label=f"{str(uni):15s} N={len(idx)}", kde=False, norm_hist=True, color=cols[j+(i-1)*4])
                sns.kdeplot(Xarr, ax=ax[i], label=f"{str(uni):15s} N={len(idx)}", clip=(5, 1150), color=cols[j+(i-1)*4])
                ax2[j+(i-1)*4].legend(loc='upper right', frameon=False)
        else:
            for j, g in enumerate(groups[:]):
                if j:
                    idx = df.loc[df.loc[:,cat].apply(lambda x: x in g),:].index
                else:
                    idx = df.index
                Xarr = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a]
#               sns.distplot(Xarr, bins=120, ax=ax[i], label=lbls[j], kde=False, norm_hist=True)
                sns.kdeplot(Xarr, ax=ax[i], label=f"{lbls[j]:15s} N={len(idx)}", clip=(5, 1150))
        ax[i].legend(loc='best', frameon=False)

    ax[0].set_ylabel('Probability')
    ax[1].set_ylabel('Probability')
    ax[2].set_ylabel('Probability')

    ax[2].set_xlabel('Intervals size (cents)')
#   ax[5].set_xlabel('Notes in scale (cents)')

#   ticks = np.arange(0, (int(lim[1]/100)+1)*100, 400)
#   ax[-1].xaxis.set_ticks(ticks)
    ax[-1].set_xlim(lim)
    set_xticks(ax[-1], 200, 100, '%d')
#   plt.savefig('Figs/culture_scale_comparison.png')
    fig.savefig(os.path.join(FIGS_DIR, 'database_intervals_kde.png'))
    fig.savefig(os.path.join(FIGS_DIR, 'database_intervals_kde.pdf'))
    fig2.savefig(os.path.join(FIGS_DIR, 'database_intervals_hist.png'))
    fig2.savefig(os.path.join(FIGS_DIR, 'database_intervals_hist.pdf'))
    plt.show()

# This was used for creating a figure for my CSLM seminar
def plot_comparison_ints_by_n(df, X='pair_ints', cat='n_notes', lim=(-5, 605)):
    uni_cat = np.array([4,5,6,7,8,9])
    fig2, ax2 = plt.subplots(3,2, sharex=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    ax2 = ax2.reshape(ax2.size)[[0,2,4,1,3,5]]
    ticks = np.arange(0, (int(lim[1]/100)+1)*100, 100)
    ax2[-1].xaxis.set_ticks(ticks)
    ax2[-1].set_xlim(lim)
    ax2[0].set_ylabel('Probability')
    ax2[1].set_ylabel('Probability')
    ax2[2].set_ylabel('Probability')
    ax2[2].set_xlabel('Interval size (cents)')
    ax2[5].set_xlabel('Interval size (cents)')
    for i, uni in enumerate(uni_cat):
        idx = df.loc[df.loc[:,cat]==uni,:].index
        Xarr = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a]
        sns.distplot(Xarr, bins=120, label="my dataset", ax=ax2[i])
        ax2[i].set_title("N = {0}".format(uni))
    ax2[5].legend(loc='best', frameon=False)

    plt.savefig('Figs/data_set_intervals.png')
    plt.show()

    fig, ax = plt.subplots(3,2, sharex=True)
    ax = ax.reshape(ax.size)[[0,2,4,1,3,5]]
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    ax[-1].set_xlim(lim)
    ax[-1].xaxis.set_ticks(ticks)
    ax[0].set_ylabel('Probability')
    ax[1].set_ylabel('Probability')
    ax[2].set_ylabel('Probability')
    ax[2].set_xlabel('Interval size (cents)')
    ax[5].set_xlabel('Interval size (cents)')

    for i, uni in enumerate(uni_cat):
        idx = df.loc[df.loc[:,cat]==uni,:].index
        Xarr = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a]
#       sns.kdeplot(Xarr, label="my dataset", ax=ax[i])
        sns.kdeplot(Xarr, ax=ax[i])
        ax[i].set_title("N = {0}".format(uni))
#   ax[5].legend(loc='upper right', frameon=False)

    for i, n in enumerate(uni_cat):
        data = np.load("Data_for_figs/unrestricted_ints_hist_n{0}.npy".format(n))
        ax[i].plot(data[:,0], data[:,1], '-', label="any intervals")
    ax[5].legend(loc='best', frameon=False)

    plt.savefig('Figs/data_model_comparison_1.png')

    for i, n in enumerate(uni_cat):
        data = np.load("Data_for_figs/restricted_ints_hist_n{0}.npy".format(n))
        ax[i].plot(data[:,0], data[:,1], '-', label="constrained")
    ax[5].legend(loc='best', frameon=False)

    plt.savefig('Figs/data_model_comparison_2.png')

    plt.show()

# This was used for creating a figure for my CSLM seminar
def plot_comparison_ints_by_n_bias(df, X='pair_ints', cat='n_notes', lim=(-5, 605)):
    uni_cat = np.array([4,5,6,7,8,9])
    fig, ax = plt.subplots(3,2, sharex=True, sharey=True)
    ax = ax.reshape(ax.size)[[0,2,4,1,3,5]]
    for i, uni in enumerate(uni_cat):
        idx = df.loc[df.loc[:,cat]==uni,:].index
        Xarr = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a]
        sns.distplot(Xarr, bins=120, label="my dataset", ax=ax[i])
        ax[i].set_title("N = {0}".format(uni))
    ax[5].legend(loc='best', frameon=False)

    for i, n in enumerate(uni_cat):
        data = np.load("Data_for_figs/biased_ints_hist_n{0}.npy".format(n))
        ax[i].plot(data[:,0], data[:,1], '-', label="bias model")
    ax[5].legend(loc='best', frameon=False)

    ticks = np.arange(0, (int(lim[1]/100)+1)*100, 100)
    ax[0].set_ylabel('Probability')
    ax[1].set_ylabel('Probability')
    ax[2].set_ylabel('Probability')
    ax[2].set_xlabel('Interval size (cents)')
    ax[5].set_xlabel('Interval size (cents)')
    ax[-1].set_xlim(lim)
    ax[-1].set_ylim(0,0.015)
#   ax[-1].xaxis.set_ticks(np.arange(0,1300,100))
    plt.savefig('Figs/data_model_comparison_3.png')
    plt.show()

def plot_comparison_scale_by_n(df, X='scale', cat='n_notes', lim=(-5,1250)):
    uni_cat = np.array([4,5,6,7,8,9])
    fig2, ax2 = plt.subplots(3,2, sharex=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    ax2 = ax2.reshape(ax2.size)[[0,2,4,1,3,5]]
    ticks = np.arange(0, (int(lim[1]/100)+1)*100, 100)
    ax2[-1].xaxis.set_ticks(ticks)
    ax2[-1].set_xlim(lim)
    ax2[0].set_ylabel('Probability')
    ax2[1].set_ylabel('Probability')
    ax2[2].set_ylabel('Probability')
    ax2[2].set_xlabel('Notes in scale (cents)')
    ax2[5].set_xlabel('Notes in scale (cents)')
    for i, uni in enumerate(uni_cat):
        idx = df.loc[df.loc[:,cat]==uni,:].index
        Xarr = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a]
        sns.distplot(Xarr, bins=120, label="my dataset", ax=ax2[i])
        ax2[i].set_title("N = {0}".format(uni))
    ax2[5].legend(loc='best', frameon=False)

    plt.savefig('Figs/data_set_intervals.png')
    plt.show()

    fig, ax = plt.subplots(3,2, sharex=True, sharey=True)
    ax = ax.reshape(ax.size)[[0,2,4,1,3,5]]
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    ax[-1].set_xlim(lim)
    ax[-1].xaxis.set_ticks(ticks)
    ax[-1].set_ylim(0,0.005)
    ax[0].set_ylabel('Probability')
    ax[1].set_ylabel('Probability')
    ax[2].set_ylabel('Probability')
    ax[2].set_xlabel('Notes in scale (cents)')
    ax[5].set_xlabel('Notes in scale (cents)')

    for i, uni in enumerate(uni_cat):
        idx = df.loc[df.loc[:,cat]==uni,:].index
        Xarr = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a]
#       sns.kdeplot(Xarr, label="my dataset", ax=ax[i])
        sns.distplot(Xarr, ax=ax[i], bins=120)
        ax[i].set_title("N = {0}".format(uni))
#   ax[5].legend(loc='upper right', frameon=False)

    for i, n in enumerate(uni_cat):
        data = np.load("Data_for_figs/unrestricted_scale_hist_n{0}.npy".format(n))
        ax[i].plot(data[:,0], data[:,1], '-', label="any intervals")
    ax[5].legend(loc='best', frameon=False)

    plt.savefig('Figs/data_model_scale_comparison_1.png')

    for i, n in enumerate(uni_cat):
        data = np.load("Data_for_figs/restricted_scale_hist_n{0}.npy".format(n))
        ax[i].plot(data[:,0], data[:,1], '-', label="constrained")
    ax[5].legend(loc='best', frameon=False)

    plt.savefig('Figs/data_model_scale_comparison_2.png')

    plt.show()

def plot_comparison_scale_by_n_bias(df, X='scale', cat='n_notes', lim=(-5,1250)):
    uni_cat = np.array([4,5,6,7,8,9])
    fig, ax = plt.subplots(3,2, sharex=True, sharey=True)
    ax = ax.reshape(ax.size)[[0,2,4,1,3,5]]
    for i, uni in enumerate(uni_cat):
        idx = df.loc[df.loc[:,cat]==uni,:].index
        Xarr = [b for a in df.loc[idx,X].apply(lambda x: [float(y) for y in x.split(';')]) for b in a]
        sns.distplot(Xarr, bins=120, label="my dataset", ax=ax[i])
        ax[i].set_title("N = {0}".format(uni))
    ax[5].legend(loc='best', frameon=False)

    for i, n in enumerate(uni_cat):
        data = np.load("Data_for_figs/biased_scale_hist_n{0}.npy".format(n))
        ax[i].plot(data[:,0], data[:,1], '-', label="bias model")
    ax[5].legend(loc='best', frameon=False)

    ticks = np.arange(0, (int(lim[1]/100)+1)*100, 100)
    ax[0].set_ylabel('Probability')
    ax[1].set_ylabel('Probability')
    ax[2].set_ylabel('Probability')
    ax[2].set_xlabel('Notes in scale (cents)')
    ax[5].set_xlabel('Notes in scale (cents)')
    ax[-1].set_xlim(lim)
    ax[-1].set_ylim(0,0.005)
#   ax[-1].xaxis.set_ticks(np.arange(0,1300,100))
    plt.savefig('Figs/data_model_scale_comparison_3.png')
    plt.show()

def subplot_x_y(n):
    if n == 1:
        return [1]*2
    elif n**0.5 == int(n**0.5):
        return [int(n**0.5)]*2
    else:
        x = int(n**0.5)
        y = x + 1
        switch = 0
        while n > x*y:
            if switch:
                x += 1
                switch = 0
            else:
                y += 1
                switch = 1
    return y, x

def plot_best_pair_dist(df_best, df_m, df_real, X='pair_ints', n=7):
    sub_y, sub_x = subplot_x_y(len(df_best))
    fig, ax = plt.subplots(sub_y, sub_x, sharex=True, sharey=True)
    try:
        ax = ax.reshape(ax.size)
    except:
        ax = [ax]
    for i in range(len(ax)):
        sns.distplot(utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n, X]), bins=100, ax=ax[i])
        df = pd.read_feather(df_m.loc[df_best.loc[i, f"idx_{n}"], 'fName'] )
        sns.distplot(utils.extract_floats_from_string(df[X]), bins=100, ax=ax[i])
#       ax[i].set_title(df_best.loc[i, 'bias'])

def simple_fit(X, Y, fit_fn='None'):
    min_idx = np.argmin(X)
    max_idx = np.argmax(X)
    dX = X[max_idx] - X[min_idx]
    dY = Y[max_idx] - Y[min_idx]
    if fit_fn == 'None':
        fit_fn = lambda x, m, a:  m*x + a
        popt, pcov = curve_fit(fit_fn, X, Y, p0=[dY/dX, Y[max_idx]])
    else:
        popt, pcov = curve_fit(fit_fn, X, Y, p0=[dY**2/dX**2, dY/dX, Y[max_idx]])
    xnew = np.linspace(X[min_idx], X[max_idx], 10)
    ynew = fit_fn(xnew, *popt)
    return xnew, ynew, popt

def plot_JSD_vs_scales(df, X='JSD', Y='fr_20', bias_group='HS', n=5, fit=False):
    df = rename_bias_groups(df)
    df = rename_biases(df)
    biases = [b for b in BIASES if BIAS_KEY[b]==bias_group]
    sub_y, sub_x = subplot_x_y(len(biases))
    sub_y, sub_x = 6,4
    fig, ax = plt.subplots(sub_y, sub_x, sharex=True, sharey=True)
    try:
        ax = ax.reshape(ax.size)
    except:
        ax = [ax]
    for i, bias in enumerate(biases):
        if not len(bias):
            continue
        if n:
            sns.scatterplot(x=X, y=Y, data=df.loc[(df.n_notes==n)&(df.bias_group==bias_group)], ax=ax[i], alpha=0.5)
            sns.scatterplot(x=X, y=Y, data=df.loc[(df.n_notes==n)&(df.bias==bias)], ax=ax[i])
            if fit:
                x_fit, y_fit, popt = simple_fit(df.loc[(df.n_notes==n)&(df.bias==bias), X], df.loc[(df.n_notes==n)&(df.bias==bias), Y])
                ax[i].plot(x_fit, y_fit)
                ax[i].text(0.2, .20, f"m={popt[0]:7.5f}", transform=ax[i].transAxes)
        else:
            sns.scatterplot(x=X, y=Y, data=df, ax=ax[i], alpha=0.5)
            sns.scatterplot(x=X, y=Y, data=df.loc[(df.bias==bias)], ax=ax[i])
        print(bias)
        ax[i].set_title(''.join(bias.split('_')))

def plot_JSD_vs_scales_bias_group(df, X='JSD', Y='fr_20', save=False, n=5):
    df = rename_bias_groups(df)
    df = rename_biases(df)
    fig, ax = plt.subplots(4,3, sharex=True, sharey=True, figsize=(10,24))
    plt.subplots_adjust(hspace=0.30) #wspace=0.3, hspace=0.2)
    ax = ax.reshape(ax.size)
    if 'cat' in df.columns:
        df = df.loc[df.cat=='pair_ints']
    for i, bias in enumerate(BIAS_GROUPS):
        if n:
            sns.scatterplot(x=X, y=Y, data=df.loc[df.n_notes==n], ax=ax[i], alpha=0.5)
            sns.scatterplot(x=X, y=Y, data=df.loc[(df.n_notes==n)&(df.bias_group==bias)], ax=ax[i])
        else:
            sns.scatterplot(x=X, y=Y, data=df, ax=ax[i], alpha=0.5)
            sns.scatterplot(x=X, y=Y, data=df.loc[(df.bias_group==bias)], ax=ax[i])
        ax[i].set_title(bias)
#       if i%2 == 0:
#           ax[i].set_ylabel(r'$f_{real}$')
#   plt.tight_layout()
#   ax[0].set_xlim(df[X].min()*0.8, df[X].max()*1.2)
#   ax[0].set_ylim(df[Y].min()*0.8, df[Y].max()*1.2)
    if save:
        plt.savefig(FIGS_DIR + 'model_comparison.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(FIGS_DIR + 'model_comparison.png')

def plot_scale_histograms(df, df_real, i, nbin=100, X='scale', neg=1.0):
    fig, ax = plt.subplots()
    n = df.loc[i, 'n_notes']
    bins = np.linspace(0, 1200, num=nbin+1)
    df_model = pd.read_feather(df.loc[i, 'fName'])
    histM, bins = np.histogram(utils.extract_floats_from_string(df_model.loc[:,X]), bins=bins, normed=True)
    histR, bins = np.histogram(utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n,X]), bins=bins, normed=True)
    xxx = bins[:-1] + 0.5 * (bins[1] - bins[0])
    df_hist = pd.DataFrame(data={'bin':xxx, 'real':histR, 'model':histM*neg})
    sns.lineplot(x='bin', y='real', data=df_hist)
    sns.lineplot(x='bin', y='model', data=df_hist)

def plot_scale_histograms_compare(df, df_real, i, j, nbin=100, X='scale', neg=1.0, partabase='none', mix=[0,0]):
    fig, ax = plt.subplots()
    n = df.loc[i, 'n_notes']
    bins = np.linspace(0, 1200, num=nbin+1)
    if partabase=='none':
        histR, bins = np.histogram(utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n,X]), bins=bins, normed=True)
    elif partabase=='theory':
        histR, bins = np.histogram(utils.extract_floats_from_string(df_real.loc[(df_real.n_notes==n)&(df_real.Theory=='Y'),X]), bins=bins, normed=True)
    elif partabase=='instrument':
        histR, bins = np.histogram(utils.extract_floats_from_string(df_real.loc[(df_real.n_notes==n)&(df_real.Theory=='N'),X]), bins=bins, normed=True)
    xxx = bins[:-1] + 0.5 * (bins[1] - bins[0])
    df_hist = pd.DataFrame(data={'bin':xxx, 'real':histR})
    sns.lineplot(x='bin', y='real', data=df_hist)
    for count, k in enumerate([i, j]):
        df_model = pd.read_feather(df.loc[k, 'fName'])
        if mix[count]:
            X2 = {'pair_ints':'mix_ints', 'scale':'mix_scale'}[X]
            histM, bins = np.histogram(utils.extract_floats_from_string(df_model.loc[:,X2]), bins=bins, normed=True)
        else:
            histM, bins = np.histogram(utils.extract_floats_from_string(df_model.loc[:,X]), bins=bins, normed=True)
        df_hist = pd.DataFrame(data={'bin':xxx, 'real':histR, f'model_{count+1}':histM*neg})
        sns.lineplot(x='bin', y=f'model_{count+1}', data=df_hist, label=f'model_{count+1}')
    ax.legend()

def plot_JSD_fr_against_input_variables(df, var, cat='pair_ints', w=20):
    varArr = sorted(df[var].unique())
    fig, ax = plt.subplots(2,1)
    sns.set_style('darkgrid')

#   for bias in df.bias.unique():
#       JSD = [df.loc[(df.cat==cat)&(df[var]==v)&(df.bias==bias),'JSD'].mean() for v in varArr]
#       JSDerr = [df.loc[(df.cat==cat)&(df[var]==v)&(df.bias==bias),'JSD'].std() for v in varArr]
#       FR = [df.loc[(df.cat==cat)&(df[var]==v)&(df.bias==bias),f"fr_{w}"].mean() for v in varArr]
#       FRerr = [df.loc[(df.cat==cat)&(df[var]==v)&(df.bias==bias),f"fr_{w}"].std() for v in varArr]
#       ax[0].plot(varArr, JSD)# min(JSD))
#       ax[1].plot(varArr, FR)# min(FR))
#       JSDmin = [df.loc[(df.cat==cat)&(df[var]==v)&(df.bias==bias),'JSD'].min() for v in varArr]
#       FRmax = [df.loc[(df.cat==cat)&(df[var]==v)&(df.bias==bias),f"fr_{w}"].max() for v in varArr]
#       ax[0].plot(varArr, JSDmin, label='min')
#       ax[1].plot(varArr, FRmax, label='max')
    ax[0].set_ylabel('JSD')
    ax[1].set_ylabel('frac_scales')
    ax[1].set_xlabel(var)

#   JSDmin = [df.loc[(df.cat==cat)&(df[var]==v),'JSD'].min() for v in varArr]
#   FRmax = [df.loc[(df.cat==cat)&(df[var]==v),f"fr_{w}"].max() for v in varArr]
    JSDmean = [df.loc[(df.cat==cat)&(df[var]==v),'JSD'].mean() for v in varArr]
    FRmean = [df.loc[(df.cat==cat)&(df[var]==v),f"fr_{w}"].mean() for v in varArr]
    ax[0].plot(varArr, JSDmean, label='mean')
    ax[1].plot(varArr, FRmean, label='mean')
#   ax[0].plot(varArr, JSDmin, label='min')
#   ax[1].plot(varArr, FRmax, label='max')

#   plt.
    plt.show()

def plot_best_variables_cumulative(df, var, cat='pair_ints', w=20):
    df_J = df.loc[df.cat==cat].sort_values(by='JSD')
    df_f = df.loc[df.cat==cat].sort_values(by=f"fr_{w:02d}", ascending=False)
    X = range(10, len(df_J))
    Y_J = [df_J.loc[:,var].iloc[:i].mean() for i in X]
    Y_f = [df_f.loc[:,var].iloc[:i].mean() for i in X]

    fig, ax = plt.subplots(2,1)
    sns.set_style('darkgrid')

    ax[0].plot(X, Y_J, label='JSD')
    ax[1].plot(X, Y_f, label='frac_scales')
        
    ax[0].set_ylabel('JSD')
    ax[1].set_ylabel('frac_scales')
    ax[1].set_xlabel('averaged over top scoring N models')

#   ax[0].set_xscale('log')
#   ax[1].set_xscale('log')

    plt.show()
        
def plot_best_constraints_cumulative(df, var, cat='pair_ints', w=20, n=10):
    if var == 'min_int':
        var_arr = np.arange(0., 110., 10.)
        dV = 10
    elif var == 'max_int':
        var_arr = np.arange(400., 1300., 50.)
        dV = 50
    df_J = df.loc[df.cat==cat].sort_values(by='JSD').reset_index(drop=True)
    df_f = df.loc[df.cat==cat].sort_values(by=f"fr_{w:02d}", ascending=False).reset_index(drop=True)
    X = np.arange(0, len(df_J), n)

#   df_J['N'] = np.array(df_J.index / n, dtype=int)
#   df_f['N'] = np.array(df_J.index / n, dtype=int)

    xi, yi = np.meshgrid(X[1:], var_arr)
    distJ = np.zeros((X.size-1, var_arr.size), dtype=int)
    distF = np.zeros((X.size-1, var_arr.size), dtype=int)
    for i, x, in enumerate(X[1:]):
        for y in df_J.loc[X[i]:x-1,var]:
            distJ[i, int(y/dV - var_arr[0]/dV)] += 1
        for y in df_f.loc[X[i]:x-1,var]:
            distF[i, int(y/dV - var_arr[0]/dV)] += 1

    fig, ax = plt.subplots(2,1)
    sns.set_style('darkgrid')

    cs1 = ax[0].contourf(xi, yi, distJ.T)
    cs2 = ax[1].contourf(xi, yi, distF.T)

    ax[0].set_ylabel(f'{var}')
    ax[1].set_ylabel(f'{var}')
    ax[0].set_xlabel(f'{var} histogram, sorted by JSD')
    ax[1].set_xlabel(f'{var} histogram, sorted by frac_scales')

#   ax[0].set_xscale('log')
#   ax[1].set_xscale('log')

    fig.colorbar(cs1, ax=ax[0])
    fig.colorbar(cs2, ax=ax[1])

    plt.show()

def plot_bias_performance_ranked(df, cat='pair_ints', w=20, n=10):
    var = 'bias_group'
    df = rename_bias_groups(df)
    biases = np.array(['none', 'HS', 'S#1', 'distW', 'distI', 'distI_S#1', 'distW_S#1', 'distW_S#2'])
    bias_dict = {biases[i]:i for i in range(len(biases))}
    df_J = df.loc[df.cat==cat].sort_values(by='JSD').reset_index(drop=True)
    df_f = df.loc[df.cat==cat].sort_values(by=f"fr_{w:02d}", ascending=False).reset_index(drop=True)
    X = np.arange(0, len(df_J), n)
    Y = np.arange(biases.size)

    xi, yi = np.meshgrid(X[1:], Y)
    distJ = np.zeros((X.size-1, biases.size), dtype=int)
    distF = np.zeros((X.size-1, biases.size), dtype=int)
    for i, x, in enumerate(X[1:]):
        for y in df_J.loc[X[i]:x-1,var]:
            distJ[i, bias_dict[y]] += 1
        for y in df_f.loc[X[i]:x-1,var]:
            distF[i, bias_dict[y]] += 1

    fig, ax = plt.subplots(2,1)
    sns.set_style('darkgrid')

    ax[0].contourf(xi, yi, distJ.T)
    ax[1].contourf(xi, yi, distF.T)

    ax[0].set_yticklabels(biases)
    ax[1].set_yticklabels(biases)

    ax[0].set_ylabel(f'{var}')
    ax[1].set_ylabel(f'{var}')
    ax[0].set_xlabel(f'{var} histogram, sorted by JSD')
    ax[1].set_xlabel(f'{var} histogram, sorted by frac_scales')

#   ax[0].set_xscale('log')
#   ax[1].set_xscale('log')

    plt.show()

def instructional_diagram():
    ### Set up figure and styles
    fig = plt.figure(figsize=(10,20))
    gs = gridspec.GridSpec(3,1, height_ratios=[2.0, 1.5, 1.0])
#   gs.update(wspace=0.25) #,hspace=0.20)
    ax1 = fig.add_subplot( gs[0,0] )
    ax2 = fig.add_subplot( gs[1,0] )
    ax3 = fig.add_subplot( gs[2,0] )
    sns.set_style('darkgrid')
    font1 = 20

    ### Interval distribution major key
    pair_ints = '200;200;100;200;200;200;100'
    all_ints  = '200;400;500;700;900;1100;1200;200;300;500;700;900;1000;100;300;500;700;800;200;400;600;700;200;400;500;200;300;100'
    bins = np.linspace(0, 1200, num=13)
    xxx = np.array(bins[1:], dtype=int)
    hist, edges = np.histogram([int(x) for x in pair_ints.split(';')], bins=bins)
    hist2, edges = np.histogram([int(x) for x in all_ints.split(';')], bins=bins)
    out_dict = {'Interval':list(xxx)*2, 'Probability':list(hist*13)+list(hist2*13), 'interval set':['adjacent']*12 + ['all']*12}
    sns.barplot(x='Interval', y='Probability', data=pd.DataFrame(data=out_dict), hue='interval set', ax=ax1)
    ax1.set_ylabel('Probability', fontsize=font1)
    ax1.set_xlabel('', fontsize=font1)
    ax1.set_xticklabels([str(int(x)) if (x/100)%2 == 0 else '' for x in np.arange(0, 1222, 100)])
    ax1.legend(frameon=False)



    ### Harmonic series attractors
    data_base = np.load(os.path.join(DATA_DIR, 'attractors_base.npy'))
    data_w40  = np.load(os.path.join(DATA_DIR, 'attractors_w40.npy'))

    ax2.plot(data_base[:,0], data_base[:,1])
    ax2.plot(data_w40[:,0], data_w40[:,1], 'o', fillstyle='none')
    ax2.set_ylabel('harmonic\nsimilarity score', fontsize=font1)
#   ax2.set_xlabel('Interval size', fontsize=font1)

    ### Smaller is better
    ax3.plot(np.linspace(0,1200, num=100), np.exp(-np.linspace(0, 500, num=100)**2 / 1200 * 0.001))
    ax3.plot(np.linspace(0,1200, num=100), np.exp(-np.linspace(0, 500, num=100)**2 / 1200 * 0.01))
    ax3.plot(np.linspace(0,1200, num=100), np.exp(-np.linspace(0, 500, num=100)**2 / 1200 * 0.1))
    ax3.set_ylabel('Probability', fontsize=font1)
    ax3.set_xlabel('Interval size', fontsize=font1)

    plt.tight_layout()
    plt.savefig(FIGS_DIR + 'instructional_diagram.pdf')
    plt.savefig(FIGS_DIR + 'instructional_diagram.png')
    plt.show()

def plot_model_performance(df_real, df, mi, ma, bias, q, X='pair_ints'):
    fig, ax = plt.subplots(2)
    for i, n in enumerate([5,7]):
        sns.distplot(utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n, X]), bins=100, ax=ax[i])
        df1 = pd.read_feather(df.loc[(df.n_notes==n)&(df.min_int==mi)&(df.max_int==ma)&(df.bias=='none'), 'fName'].values[0])
        sns.distplot(utils.extract_floats_from_string(df1[X]), bins=100, ax=ax[i])
#       df2 = pd.read_feather(df.loc[(df.n_notes==n)&(df.min_int==mi)&(df.max_int==ma)&(df.bias==bias)&(df.beta==beta), 'fName'].values[0])
        df2 = df.loc[(df.n_notes==n)&(df.min_int==mi)&(df.max_int==ma)&(df.bias==bias)]
        idxMin = np.abs(df2['quantile']-q).idxmin()
        print(df2.loc[idxMin])
        df2 = pd.read_feather(df2.loc[idxMin, 'fName'])
        sns.distplot(utils.extract_floats_from_string(df2[X]), bins=100, ax=ax[i])
        
def plot_clusters_df(df, labels, X='pair_ints', n=0, nc=5, count=False):
    if count:
        fig, ax = plt.subplots(n, n)
    else:
        fig, ax = plt.subplots(n, n, sharey=True, sharex=True)
    ax = ax.reshape(ax.size)
    lab_uni = sorted(np.unique(labels))
    for i, lbl in enumerate(lab_uni):
        if count:
            sns.countplot(X[labels==lbl], ax=ax[i])
        elif isinstance(X[0], float):
            sns.distplot(X[labels==lbl], bins=50, ax=ax[i])
        elif isinstance(X[0], int):
            sns.distplot(X[labels==lbl], bins=50, ax=ax[i])
        elif isinstance(X[0], np.int64):
            sns.distplot(X[labels==lbl], bins=50, ax=ax[i], kde=False)
        else:
            sns.distplot([x for y in X[labels==lbl] for x in y], bins=100, ax=ax[i])

def plot_clusters(X, labels, n=0, n2=5, count=False, bins=50):
    if count:
        sub_y, sub_x = subplot_x_y(len(set(labels)))
        fig, ax = plt.subplots(sub_y, sub_x)
        ax = ax.reshape(ax.size)
    else:
#       fig, ax = plt.subplots(n2, n2, sharey=True, sharex=True)
#   ax = ax.reshape(ax.size)
        sub_y, sub_x = subplot_x_y(len(set(labels)))
        fig, ax = plt.subplots(sub_y, sub_x, sharex=True, sharey=True)
        try:
            ax = ax.reshape(ax.size)
        except:
            ax = [ax]
    if n:
        idx = [i for i in range(len(X)) if len(X[i])==n]
        X = np.array([X[i] for i in idx])
        labels = np.array([labels[i] for i in idx])
    lab_uni = sorted(np.unique(labels))
    for i, lbl in enumerate(lab_uni):
        if count:
            sns.countplot(X[labels==lbl], ax=ax[i])
        elif isinstance(X[0], float):
            sns.distplot(X[labels==lbl], bins=bins, ax=ax[i])
        elif isinstance(X[0], int):
            sns.distplot(X[labels==lbl], bins=bins, ax=ax[i])
        elif isinstance(X[0], np.int64):
            sns.distplot(X[labels==lbl], bins=bins, ax=ax[i], kde=False)
        else:
            sns.distplot([x for y in X[labels==lbl] for x in y], bins=bins, ax=ax[i])

def plot_windows_of_attraction(ax, X, Y, diff):
    xgrid = []
    ygrid = []
    for i in range(1,len(X)-1):
        if i == 1:
            x_start = 0 + diff
        else:
            x_start = x_end

        if i == len(X)-2:
            x_end = 1200 - diff
        else:
            if Y[i+1] > Y[i]:
                x_end = X[i+1] - diff
            else:
                x_end = X[i] + diff
        xgrid.append([x_start, x_end])
        ygrid.append([Y[i]]*2)
        if i%2:
            c = [0.65]*3
        else:
            c = 'k'
        ax.fill_between(xgrid[i-1], ygrid[i-1], color=c)
    return ax
    

def plot_distributions_over_attractors(df_real, ax=0, diff=20):
    att = utils.get_attractors(1, diff=diff)
    if ax==0:
        fig, ax = plt.subplots()
#   ax = plot_windows_of_attraction(ax, att[1], np.array(att[3]), diff)
#   return
    pair_ints = np.array([np.array([float(x) for x in y.split(';')]) for y in df_real.pair_ints])
    pi1 = utils.extract_floats_from_string(df_real.pair_ints)
    pi2 = [ y[i-1] + y[i]  for y in pair_ints for i in range(1,len(y))]
    pi3 = [ y[i-2] + y[i-1] + y[i]  for y in pair_ints for i in range(2,len(y))]
    pi4 = [ y[i-3] + y[i-2] + y[i-1] + y[i]  for y in pair_ints for i in range(3,len(y)) if len(y)>3]
    pi5 = [ y[i-4] + y[i-3] + y[i-2] + y[i-1] + y[i]  for y in pair_ints for i in range(3,len(y)) if len(y)>4]
    pi6 = [ y[i-5] + y[i-4] + y[i-3] + y[i-2] + y[i-1] + y[i]  for y in pair_ints for i in range(3,len(y)) if len(y)>5]
    pi7 = [ y[i-6] + y[i-5] + y[i-4] + y[i-3] + y[i-2] + y[i-1] + y[i]  for y in pair_ints for i in range(3,len(y)) if len(y)>6]

    width = 10.0
    bins = np.arange(0, 1250, width)
    xxx = bins[:-1]
    cols = Paired_12.hex_colors
    for i, pi in enumerate([pi1, pi2, pi3, pi4, pi5, pi6, pi7]):
        hist, bins = np.histogram(pi, bins=bins)#normed=True)
        if not i:
            plt.bar(xxx, hist, width, color=cols[i], label=f'{i+1} interval, N={len(pi)}')
            base = hist
        else:
            plt.bar(xxx, hist, width, bottom=base, color=cols[i], label=f'{i+1} interval, N={len(pi)}')
            base = base + hist
    ylim = ax.get_ylim()
    ax = plot_windows_of_attraction(ax, att[1], np.array(att[3])/100.*ylim[1]*2, diff)
    ax.legend(loc='upper center', frameon=False, ncol=2)
#   ax.plot(att[1], np.array(att[3])/100.*ylim[1], 'o', c='k')


def plot_2note_probabilities(df_real, n=7, w=0.2):
    fig, ax = plt.subplots(1,2)
#   ax = ax.reshape(ax.size)
    pair_ints = np.array([np.array([float(x) for x in y.split(';')]) for y in df_real.loc[df_real.n_notes==n,'pair_ints']])
    bins = [0] + [1200./n * (1. + i * w) for i in [-1, 1]]
    print(bins)
#   bins = [0, 151, 251, 500]
    xxx_1 = np.arange(3)
    xxx_2 = np.arange(4)
    lbls_1 = ['small', 'medium', 'large']
    lbls_2 = ['medium-medium', 'medium-extreme', 'extreme-self', 'extreme-other']
    width_1 = 0.8
    width_2 = 0.4

    hist_1  = np.zeros(3)
    hist_2  = np.zeros(4)
    hist_3  = np.zeros(4)

    total_scales = len(pair_ints)
    
    for y in pair_ints:
        for i in range(len(y)):
            if y[i-1] < bins[1]:
                hist_1[0] += 1
            elif bins[1] <= y[i-1] < bins[2]:
                hist_1[1] += 1
            else:
                hist_1[2] += 1

            if not i:
                continue

            if y[i-1] < bins[1] and y[i] < bins[1]:
                hist_2[2] += 1
            elif y[i-1] >= bins[2] and y[i] >= bins[2]:
                hist_2[2] += 1
            elif y[i] < bins[1] and y[i-1] >= bins[2]:
                hist_2[3] += 1
            elif y[i-1] < bins[1] and y[i] >= bins[2]:
                hist_2[3] += 1
            elif bins[1] <= y[i-1] < bins[2] and bins[1] <= y[i] < bins[2]:
                hist_2[0] += 1
            elif bins[1] <= y[i-1] < bins[2] and not bins[1] <= y[i] < bins[2]:
                hist_2[1] += 1
            elif bins[1] <= y[i] < bins[2] and not bins[1] <= y[i-1] < bins[2]:
                hist_2[1] += 1


    hist_1 /= float(total_scales * 7)
    hist_2 /= float(total_scales * 6)

    hist_3[0] = hist_1[1]**2
    hist_3[1] = hist_1[1] * sum(hist_1[[0,2]]) * 2
    hist_3[2] = hist_1[0]**2 + hist_1[2]**2
    hist_3[3] = hist_1[0]*hist_1[2]*2


    ax[0].bar(xxx_1, hist_1, width_1)
    ax[0].set_xticks(xxx_1)
    ax[0].set_xticklabels(lbls_1)

    ax[1].bar(xxx_2, hist_2, width_2, label='real')
    ax[1].bar(xxx_2+width_2, hist_3, width_2, label='rand')
    ax[1].set_xticks(xxx_2)
    ax[1].set_xticklabels(lbls_2)
    ax[1].legend(loc='best', frameon=False)



def plot_2note_combinations(df_real, N=2):
    fig, ax = plt.subplots(2,2)
    ax = ax.reshape(ax.size)
    pair_ints = np.array([np.array([float(x) for x in y.split(';')]) for y in df_real.pair_ints])
    pi1 = [sum(y[i-N+1:i+1])  for y in pair_ints for i in range(N-1,len(y))]
    sns.kdeplot(pi1, label='real', ax=ax[3])
    for i, n in enumerate([5, 50, 500]):
        pi_new = []
        for j in range(n):
#           [pi1.append(sum(y[i-N+1:i+1]))  for y in pair_ints for i in range(N-1,len(y))]
            [pi_new.append(pi[np.random.rand(pi.size).argsort()]) for pi in pair_ints]
        pi2 = [sum(y[i-N+1:i+1])  for y in pi_new for i in range(N-1,len(y))]
        sns.distplot(pi1, bins=100, label='real', ax=ax[i])
        sns.distplot(pi2, bins=100, label=f'shuffle_{n}', ax=ax[i])
        sns.kdeplot(pi2, label=f'shuffle_{n}', ax=ax[3])
        ax[i].legend(loc='best')
    ax[3].legend(loc='best')

def plot_all_pair_dist_by_model(df_best, df, df_real, idx, X='pair_ints', mix=[], nbin=60, partabase='none'):
    fig, ax = plt.subplots(3,2)
    ax = ax.reshape(ax.size)[[0,2,4,1,3,5]]
#   ax = ax.reshape(ax.size)[[0,2,1,3]]#,3,5]]
    if X=='pair_ints':
        ylims = [0.008, 0.012, 0.016, 0.015, 0.015, 0.02]
    else:
        ylims = [0.003, 0.003, 0.003, 0.003, 0.003, 0.003]
    for i, n in enumerate(range(4,10)):
        if partabase=='none':
            sns.distplot(utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n, X]), bins=nbin+1, ax=ax[i], label='real', color='k')
            n_data = len(df_real.loc[df_real.n_notes==n, X])
        elif partabase=='theory':
            sns.distplot(utils.extract_floats_from_string(df_real.loc[(df_real.n_notes==n)&(df_real.Theory=='Y'), X]), bins=nbin+1, ax=ax[i], label='real')
            n_data = len(df_real.loc[(df_real.n_notes==n)&(df_real.Theory=='Y'), X])
        elif partabase=='instrument':
            sns.distplot(utils.extract_floats_from_string(df_real.loc[(df_real.n_notes==n)&(df_real.Theory=='N'), X]), bins=nbin+1, ax=ax[i], label='real')
            n_data = len(df_real.loc[(df_real.n_notes==n)&(df_real.Theory=='N'), X])
        for count, j in enumerate(idx):
            mi, ma, bias, beta = df_best.loc[j, ['min_int', 'max_int', 'bias', 'beta']]
            fr_10 = df_best.loc[j, f'fr10_{n}']
            if X=='pair_ints':
                lbl = df_best.loc[j, 'bias']
            else:
                lbl = df_best.loc[j, 'bias'] + f'_{round(fr_10,2)}'
#           check = len(df.loc[(df.min_int==mi)&(df.max_int==ma)&(df.bias==bias)&(df.beta==beta)&(df.n_notes==n), 'fName'])
#           if check:
            try:
                df_m = pd.read_feather(df.loc[(df.min_int==mi)&(df.max_int==ma)&(df.bias==bias)&(df.beta==beta)&(df.n_notes==n), 'fName'].values[0])
            except:
#               print(mi, ma, bias, beta, n)
                continue
            
            if X=='pair_ints':
                sns.kdeplot(utils.extract_floats_from_string(df_m[X]), ax=ax[i], label=lbl)
            else:
                dx = 0.5 * (1200. / float(nbin))
                bins = np.linspace(0 - dx, 1200 + dx, num=nbin+2)
                xxx = bins[:-1] + 0.5 * (bins[1] - bins[0])
                if len(mix):
                    if mix[count]:
                        X2 = {'pair_ints':'mix_ints', 'scale':'mix_scale'}[X]
                        histM, bins = np.histogram(utils.extract_floats_from_string(df_m.loc[:,X2]), bins=bins, normed=True)
                else:
                    histM, bins = np.histogram(utils.extract_floats_from_string(df_m.loc[:,X]), bins=bins, normed=True)
                df_hist = pd.DataFrame(data={'bin':xxx, f'model_{j}':histM})
                sns.lineplot(x='bin', y=f'model_{j}', data=df_hist, label=lbl, ax=ax[i])
#               sns.distplot(utils.extract_floats_from_string(df_m[X]), bins=100, ax=ax[i], label=lbl, kde=False, norm_hist=True)
        ax[i].set_ylim(0, ylims[i])
        ax[i].legend(loc='best')
        ax[i].set_title(f"N={n}, sample={n_data}")

def check_packing_variables(results, bias, Z='met1'):
    fig, ax = plt.subplots()
    sns.heatmap(data=results.loc[results.bias_group==bias].pivot('min_int', 'max_int', Z))
    ax.set_title(bias)

def check_scale_qualities(df_real, ss):
    fig, ax = plt.subplots(3,2)
    ax = ax.reshape(ax.size)

    idx = [int(x) for x in ss.split(';')]

    sns.countplot(df_real.loc[idx, 'Theory'], ax=ax[0])
    sns.countplot(df_real.loc[idx, 'Continent'], ax=ax[1])
    sns.countplot(df_real.loc[idx, 'cl_04'], ax=ax[2])
    sns.countplot(df_real.loc[idx, 'cl_08'], ax=ax[3])
    sns.countplot(df_real.loc[idx, 'cl_16'], ax=ax[4])

def check_scale_qualities_stacked_bar(df, c1='ss_di_10', c2='ss_hs_10', frac=True):
    fig, ax = plt.subplots(2,2)
    ax = ax.reshape(ax.size)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    truth_table  = [[False, False], [True, False], [False, True], [True, True]]
    lbls = ['not found', 'di', 'hs', 'both']
    cats = ['Theory', 'Continent', 'cl_16']
    rot = [0, 45, 45]

    x_labels =  [['N', 'Y'], sorted(df.Continent.unique()),
                 [f'{x:15s}' for x in ['equi_5', 'broad_4', 'rough_tetra_5-7', 'rough_bi_5-6', 'rough_bi_5', 'equi_7',
                 'rough_equi_7', 'tri_6-7', 'rough_tri_6-8', 'rough_tri_7',
                 'rough_tetra_5-7', 'bi_7', 'rough_bi_8-9', 'rough_bi_7-9', 'pelog_bi_7', 'broad_6-7']]]

    ft = 12
    width = 0.80
    col = np.array(Paired_12.hex_colors)[[0,6,2,4]]

    idxsort = []

    for j, cat in enumerate(cats):
        tots = {k:float(len(df.loc[(df.n_notes<=9)&(df.n_notes>=4)&(df[cat]==k)])) for k in df[cat].unique()}
        uniq = sorted([x for x in df.loc[df[cat].notnull(),cat].unique()])

        for i, tt in enumerate(truth_table):
            idx = df.loc[(df[c1]==tt[0])&(df[c2]==tt[1])].index
            cnts = df.loc[idx, cat].value_counts()
            if frac:
                Y = np.array([float(cnts[k])/float(tots[k]) if k in cnts.keys() else 0 for k in uniq])
            else:
                Y = np.array([cnts[k] if k in cnts.keys() else 0 for k in uniq])

            if not i:
                if j == 0:
                    idxsort.append(np.argsort(Y))
                elif j == 1:
                    idxsort.append([7, 1, 2, 6, 0, 5, 4, 3])
                elif j == 2:
                    idxsort.append([11, 5, 4, 12, 10, 0, 13, 2, 7, 1, 3, 9, 6, 15, 8, 14])
                X = np.arange(1,len(uniq)+1)
                ax[j].set_xticks(X)
                ax[j].set_xticklabels(np.array(x_labels[j])[idxsort[j]], rotation=rot[j], fontsize=ft)
            else:
                X = np.arange(1,len(uniq)+1)

            Y = Y[idxsort[j]]

            if not i:
                ax[j].bar(X, Y, width, color=col[i])
                base = Y
            else:
                ax[j].bar(X, Y, width, bottom=base, color=col[i])
                base = Y + base
            if not j:
                ax[3].plot([], [], label=lbls[i], color=col[i])
        ax[j].set_title(cat)


#   ax[0].set_xticks(range(1,3))
#   ax[0].set_xticklabels(['N', 'Y'], fontsize=ft)

#   ax[1].set_xticks(range(1,9))
#   ax[1].set_xticklabels(sorted(df.Continent.unique()), rotation=45, fontsize=ft)

#   ax[2].set_xticks(range(1,17))
#   ax[2].set_xticklabels([f'{x:15s}' for x in x_labels], rotation=45, fontsize=ft)
    ax[3].legend(loc='center', fontsize=24, frameon=False)

def plot_scatter_over_joint(df, X='hs_r3_w20', Y='distI_2_0', c1='ss_di_10', c2='ss_hs_10'):
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    ax = ax.reshape(ax.size)
    truth_table  = [[True, False], [False, True], [True, True], [False, False]]
    lbls = ['dI', 'HS', 'both', 'not found']
    mrks = ['<', '>', 'd', 'o']
    al = [0.8]*3 + [0.5]
    cols = list(np.array(Paired_12.hex_colors)[[1,3,5,9]]) + ['k']
    for i, tt in enumerate(truth_table):
        sns.kdeplot(df[X], df[Y], shade=True, shade_lowest=False, ax=ax[i])
        sns.kdeplot(df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), X], df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), Y], ax=ax[i], label=lbls[i], color=cols[i])
#       sns.scatterplot(df.loc[(df.ss_dI==tt[0])&(df.ss_hs_r3==tt[1]), X], df.loc[(df.ss_dI==tt[0])&(df.ss_hs_r3==tt[1]), Y], label=lbls[i], alpha=al[i], marker=mrks[i], color=cols[i], ax=ax[i])

def plot_2d_kde(df, X='hs_r3_w20', Y='distI_2_0', c1='ss_di_10', c2='ss_hs_10'):
    fig, ax = plt.subplots(3,2, sharex=True, sharey=True)
    ax = ax.reshape(ax.size)
    truth_table  = [[False, False], [True, False], [False, True], [True, True]]
    lbls = ['not found', c1[3:5], c2[3:5], 'both', 'either']
    mrks = ['o', '<', '>', 'd']
    al = [0.8]*3 + [0.5]
    cols = list(np.array(Paired_12.hex_colors)[[11,3,5,9,7]]) + ['k']
    if 0:
        sns.kdeplot(1./(5.+df[X]), df[Y], shade=True, shade_lowest=False, ax=ax[0], label='all_real')
        sns.scatterplot(1./(5.+df[X]), df[Y], ax=ax[0], label='all_real')
        for i, tt in enumerate(truth_table):
            sns.kdeplot(1./(5.+df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), X]), df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), Y], ax=ax[i+1], label=lbls[i], color=cols[i], shade=True, shade_lowest=False)
            sns.scatterplot(1./(5.+df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), X]), df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), Y], label=lbls[i], alpha=al[i], marker=mrks[i], color=cols[i], ax=ax[i+1])
        sns.kdeplot(1./(5.+df.loc[(df[c1]==True)|(df[c2]==True), X]), df.loc[(df[c1]==True)|(df[c2]==True), Y], ax=ax[5], label=lbls[4], color=cols[4], shade=True, shade_lowest=False)
        sns.scatterplot(1./(5.+df.loc[(df[c1]==True)|(df[c2]==True), X]), df.loc[(df[c1]==True)|(df[c2]==True), Y], ax=ax[5], label=lbls[4], color=cols[4])
    else:
        sns.kdeplot(df[X], df[Y], shade=True, shade_lowest=False, ax=ax[0], label='all_real')
#       sns.scatterplot(df[X], df[Y], ax=ax[0], label='all_real')
        for i, tt in enumerate(truth_table):
            sns.kdeplot(df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), X], df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), Y], ax=ax[i+1], label=lbls[i], color=cols[i], shade=True, shade_lowest=False)
            sns.scatterplot(df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), X], df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), Y], label=lbls[i], alpha=al[i], marker=mrks[i], color=cols[i], ax=ax[i+1])
        sns.kdeplot(df.loc[(df[c1]==True)|(df[c2]==True), X], df.loc[(df[c1]==True)|(df[c2]==True), Y], ax=ax[5], label=lbls[4], color=cols[4], shade=True, shade_lowest=False)
        sns.scatterplot(df.loc[(df[c1]==True)|(df[c2]==True), X], df.loc[(df[c1]==True)|(df[c2]==True), Y], ax=ax[5], label=lbls[4], color=cols[4])

    [a.legend(loc='best') for a in ax]

def plot_count(df, X='Continent', c1='ss_dI', c2='ss_hs_r3'):
    fig, ax = plt.subplots(3,2, sharex=True)#sharey=True)
    ax = ax.reshape(ax.size)
    truth_table  = [[False, False], [True, False], [False, True], [True, True]]
    lbls = ['not found', 'dI', 'HS', 'both', 'either']
    mrks = ['o', '<', '>', 'd']
    al = [0.8]*3 + [0.5]
    cols = list(np.array(Paired_12.hex_colors)[[11,3,5,9,7]]) + ['k']
    sns.countplot(df[X], ax=ax[0], label='all_real')
    for i, tt in enumerate(truth_table):
        sns.countplot(df.loc[(df[c1]==tt[0])&(df[c2]==tt[1]), X], ax=ax[i+1], label=lbls[i])
    sns.countplot(df.loc[(df[c1]==True)|(df[c2]==True), X], ax=ax[5], label=lbls[4])

    [a.legend(loc='best') for a in ax]

def plot_heatmap_fraction_found(df, X='hs_r3_w20', Y='distI_2_0', c1='ss_di_20', c2='ss_hs_20', dx=5, dy=0.01):
    xgrid = np.arange(0,35+dx, dx) 
    ygrid = np.arange(0,0.06+dy, dy) 
    new_df = pd.DataFrame(columns=[X, Y, 'frac_found', 'total'])

    for i, j in product(range(xgrid.size-1), range(ygrid.size-1)):
        total = len(df.loc[(df[X]>xgrid[i])&(df[X]<=xgrid[i+1])&(df[Y]>ygrid[j])&(df[Y]<=ygrid[j+1])])
        found = len(df.loc[(df[X]>xgrid[i])&(df[X]<=xgrid[i+1])&(df[Y]>ygrid[j])&(df[Y]<=ygrid[j+1])&((df[c1]==True)|(df[c2]==True))])
        if total == 0:
            new_df.loc[len(new_df)] = [np.mean(xgrid[i:i+2]), np.mean(ygrid[j:j+2]), np.nan, total]
        else:
            new_df.loc[len(new_df)] = [np.mean(xgrid[i:i+2]), np.mean(ygrid[j:j+2]), float(found)/float(total), total]
    
    fig, ax = plt.subplots(1,2)
    sns.heatmap(data=new_df.pivot(Y, X, 'frac_found'), ax=ax[0])
    sns.heatmap(data=new_df.pivot(Y, X, 'total'), ax=ax[1])
    [a.invert_yaxis() for a in ax]

def compare_HS_models(df, X='logq', Y='fr_10', bias_group='HS', n=5):
#   df = rename_bias_groups(df)
#   df = rename_biases(df)
    biases = [b for b in BIASES if BIAS_KEY[b]==bias_group]
    sub_y, sub_x = subplot_x_y(len(biases))
#   sub_y, sub_x = 3,4
#   sub_y, sub_x = 4,4
    fig, ax = plt.subplots(sub_y, sub_x, sharex=True, sharey=True)
    all_mi = df.loc[(df.bias=='hs_n3_w20'), 'min_int'].unique()
    all_ma = df.loc[(df.bias=='hs_n3_w20'), 'max_int'].unique()
    try:
        ax = ax.reshape(ax.size)
    except:
        ax = [ax]
    for i, bias in enumerate(biases):
        if not len(bias):
            continue
        if n:
            sns.scatterplot(x=X, y=Y, data=df.loc[(df.n_notes==n)&(df.min_int.apply(lambda x: x in all_mi))&(df.max_int.apply(lambda x: x in all_ma))&(df.bias_group==bias_group)], ax=ax[i], alpha=0.5)
            sns.scatterplot(x=X, y=Y, data=df.loc[(df.n_notes==n)&(df.min_int.apply(lambda x: x in all_mi))&(df.max_int.apply(lambda x: x in all_ma))&(df.bias==bias)], ax=ax[i])
            if 0:
                x_in = df.loc[(df.n_notes==n)&(df.min_int.apply(lambda x: x in all_mi))&(df.max_int.apply(lambda x: x in all_ma))&(df.bias==bias), X]
                y_in = df.loc[(df.n_notes==n)&(df.min_int.apply(lambda x: x in all_mi))&(df.max_int.apply(lambda x: x in all_ma))&(df.bias==bias), Y]
                x_fit, y_fit, popt = simple_fit(x_in, y_in)
                ax[i].plot(x_fit, y_fit)
#               ax[i].text(0.2, .20, f"m={popt[0]:7.5f}", transform=ax[i].transAxes)
        else:
            sns.scatterplot(x=X, y=Y, data=df, ax=ax[i], alpha=0.5)
            sns.scatterplot(x=X, y=Y, data=df.loc[(df.bias==bias)], ax=ax[i])
        ax[i].set_title(bias)

def plot_packing_probabilities(df):
    points = [x for x in df.columns if x[:2] == 'p_']
    df_new = pd.DataFrame(columns=['mi', 'ma', 'f_0', 'mean', 'mean_inc0'])
    for p in points:
        mi = int(p.split('_')[1])
        ma = int(p.split('_')[2])
        f0 = float(sum(df[p]==0)) / float(len(df))
        mean  = df.loc[df[p]!=0, p].mean()
        meanI = df[p].mean()
        df_new.loc[len(df_new)] = [mi, ma, f0, mean, meanI]

    if 1:
        fig, ax = plt.subplots(3,1)
        for i, lbl in enumerate(['f_0', 'mean', 'mean_inc0']):
#           df_new.mean_inc0 /= df_new.mean_inc0.min()
            sns.heatmap(data=df_new.pivot('mi', 'ma', lbl), ax=ax[i])
            ax[i].set_title(lbl)
            if i < 2:
                ax[i].set_xticklabels([])
            ax[i].invert_yaxis()
    else:
        fig, ax = plt.subplots(3,1)
        for i, lbl in enumerate(['f_0', 'mean', 'mean_inc0']):
#           for mi in [0, 30, 60, 90, 110]:
#               ax[i].plot(df_new.loc[df_new.mi==mi, 'ma'], df_new.loc[df_new.mi==mi, lbl] / df_new.loc[df_new.mi==mi, lbl].min(), label=str(mi))
            for ma in sorted(df_new.ma.unique()):
                ax[i].plot(df_new.loc[df_new.ma==ma, 'mi'], df_new.loc[df_new.ma==ma, lbl] / df_new.loc[df_new.ma==ma, lbl].min(), label=str(ma))
            ax[i].legend(loc='best', frameon=False)
        
def plot_clusters_biases(df):
    X = [10, 11, 5, 7, 9, 4, 1, 6, 12, 16, 2, 8, 3, 15, 14, 13]
    fig, ax = plt.subplots(4,4, sharex=True, sharey=True)
    ax = ax.reshape(ax.size)
    for i, x in enumerate(X):
        sns.kdeplot(df['im5_r0.0_w20'], df['distI_2_0'],shade=True, shade_lowest=False, ax=ax[i])
        sns.scatterplot(df.loc[df.cl_16==x, 'im5_r0.0_w20'], df.loc[df.cl_16==x,'distI_2_0'], ax=ax[i])
#       try:
#           sns.kdeplot(df.loc[df.cl_16==x, 'im5_r0.0_w20'], df.loc[df.cl_16==x,'distI_2_0'], ax=ax[i])
#       except:
#           sns.scatterplot(df.loc[df.cl_16==x, 'im5_r0.0_w20'], df.loc[df.cl_16==x,'distI_2_0'], ax=ax[i])



