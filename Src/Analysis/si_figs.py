from collections import Counter, OrderedDict
from itertools import product
import os
import re
import sys
import string
import time

import geopandas
import matplotlib.pyplot as plt
from itertools import permutations
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from palettable.colorbrewer.qualitative import Paired_12, Set1_9, Dark2_8
from palettable.colorbrewer.diverging import  RdYlGn_6, RdYlGn_11
from palettable.cmocean.sequential    import  Haline_16, Thermal_10
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import linregress, pearsonr
from shapely.geometry.point import Point

import biases
import graphs
import utils


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

BASE_DIR = '/home/johnmcbride/projects/Scales/Data_compare/'
SRC_DIR = "/home/johnmcbride/projects/Scales/Data_compare/Src"
DATA_DIR = '/home/johnmcbride/projects/Scales/Data_compare/Data_for_figs/'
REAL_DIR = '/home/johnmcbride/projects/Scales/Data_compare/Processed/Real'

FIG_DIR = '/home/johnmcbride/Dropbox/phd/LaTEX/Scales/Figures'


###########################
###     FIG 1   ###########
###########################

### See Distinguishability folder for code

###########################
###     FIG 2   ###########
###########################

def sensitivity_all(df1, df2, X='logq', Y1='JSD'):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(5,3, width_ratios=[1, .3, 1], height_ratios=[1, 1, .4, 1, 1])
    gs.update(wspace=0.00 ,hspace=0.20)
    ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0]),
          fig.add_subplot(gs[0,2]), fig.add_subplot(gs[1,2]),
          fig.add_subplot(gs[3,0]), fig.add_subplot(gs[4,0]),
          fig.add_subplot(gs[3,2]), fig.add_subplot(gs[4,2])]

    col = [(0,0,0)] + list(np.array(Thermal_10.mpl_colors)[[2,5,8]])
    al = 0.7
#   df.loc[df.bias=='Nhs_n1_w10', 'bias'] = 'hs_n1_w10'
#   df.loc[df.bias=='Nhs_n1_w20', 'bias'] = 'hs_n1_w20'
    
    df = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)
    df = df.loc[(df.logq > -5.5)].reset_index(drop=True)
#   if df.JSD.max() < 0.1:
#       df['JSD'] = df['JSD'] * 1000.
#   if df1.JSD.max() < 0.1:
#       df1['JSD'] = df1['JSD'] * 1000.
#   if df2.JSD.max() < 0.1:
#       df2['JSD'] = df2['JSD'] * 1000.

    ddf = df1.loc[(df1.n_notes==7)&(df1.bias_group=='TRANS')&(df1.min_int==80)&(df1.max_int==1200)].reset_index(drop=True)
    biases = [f"TRANS_{i}" for i in range(1,4)]
    all_b = biases
    lbl = [r'n={0}'.format(n) for n in range(1,4)]
    for i, bias in enumerate(biases):
        for j, Y in enumerate(['JSD', 'fr_10']):
            if j:
                sns.scatterplot(x=X, y=Y, data=ddf.loc[(ddf.bias==bias)], ax=ax[j+4], c=col[i])
#               x_fit, y_fit, popt = graphs.simple_fit(ddf.loc[(ddf.bias==bias), X], ddf.loc[(ddf.bias==bias), Y])
            else:
                sns.scatterplot(x=X, y=Y, data=ddf.loc[(ddf.bias==bias)], ax=ax[j+4], label=lbl[i], c=col[i])
                fn = lambda x, a, b, c: a*x**2 + b*x + c
#               x_fit, y_fit, popt = graphs.simple_fit(ddf.loc[(ddf.bias==bias), X], ddf.loc[(ddf.bias==bias), Y], fit_fn=fn)
#           ax[j+4].plot(x_fit, y_fit, c=col[i], alpha=al)

    ddf = df2.loc[(df2.n_notes==7)&(df2.bias_group=='HAR')&(df2.min_int==80)&(df2.max_int==1200)&(df2.fName.apply(lambda x: len(x.split('_'))==9))].reset_index(drop=True)
    biases = [f"HAR_{w}_1" for w in [5,10,15,20]]
    all_b += biases
    lbl = [r'w={0}'.format(w*2) for w in range(5,25,5)]
    for i, bias in enumerate(biases):
        for j, Y in enumerate(['JSD', 'fr_10']):
            if j:
                sns.scatterplot(x=X, y=Y, data=ddf.loc[(ddf.bias==bias)], ax=ax[j], c=col[i])
            else:
                sns.scatterplot(x=X, y=Y, data=ddf.loc[(ddf.bias==bias)], ax=ax[j], label=lbl[i], c=col[i])
#           x_fit, y_fit, popt = graphs.simple_fit(ddf.loc[(ddf.bias==bias), X], ddf.loc[(ddf.bias==bias), Y])
#           ax[j].plot(x_fit, y_fit, c=col[i], alpha=al)

    ddf = df2.loc[(df2.n_notes==7)&(df2.bias_group=='FIF')&(df2.min_int==80)&(df2.max_int==1200)].reset_index(drop=True)
    biases = [f"FIF_{w}" for w in [5,10,15,20]]
    all_b += biases
    for i, bias in enumerate(biases):
        for j, Y in enumerate(['JSD', 'fr_10']):
            if j:
                sns.scatterplot(x=X, y=Y, data=ddf.loc[(ddf.bias==bias)], ax=ax[j+2], c=col[i])
            else:
                sns.scatterplot(x=X, y=Y, data=ddf.loc[(ddf.bias==bias)], ax=ax[j+2], label=lbl[i], c=col[i])
#           x_fit, y_fit, popt = graphs.simple_fit(ddf.loc[(ddf.bias==bias), X], ddf.loc[(ddf.bias==bias), Y])
#           ax[j+2].plot(x_fit, y_fit, c=col[i], alpha=al)

    df = df.loc[(df.n_notes==7)&(df.bias.apply(lambda x: x in all_b))&(df.max_int==1200)&(df.logq>-5)].reset_index(drop=True)
    sns.catplot(x='min_int', y='JSD', data=df, kind='boxen', ax=ax[-2])
    sns.catplot(x='min_int', y='fr_10', data=df, kind='boxen', ax=ax[-1])

    ax = np.array(ax)

    txt = ['HAR', 'FIF', 'TRANS']
    for a in ax[:6]:
        a.set_xlim(-5.5, 0)
    for a in ax[[0, 2, 4, 6]]:
        a.set_ylabel('JSD')
    for a in ax[[1, 3, 5, 7]]:
        a.set_xlabel(r'$\log_{10}q$')
        a.set_ylabel(r'$f_\textrm{D}$')
    for i, a in enumerate(ax[[0, 2, 4]]):
#       a.set_ylim(0.0, 0.5)
        a.set_title(txt[i], fontsize=16)
    for a in ax[[0,2]]:
        a.legend(bbox_to_anchor=(0.90, 0.80), frameon=False, ncol=2, handletextpad=0, columnspacing=0)
    ax[4].legend(bbox_to_anchor=(1.00, 0.40), frameon=False, ncol=2, handletextpad=0, columnspacing=0)
    for a in ax[[0, 2, 4, 6]]:
        a.set_xticks([])
        a.set_xlabel('')
    for a in ax[[1, 3, 5]]:
        a.set_ylim(0, 0.40)
    ax[7].set_xlabel(r'$I_{\textrm{min}}$')

#   X = [-0.11, -0.11, -0.27, -0.17]
#   Y = [1.05, 1.05, 1.02, 1.02]
    for i, a in enumerate(ax[[0,2,4,6]]):
        a.text(-.17, 1.05, string.ascii_uppercase[i], transform=a.transAxes, weight='bold', fontsize=16)

    fig.savefig(os.path.join(FIG_DIR, 'sensitivity.pdf'), bbox_inches='tight')


###########################
###     FIG 3   ###########
###########################

def plot_2note_probabilities(df_real, paths, n=7, w=0.2):
    fig, ax = plt.subplots(2,5, figsize=(10,6))
    plt.subplots_adjust(hspace=0.80) #wspace=0.3, hspace=0.2)

    xxx_1 = np.arange(3)
    xxx_2 = np.arange(4)
    width_1 = 0.8
    width_2 = 0.4
    lbls_1 = ['S', 'M', 'L']
    lbls_2 = ['M\nM', 'M\nX', 'X\nE', 'X\nO']
    lbls_3 = ['MIN', 'TRANS', 'HAR', 'FIF', 'DAT']

    df_list = [pd.read_feather(paths[l][n]) for l in lbls_3[:4]] + [df_real]
    hist_1  = np.zeros(3)
    hist_2  = np.zeros(4)
    hist_3  = np.zeros(4)
    col1 = RdYlGn_6.hex_colors
    col2 = Paired_12.mpl_colors
    col = [col2[6], col2[0], col2[2], col2[4], 'k']

    for i, df in enumerate(df_list):
        pair_ints = np.array([np.array([float(x) for x in y.split(';')]) for y in df.loc[df.n_notes==n,'pair_ints']])
        bins = [0] + [1200./n * (1. + i * w) for i in [-1, 1]]
        total_scales = len(pair_ints)
        
        for y in pair_ints:
            for j in range(len(y)):
                if y[j-1] < bins[1]:
                    hist_1[0] += 1
                elif bins[1] <= y[j-1] < bins[2]:
                    hist_1[1] += 1
                else:
                    hist_1[2] += 1

                k = j-1
                if k == -1:
                    k = n-1

                if y[k] < bins[1] and y[j] < bins[1]:
                    hist_2[2] += 1
                elif y[k] >= bins[2] and y[j] >= bins[2]:
                    hist_2[2] += 1
                elif y[j] < bins[1] and y[k] >= bins[2]:
                    hist_2[3] += 1
                elif y[k] < bins[1] and y[j] >= bins[2]:
                    hist_2[3] += 1
                elif bins[1] <= y[k] < bins[2] and bins[1] <= y[j] < bins[2]:
                    hist_2[0] += 1
                elif bins[1] <= y[k] < bins[2] and not bins[1] <= y[j] < bins[2]:
                    hist_2[1] += 1
                elif bins[1] <= y[j] < bins[2] and not bins[1] <= y[k] < bins[2]:
                    hist_2[1] += 1


        hist_1 /= float(total_scales * n)
        hist_2 /= float(total_scales * n)

        hist_3[0] = hist_1[1]**2
        hist_3[1] = hist_1[1] * sum(hist_1[[0,2]]) * 2
        hist_3[2] = hist_1[0]**2 + hist_1[2]**2
        hist_3[3] = hist_1[0]*hist_1[2]*2


        ax[0,i].bar(xxx_1, hist_1, width_1, color=col[i], edgecolor='grey')
        ax[0,i].set_xticks(xxx_1)
        ax[0,i].set_xticklabels(lbls_1)

        ax[1,i].bar(xxx_2, hist_2, width_2, color=col[i], label=lbls_3[i], edgecolor='grey')
        ax[1,i].bar(xxx_2+width_2, hist_3, width_2, label='rand', color='w', edgecolor='grey', hatch='///')
        ax[1,i].set_xticks(xxx_2)
        ax[1,i].set_xticklabels(lbls_2, rotation=00)
        ax[1,i].legend(bbox_to_anchor=(1.1, 1.50), frameon=False)


        ### 2gram distribution
#       dist = utils.get_2grams_dist(df.loc[df.n_notes==7], dI=60)
#       sns.heatmap(np.log(dist+0.1), label=str(n), ax=ax[0,i])
#       ax[0,i].invert_yaxis()
#       ax[0,i].set_title(lbls_3[i])

    for a in ax[0,:]:
        a.set_ylim(0, 0.58)
    for a in ax[1,:]:
        a.set_ylim(0, 0.52)

    for a in ax[:,1:].ravel():
        a.set_yticks([])

    ax[0,0].set_ylabel('Probability')
    ax[1,0].set_ylabel('Probability')

    for i, a in enumerate(ax[:,0]):
        a.text(-.50, 1.05, string.ascii_uppercase[i], transform=a.transAxes, weight='bold', fontsize=16)

    plt.savefig(os.path.join(FIG_DIR, 'mixing_categories.pdf'), bbox_inches='tight')



###########################
###     FIG 4   ###########
###########################

def plot_mixing_effects(df1, df2, n=7):
    df = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)
    biases = [[f"HAR_{w:2d}" for w in [5,10,15,20]] + [f"HAR_{w:2d}_1" for w in [5,10,15,20]]]
    biases.append([f"FIF_{w:2d}" for w in [5,10,15,20]])
    biases.append([f"TRANS_{i}" for i in range(1,4)])
    fig, ax = plt.subplots(figsize=(10,10))
    lbl = ['HAR', 'FIF', 'TRANS']
    col = np.array(Paired_12.mpl_colors)[[3,5,1]]
    ax.plot([0,.40], [0,.40], '-', c='k')
    for i, bias in enumerate(biases):
        print(n, bias)
        ddf = utils.return_beta_below_optimum(df, bias, n)
        print(len(ddf))
        print(ddf.mfr_10.sum() / ddf.fr_10.sum())
        sns.scatterplot(x='fr_10', y='mfr_10', data=ddf, label=lbl[i], alpha=0.5, c=col[i])
        if len(ddf):
            reg = linregress(ddf['fr_10'], ddf['mfr_10'])
            X2 = np.arange(0,0.5, 0.1)
            ax.plot(X2, reg.intercept + reg.slope * X2, '-', c=col[i])
            ax.annotate(r"$y={0:4.2f} + {1:4.2f}x$".format(reg.intercept, reg.slope), (0.30, 0.05+i*0.03), color=col[i])
    ax.set_xlabel(r'Original $f_\textrm{D}$')
    ax.set_ylabel(r'Well-mixed scales $f_\textrm{D}$')
    ax.legend(loc='best', frameon=False)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 0.4)
    
    plt.savefig(os.path.join(FIG_DIR, 'mixing.pdf'), bbox_inches='tight')


###########################
###     FIG 5   ###########
###########################

def harmonic_model_correlation(df, corr_mat='None', confidence='None'):
    n_arr = np.array([1, 2, 3, 5, 10])
    w_arr = np.array([5, 10, 15, 20])
    if isinstance(corr_mat, str):
        corr_mat = np.zeros((w_arr.size, n_arr.size), dtype=float)
        confidence = np.zeros((w_arr.size, n_arr.size), dtype=float)
        for i, w in enumerate(w_arr):
            FIF = [float(len([z for z in y.split(';') if abs(702-int(z)) <= w]) / len(y.split(';'))) for y in df.all_ints2]
            for j, n in enumerate(n_arr):
                att = utils.get_attractors(n, diff=w)
                HAR = df.all_ints2.apply(lambda x: np.mean([utils.get_similarity_of_nearest_attractor(int(y), att[1], att[3]) for y in x.split(';')]))
                corr = pearsonr(FIF, HAR)
                corr_mat[i,j] = corr[0]
                confidence[i,j] = corr[1]

    xi, yi = np.meshgrid(n_arr, w_arr)
    df_heat = pd.DataFrame(data={r'$m$':xi.ravel(), r'$w$':yi.ravel(), "R":corr_mat.ravel()})
    df_heat[r'$w$'] = df_heat[r'$w$'] * 2
            
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,2, width_ratios=[1, 1])
    gs.update(wspace=0.40 ,hspace=0.00)
    ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    sns.set(font_scale=1.5)
    sns.set_style('white')
    sns.set_style("ticks", {'ytick.major.size':4})

    models = np.load("/home/johnmcbride/projects/Scales/Vocal/gill_vs_model_correlations.npy")
    fif = 0.419
    corr = list(np.array(models[1], dtype=float)) + [fif]

    lbls = ["Harrison 2018", "Milne 2013", "Pancutt 1988", "Parncutt 1994", "Stolzenburg 2015", "FIF"]
    ax[0].bar(range(len(corr)), np.abs(corr), color='white', edgecolor='k', ecolor='k')
    for i in range(len(corr)):
        ax[0].annotate(f"{abs(corr[i]):4.2f}", (i-0.29, abs(corr[i])+0.05), fontsize=14)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_xticks(range(len(corr)))
    ax[0].set_xticklabels(lbls, rotation=90)
    ax[0].set_ylim(0,1.3)
    ax[0].set_ylabel(r"Pearson's $r$")

    for i, a in enumerate(ax):
        a.text(-.15, 1.05, string.ascii_uppercase[i], transform=a.transAxes, weight='bold', fontsize=16)

    sns.heatmap(df_heat.pivot(r'$m$', r'$w$', 'R'), ax=ax[1], annot=corr_mat.T, cbar_kws={'label':r"Pearson's $r$"})
    ax[1].invert_yaxis()
    plt.savefig(os.path.join(FIG_DIR, 'har_metric_correlations.pdf'), bbox_inches='tight')

    return corr_mat, confidence


###########################
###     FIG 6   ###########
###########################

def found_scales(paths, df):
    fig = plt.figure(figsize=(10,22))
    gs = gridspec.GridSpec(3,3, width_ratios=[1, 1, 1], height_ratios=[1.2, 0.2, 0.6])
    gs.update(wspace=0.20 ,hspace=0.60)
    ax = np.array([[fig.add_subplot(gs[i,j]) for j in range(3)] for i in range(3)])

    theoryA = "TRANS"
    theoryB = "FIF"
    theoryC = "HAR"

    idx1 = list(set([int(i) for n in range(4,10) for ss in pd.read_feather(paths[theoryA][n])["ss_w10"].values for i in ss.split(';') if len(i)]))
    idx2 = list(set([int(i) for n in range(4,10) for ss in pd.read_feather(paths[theoryB][n])["ss_w10"].values for i in ss.split(';') if len(i)]))
    idx3 = list(set([int(i) for n in range(4,10) for ss in pd.read_feather(paths[theoryC][n])["ss_w10"].values for i in ss.split(';') if len(i)]))
    truth_table = [lambda x, idx1, idx2: x not in idx1 and x not in idx2,
    lambda x, idx1, idx2: x in idx1 and x not in idx2,
    lambda x, idx1, idx2: x not in idx1 and x in idx2,
    lambda x, idx1, idx2: x in idx1 and x in idx2,
    lambda x, idx1, idx2: x in idx1 or x in idx2]

    lbls = [['neither', a, b, 'both'] for a, b in zip([theoryA, theoryA, theoryC], [theoryB, theoryC, theoryB])]
    cat = 'cl_16'

    c_lbls = 'abcdefghijklmnop'

    ft = 12
    width = 0.80

    ###############################
    ### Stacked bar chart
    ###############################

    new_red = [min(1, x) for x in np.array(Paired_12.mpl_colors[5])*1.0]
    new_blu = [min(1, x) for x in np.array(Paired_12.mpl_colors[1])*1.0]
    new_gre = [min(1, x) for x in np.array(Paired_12.mpl_colors[3])*1.0]

    col = [[0.8]*3, new_blu, new_red, [.5]*3]
    al = [1] + [0.7]*2 + [1]

    tots = {k:float(len(df.loc[(df.n_notes<=9)&(df.n_notes>=4)&(df[cat]==k)])) for k in df[cat].unique()}
    uniq = sorted([x for x in df.loc[df[cat].notnull(),cat].unique()])

    idx = [i for i in df.index if truth_table[-1](i, idx1, idx2)]
    parts = {k:float(len(df.loc[(df.n_notes<=9)&(df.n_notes>=4)&(df[cat]==k)&([True if x in idx else False for x in df.index])])) for k in df[cat].unique()}
    fracs = [parts[k] / tots[k] for k in uniq]

#   idxsort = np.argsort(fracs)[::-1]
#   print(idxsort)
    idxsort = [9, 4, 10, 6, 8, 3, 11, 5, 0, 15, 1, 14, 7, 13, 2, 12] 

    idx_list = zip([idx1, idx1, idx3], [idx2, idx3, idx2])
    cols = [[[0.8]*3, a, b, [.5]*3] for a, b in zip([new_blu, new_blu, new_gre], [new_red, new_gre, new_red])]

    for j, idx_set in enumerate(idx_list):
        base = np.zeros(len(c_lbls))
        for i, tt in enumerate(truth_table[:4]):
            idx = [i for i in df.index if tt(i, idx_set[0], idx_set[1])]
            cnts = df.loc[idx, cat].value_counts()
            Y = np.array([cnts[k] if k in cnts.keys() else 0 for k in uniq])

#           print(f"{lbls[j][i]} total: {Y.sum()}")

            X = np.arange(1,len(uniq)+1)[::-1]

            Y = Y[idxsort]

            ax[0,j].barh(X, Y, width, left=base, color=cols[j][i], label=lbls[j][i], alpha=al[i])
            base = Y + base

        ax[0,j].set_ylim(0.5, 16.5)
        ax[0,j].set_yticks(range(1,17)[::-1])
        ax[0,j].set_xticks(np.arange(0,125,25))
        ax[0,j].set_xticklabels(np.arange(0,125,25), fontsize=ft+4)
        ax[0,j].set_xlabel(f"scales found", fontsize=ft+4)
        ax[0,j].set_yticklabels(list(c_lbls), fontsize=ft+4, ha="center")
        ax[0,j].tick_params(axis='y', which='major', pad=8, width=0.5)

        ax[0,j].spines['top'].set_visible(False)
        ax[0,j].spines['right'].set_visible(False)

    handles = [mpatches.Patch(color=c, label=l) for c,l in zip([[0.8]*3, new_blu, new_red, new_gre, [0.5]*3], ['neither', 'TRANS', 'FIF', 'HAR', 'both'])]
    ax[0,1].legend(loc='center right', bbox_to_anchor=( 2.00,  1.10), handles=handles, frameon=False, ncol=5, fontsize=ft+2, columnspacing=2.2)


    cols = [['k']*3, [[0.8]*3]*3, [new_blu, new_red, new_gre]]
    lbls = [theoryA, theoryB, theoryC]
    width = 0.5
    al = [1, 1, 0.7]
    X = range(2)[::-1]
    for i, idx in enumerate([idx1, idx2, idx3]):
        base = np.array([len(df.loc[((df.min_int<70)|(np.abs(df.octave-1200)>10))&(df.Theory==s)]) for s in ['Y', 'N']])
        ax[1,i].barh(X, base, width, color=cols[0][i], alpha=0.9)
        for j, tt in enumerate([False, True]):
            count = np.array([df.loc[(df.min_int>=70)&(np.abs(df.octave-1200)<=10)&([(i in idx)==tt for i in df.index]), 'Theory'].value_counts()[s] for s in ['Y', 'N']])
            ax[1,i].barh(X, count, width, left=base, color=cols[j+1][i], alpha=al[j+1])
            base += count

        ax[1,i].set_yticks([])
        ax[1,i].set_xticks(np.arange(0,310,100))
        ax[1,i].set_xticklabels(np.arange(0,310,100), fontsize=ft+4)
        ax[1,i].set_xlabel("Number of scales", fontsize=ft+4)

    ax[1,0].set_yticks(X)
    ax[1,0].set_yticklabels(['Theory', 'Measured'], fontsize=ft+4)
    handles = [mpatches.Patch(color=c, label=l) for c,l in zip(['k', [0.8]*3, new_blu, new_red, new_gre], ['prohibited', 'not found', 'TRANS', 'FIF', 'HAR'])]
    ax[1,2].legend(loc='center right', bbox_to_anchor=(1.00,  1.40), handles=handles, frameon=False, ncol=5, fontsize=ft+2, columnspacing=2.2)


    Cont = ['Western', 'Middle East', 'South Asia', 'East Asia', 'South East Asia', 'Africa', 'Oceania', 'South America']
    X = np.arange(len(Cont))[::-1]
    width = 0.5
    for i, idx in enumerate([idx1, idx2, idx3]):
        base = np.array([len(df.loc[((df.min_int<70)|(np.abs(df.octave-1200)>10))&(df.Continent==s)]) for s in Cont])
        ax[2,i].barh(X, base, width, color=cols[0][i], alpha=0.9)
        for j, tt in enumerate([False, True]):
            count = np.array([len(df.loc[(df.min_int>=70)&(np.abs(df.octave-1200)<=10)&([(i in idx)==tt for i in df.index])&(df.Continent==s)]) for s in Cont])
            ax[2,i].barh(X, count, width, left=base, color=cols[j+1][i], alpha=al[j+1])
            base += count

        ax[2,i].set_yticks([])
        ax[2,i].set_xticks(np.arange(0,150,50))
        ax[2,i].set_xticklabels(np.arange(0,150,50), fontsize=ft+4)
        ax[2,i].set_xlabel("Number of scales", fontsize=ft+4)

    ax[2,0].set_yticks(X)
    ax[2,0].set_yticklabels(Cont, fontsize=ft+4)

    handles = [mpatches.Patch(color=c, label=l) for c,l in zip(['k', [0.8]*3, new_blu, new_red, new_gre], ['prohibited', 'not found', 'TRANS', 'FIF', 'HAR'])]
    ax[2,2].legend(loc='center right', bbox_to_anchor=(1.00,  1.15), handles=handles, frameon=False, ncol=5, fontsize=ft+2, columnspacing=2.2)

    ax[0,0].text(-0.50, 1.15, "A", transform=ax[0,0].transAxes, fontsize=ft+10)
    ax[1,0].text(-0.50, 1.15, "B", transform=ax[1,0].transAxes, fontsize=ft+10)
    ax[2,0].text(-0.50, 1.15, "C", transform=ax[2,0].transAxes, fontsize=ft+10)

    fig.savefig(os.path.join(FIG_DIR, 'si_found_scales.pdf'), bbox_inches='tight')



###########################
###     FIG 7   ###########
###########################

def trans_fif_correlation(df_min, df_real, X='distI_2_0'):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    plt.subplots_adjust(wspace=0.3)#hspace=0.2)
    lbls = ['MIN', 'DAT']
    cols = [RdYlGn_6.hex_colors[1], 'k']
    C = [1.0, 0]
    df_min2 = pd.concat(df_min)
    X2 = np.arange(-0.02, 0.10, 0.01)
    al = 0.2
    idx = np.random.randint(len(df_min2), size=10000)
    for i, Y in enumerate(['Nim5_r0.0_w10', 'hs_n1_w10']):
        
        print(df_min2.loc[idx,X].values)
        ax[i].scatter(df_min2.loc[idx, X].values, 1./(C[i] + np.array(df_min2.loc[idx, Y].values)), alpha=al, color='w', edgecolor=cols[0])
        ax[i].scatter(df_real[X], 1./(C[i] + df_real[Y]), alpha=al, color='w', edgecolor=cols[1])

        reg_min = linregress(df_min2[X], 1./(C[i] + df_min2[Y]))
        reg_dat = linregress(df_real[X], 1./(C[i] + df_real[Y]))

        ax[i].plot(X2, reg_min.intercept + reg_min.slope * X2, '-', c=cols[0])
        ax[i].plot(X2, reg_dat.intercept + reg_dat.slope * X2, '-', c=cols[1])

        ax[i].annotate(r"$r={0:5.2f}$".format(reg_min.rvalue), (0.65, 0.10), xycoords="axes fraction", color=cols[0])
        ax[i].annotate(r"$r={0:5.2f}$".format(reg_dat.rvalue), (0.65, 0.20), xycoords="axes fraction", color=cols[1])

        print(f"Correlation between {X} and {Y} in {lbls[0]}:\t{pearsonr(df_min2[X], df_min2[Y])[0]}")
        print(f"Correlation between {X} and {Y} in {lbls[1]}:\t{pearsonr(df_real[X], df_real[Y])[0]}")
        print(reg_min)
        print(reg_dat)

        ax[i].set_xticks(np.arange(0,0.10, 0.02))
        ax[i].set_xlabel(r"$C_{\textrm{TRANS}}(n=2)$")
    ax[0].set_yticks([.8, .9, 1])
    ax[1].set_yticks(np.arange(.02, .1, .02))
    ax[1].set_ylim(0.010, 0.085)
    ax[0].set_ylabel(r"$C_{\textrm{FIF}}(w=20)$")
    ax[1].set_ylabel(r"$C_{\textrm{HAR}}(w=20)$")
        
    fig.savefig(os.path.join(FIG_DIR, 'trans_fif_corr.pdf'), bbox_inches='tight')



###########################
###     FIG 8   ###########
###########################

def diabolo_in_music_SI(paths, df_real, diff=20, ver=2, dia=600):
    fig, ax = plt.subplots()
    nnn = range(4,10)
    idx = [64, 56, 152, 62, 23]
    lbls = ['RAN', 'MIN', 'FIF', 'HAR', 'TRANS']
    count = []
    pat = ['--'] + ['-']*4
    col1 = RdYlGn_6.hex_colors
    col2 = Paired_12.mpl_colors
    cols = ['k', col1[1], col1[0], col2[3], col2[1]]

    if ver == 2:
        df_real = utils.get_all_ints(df_real)

    for i, n in enumerate(nnn):
        if ver == 2:
            all_ints = utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n,'all_ints2'])
        else:
            all_ints = utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n,'all_ints'])
        count.append(len([1 for x in all_ints if dia-diff<x<dia+diff]) / len(all_ints))
    ax.plot(nnn, np.array(count)*100, '-', label="DAT", c='k')

    for i in range(len(idx)):
        count = []
        for j, n in enumerate(nnn):
            df = pd.read_feather(paths[lbls[i]][n])
            if ver == 2:
#               df = utils.get_all_ints(df)
                all_ints = utils.extract_floats_from_string(df.all_ints2)
            else:
                all_ints = utils.extract_floats_from_string(df.all_ints)
            count.append(len([1 for x in all_ints if dia-diff<x<dia+diff]) / len(all_ints))
        ax.plot(nnn, np.array(count)*100, pat[i], label=lbls[i], c=cols[i])

    ax.legend(loc='best', frameon=False, ncol=2)
    ax.set_xlabel(r'$N$')
    ax.set_ylabel("Percentage of tritone intervals")
#   ax.set_yticks([])

    plt.savefig(os.path.join(FIG_DIR, 'tritone.pdf'), bbox_inches='tight')


############################
###     FIG 9    ###########
############################

def count_harmonic_intervals(df_real, att, n_att=20, real=True, inv=False):
    fig, ax = plt.subplots(2,1, figsize=(10,7))
    plt.subplots_adjust(hspace=0.20) #wspace=0.3, hspace=0.2)
    if inv:
        att_idx = np.argsort(att[3])[:n_att]
    else:
        att_idx = np.argsort(att[3])[::-1][2:n_att+2]
    int_cents = att[1][att_idx]
    count = {}
    for n in range(5,8):
        if real:
            count[n] = utils.count_ints(df_real.loc[df_real.n_notes==n], int_cents)
        else:
            count[n] = utils.count_ints(df_real[n], int_cents)
        ax[0].plot(count[n], label=f"N={n}")
        sns.regplot(np.array(att[3])[att_idx], count[n], ax=ax[1], label=f"N={n}", scatter_kws={'alpha':0.5})
#       ax[1].plot(np.array(att[3])[att_idx], count[n])
    ratios = ["{0}/{1}".format(*[int(x) for x in att[2][i]]) for i in att_idx]
    ax[0].set_xticks(range(n_att))
    ax[0].set_xticklabels(ratios)
    ax[1].set_xticks(np.arange(10, 80, 10))
    ax[0].set_xlabel('Interval frequency ratio')
    ax[1].set_xlabel('Harmonicity Score')
    for a in ax:
        a.set_ylabel('Frequency')
        a.legend(loc='best', frameon=False)

    for i, a in enumerate(ax):
        a.text(-.10, 1.05, string.ascii_uppercase[i], transform=a.transAxes, weight='bold', fontsize=16)

    plt.savefig(os.path.join(FIG_DIR, 'harmonic_intervals.pdf'), bbox_inches='tight')


############################
###     FIG 10   ###########
############################

def scale_variability(df):
    fig, ax = plt.subplots(3,3, figsize=( 8,12))
    plt.subplots_adjust(hspace=0.50) #wspace=0.3, hspace=0.2)

    lbls = [r"$C_{{\textrm{{{0}}}}}$".format(c) for c in ["TRANS", "FIF", "HAR"]]
    C = [0, 1., 0]
    for i, bias in enumerate(['distI_2_0', 'Nim5_r0.0_w10', 'hs_n1_w10']):
        if i >=1:
            sns.distplot(1./(C[i]+df.loc[df.n_notes==7, bias]), ax=ax[0,i], label='DAT')
            sns.distplot(1./(C[i]+df.loc[(df.n_notes==7)&(df.Culture=='Gamelan'), bias]), ax=ax[0,i], label='Pelog')
        else:
            sns.distplot(df.loc[df.n_notes==7, bias], ax=ax[0,i], label='DAT')
            sns.distplot(df.loc[(df.n_notes==7)&(df.Culture=='Gamelan'), bias], ax=ax[0,i], label='Pelog')
        ax[0,i].set_xlabel(lbls[i])
    ax[0,0].legend(loc='upper right', bbox_to_anchor=(2.8, 1.35), frameon=False, ncol=2)


    cat = ['pair_ints', 'scale']
    indices = [df.loc[df.Culture=='Thai'].index,
               df.loc[(df.n_notes==5)&(df.Culture=='Gamelan')].index,
               df.loc[(df.n_notes==7)&(df.Culture=='Gamelan')].index]
    labels = ['Thai', 'Slendro', 'Pelog']
    bins = [np.arange(0, 350, 20), np.arange(0, 1270, 20), np.arange(0, 1270, 20)]

    e5 = 1200./5.
    e7 = 1200./7.
    e9 = 1200./9.
    X_arr = [[e7], np.arange(e7, 1100, e7),
             [e5], np.arange(e5, 1100, e5),
             [e9, e9*2], np.arange(e9, 1100, e9)]
    Y_arr = [.03, .03, .012, .004, .005, .003]
    
    for i, idx in enumerate(indices):
        for j in range(2):
            sns.distplot(utils.extract_floats_from_string(df.loc[idx, cat[j]]), ax=ax[j+1,i], bins=bins[j], kde=False, norm_hist=True)
            X = X_arr[i*2+j]
            for x in X:
                ax[j+1,i].plot([x]*2, [0, Y_arr[j*3+i]], '-', color='k')

        ax[1,i].set_title(labels[i])
        ax[1,i].set_xlabel(r'$I_A$ / cents')
        ax[2,i].set_xlabel("Notes in scale / cents")
        ax[1,i].set_xticks(range(0,300,100))
        ax[2,i].set_xticks(range(0,1300,400))
    for a in ax.ravel():
        a.set_yticks([])
    for a in ax[:,0]:
        a.set_ylabel("Probability")

    ax[1,0].text( 80, 0.020, r'$\frac{1200}{7}$')
    ax[1,1].text(150, 0.020, r'$\frac{1200}{5}$')
    ax[1,2].text( 40, 0.008, r'$\frac{1200}{9}$')
    ax[1,2].text(180, 0.008, r'$\frac{2400}{9}$')

    for i, a in enumerate(ax[:,0]):
        a.text(-.20, 1.05, string.ascii_uppercase[i], transform=a.transAxes, weight='bold', fontsize=16)
    plt.savefig(os.path.join(FIG_DIR, 'scale_variability.pdf'), bbox_inches='tight')


############################
###     FIG 11   ###########
############################

def database_sensitivity(paths, resamp_conf, X='JSD', Y='fr_10', mean='euc'):
#   df = pd.read_feather(os.path.join(BASE_DIR,'Processed/database_sensitivity.feather'))
    col1 = RdYlGn_11.hex_colors
    col2 = Paired_12.hex_colors
    col = [col2[5], col1[4], col1[6], col2[3], col2[1], col2[7], 'k']
#   lbls = ['FIF', r'$\text{HAR}^{3}$', r'$\text{HAR}^{2}$', r'$\text{HAR}$', 'TRANS', 'MIN', 'RAN']
    lbls = ['FIF', 'HAR3', 'HAR2', 'HAR', 'TRANS', 'MIN', 'RAN']
    bias_group = ['im5', 'HAR3', 'HAR2', 'HAR', 'distI', 'none', 'none']
    titles = [r'Theory', r'Measured', r'$n=0.4S$', r'$n=0.6S$', r'$n=0.8S$']
    resamp_keys = ['theory', 'measured', 'frac0.4', 'frac0.6', 'frac0.8']
    lbls2 = ['FIF', 'HAR3', 'HAR2', 'HAR', 'TRANS', 'MIN', 'RAN']
#   col = ['k'] + list(np.array(Paired_12.hex_colors)[[7,9,3,5,1,11]])
#   bias_group = ['none', 'none', 'HAR', 'distI', 'im5', 'HAR2', 'HAR3']
#   min_int = [80]*6 + [0]
    
    fig, ax = plt.subplots(3,2, sharex=True, sharey=True, figsize=( 6, 6))
    plt.subplots_adjust(wspace=0.10, hspace=0.50)
    ax = ax.reshape(ax.size)
    samples = np.array(['theory', 'instrument'] + [f"sample_f{frac:3.1f}_{i:02d}" for frac in [0.4, 0.6, 0.8] for i in range(10)])
    idx = [[0], [1]] + [list(range(2+i*10,2+(i+1)*10)) for i in range(3)]
    for i, l in enumerate(lbls):
        for j in range(5):
#           Xarr = np.array([[df.loc[(df.bias_group==bias_group[i])&(df.min_int==min_int[i])&(df.n_notes==n), f"{s}_{X}"].values[0] for s in samples[idx[j]]] for n in [5,7]]).T
#           Yarr = np.array([[df.loc[(df.bias_group==bias_group[i])&(df.min_int==min_int[i])&(df.n_notes==n), f"{s}_{Y}"].values[0] for s in samples[idx[j]]] for n in [5,7]]).T
#           n_real = np.array([[df.loc[(df.bias_group==bias_group[i])&(df.min_int==min_int[i])&(df.n_notes==n), f"{s}_n_real"].values[0] for s in samples[idx[j]]] for n in [5,7]]).T

#           if mean=='geo':
#               Xarr = np.array([Xarr[i][0]**(n_real[i][0] / n_real[i].sum()) * Xarr[i][1]**(n_real[i][1] / n_real[i].sum()) for i in range(Xarr.shape[0])])
#               Yarr = np.array([Yarr[i][0]**(n_real[i][0] / n_real[i].sum()) * Yarr[i][1]**(n_real[i][1] / n_real[i].sum()) for i in range(Yarr.shape[0])])
#           elif mean=='euc':
#               Xarr = np.array([Xarr[i][0]*(n_real[i][0] / n_real[i].sum()) + Xarr[i][1]*(n_real[i][1] / n_real[i].sum()) for i in range(Xarr.shape[0])])
#               Yarr = np.array([Yarr[i][0]*(n_real[i][0] / n_real[i].sum()) + Yarr[i][1]*(n_real[i][1] / n_real[i].sum()) for i in range(Yarr.shape[0])])

            Xval = [resamp_conf[resamp_keys[j]][l]['jsd_int']['mean']['mean']]
            Xerr = [[Xval[0] - resamp_conf[resamp_keys[j]][l]['jsd_int']['mean']['lo']],
                    [resamp_conf[resamp_keys[j]][l]['jsd_int']['mean']['hi'] - Xval[0]]]
            Yval = [resamp_conf[resamp_keys[j]][l]['fD']['mean']['mean']]
            Yerr = [[Yval[0] - resamp_conf[resamp_keys[j]][l]['fD']['mean']['lo']],
                    [resamp_conf[resamp_keys[j]][l]['fD']['mean']['hi'] - Yval[0]]]

            ax[j].errorbar(Xval, Yval, xerr=Xerr, yerr=Yerr, color=col[i], fmt='o', label=l, mec='k', alpha=0.7, ecolor='k')
            ax[j].plot(paths[lbls2[i]][X][0], paths[lbls2[i]][Y][0], 'o', color='w', mec=col[i], label=l)

    fig.delaxes(ax[5])

    for i, a in enumerate(ax[:5]):
        a.set_title(titles[i])
        a.set_xlabel(r'$\textrm{JSD}$')
        a.set_ylabel(r'$f_{\textrm{D}}$')
        a.tick_params(axis='both', which='major', direction='in', length=6, width=2, pad=8)
#       a.set_xlim(3.0, 9.8)
        a.set_xticks(np.arange(.1,.5,.1))
        a.set_xticklabels([round(x,1) for x in np.arange(.1,.5,.1)])
#       a.set_xticklabels(np.arange(4, 10, 2))
#       a.set_yticks(np.arange(0, 0.5, 0.2))
        for tk in a.get_xticklabels():
            tk.set_visible(True)
#   ax[4].text(12.2, 0.5, r"$\textsc{\larger{dat}}$")
#   ax[4].text(15.5, 0.5, r"Subsample")
    ax[4].legend(bbox_to_anchor=(2.2, 0.9), frameon=False, ncol=2)


    fig.savefig(os.path.join(FIG_DIR, 'database_sensitivity.pdf'), bbox_inches='tight')


############################
###     FIG 12   ###########
############################

def essen_collection():
    fig, ax = plt.subplots()
    all_df, europe_df = [],  []
    df = pd.read_feather("/home/johnmcbride/projects/ABCnotation/Data/Essen/n_notes.feather")
    all_df.append(df)
    bins = np.arange(2.5, 11, 1.0)
    X = np.arange(3,11,1)
    width = 0.15
    mec = (.25, .25, .25)
    lw = 0.2
    lbls = ['Chinese', 'European']
    cols = np.array(Paired_12.mpl_colors)#[list(range(0,12,2))]
    for i, cont in enumerate(['asia', 'europa']):
        hist, bins = np.histogram(df.loc[df.Cont==cont,'n_notes'], bins=bins, normed=True)
#       ax.bar(X + (i*2-7)*width/2, hist, width, label=lbls[i], color=cols[i])
        if cont == 'asia':
            ax.bar(X + (i*2-4)*width/2, hist, width, label=lbls[i], color=cols[i], edgecolor=mec, linewidth=lw)
        else:
            europe_df.append(df)

#   collections = ['Uzun_hava', 'Native_American', 'Polish', 'European', 'Nova_Scotia', 'Meertens']
#   lbls = ['Turkish', 'Native American', 'Polish', 'European', 'Nova Scotia', 'Dutch']
    collections = ['Native_American', 'Uzun_hava', 'Polish', 'European', 'Meertens']
    lbls = ['Native American', 'Turkish', 'Polish', 'European', 'Dutch']
    for i, ext in enumerate(collections):
        df = pd.read_feather(f"/home/johnmcbride/projects/ABCnotation/Data/{ext}/n_notes.feather")
        all_df.append(df)
        hist, bins = np.histogram(df.n_notes, bins=bins, normed=True)
        if ext in ['Polish', 'European', 'Meertens']:
            europe_df.append(df)
        else:
            ax.bar(X + (i*2-2)*width/2, hist, width, label=lbls[i], color=cols[i*2+2], edgecolor=mec, linewidth=lw)

    df = pd.concat(europe_df, ignore_index=True)
    hist, bins = np.histogram(df.n_notes, bins=bins, normed=True)
    ax.bar(X + (2)*width/2, hist, width, label='European', color=cols[6], edgecolor=mec, linewidth=lw)

    df = pd.concat(all_df, ignore_index=True)
    hist, bins = np.histogram(df.n_notes, bins=bins, normed=True)
#   ax.bar(X + (2*len(collections)-3)*width/2, hist, width, label='All', color='k')
    ax.bar(X + (4)*width/2, hist, width, label='All', color='grey', edgecolor=mec, linewidth=lw)

    ax.legend(loc='best', frameon=False)
    ax.set_xlabel(r'$N$')
    ax.set_ylabel("Normalised probability distribution")
    ax.set_yticks([])

    plt.savefig(os.path.join(FIG_DIR, 'essen_database.pdf'), bbox_inches='tight')



############################
###     FIG 13   ###########
############################

def choosing_A(m=1):
    X, Y = np.load(os.path.join(BASE_DIR, 'Processed/Cleaned/MIN_cost_func/MIN_7_HAR_stats.npy'))

    Xmin = X[Y>0][0]
    Xmax = X[Y>0][-1]
    beta_arr = np.array([10**float(n) for n in np.arange(0, 8, 0.1)])
    fig, ax = plt.subplots(3,2, figsize=(30,30))
    plt.subplots_adjust(wspace=0.30, hspace=0.30)
    col = ['b', 'orange', 'g']
    col = Dark2_8.hex_colors
    target_sel = 0.5
    eqn = {0:r'$C_1 = 1 - (\bar H / A)$',
           1: r'$C_2 = 1 / (\bar H + A)$'}

    lbl2 = [r'$A=\bar H_{max}$', r'$A=-\bar H_{min}$']

    for j in range(2):
        if j==0:
            fn = lambda x, b, a, m: np.exp(-b*(1-(x/a)**m))
            a_inp = np.array([0.8, 1.0, 2.0])
            a_arr1 = np.arange(0.8, 1.4, 0.2)*Xmax
            a_arr1 = a_inp*Xmax
            a_arr2 = np.arange(0.7, 2, 0.1)*Xmax
            idx = np.where(np.abs(a_arr2-Xmax)<0.1)[0][0]
#           a_lbl = [r"$A={0:4.1f} \bar H_\textrm{{max}}$".format(a) for a in np.arange(0.8, 1.4, 0.2)]
            a_lbl = [r"$A={0:4.1f} \bar H_{{max}}$".format(a) for a in a_inp]
        else:
            fn = lambda x, b, a, m: np.exp(-b/(x+a)**m)
            a_inp = - np.array([1.2, 1.0, 0.5])
            a_arr1 = - np.arange(0.8, 1.4, 0.2)*Xmin
            a_arr1 = a_inp*Xmin
            a_arr2 = - np.arange(0.2, 1.2, 0.1)*Xmin
            idx = np.where(np.abs(a_arr2+Xmin)<0.1)[0][0]
#           a_lbl = [r"$A={0:4.1f}\barH_{{\textrm{{min}}}}$".format(a) for a in -np.arange(0.8, 1.4, 0.2)]
            a_lbl = [r"$A={0:4.1f} \bar H_{{min}}$".format(a) for a in a_inp]


        for i, a in enumerate(a_arr1):
            acc, sel = np.array([biases.evaluate_cost_function(X, Y, fn, [beta, a, m]) for beta in beta_arr]).T
            ax[1,j].plot(sel, acc, label=a_lbl[i], c=col[i])
            qX = biases.evaluate_cost_function(X, Y, fn, [biases.get_beta_for_acc(sel, beta_arr, target_sel), a, m], qX_only=True)
            acc, sel = biases.evaluate_cost_function(X, Y, fn, [biases.get_beta_for_acc(sel, beta_arr, target_sel), a, m])
            ax[1,j].plot(sel, acc, 'o', c=col[i], fillstyle='none', ms=10)
            ax[0,j].plot(X, qX, c=col[i], label=a_lbl[i])

        ax[0,j].plot(X, Y, c='k')

        acc_arr = []
        for a in a_arr2:
            acc, sel = np.array([biases.evaluate_cost_function(X, Y, fn, [beta, a, m]) for beta in beta_arr]).T
            acc, sel = biases.evaluate_cost_function(X, Y, fn, [biases.get_beta_for_acc(sel, beta_arr, target_sel), a, m])
            acc_arr.append(acc)
        ax[2,j].plot(a_arr2, acc_arr, c='k', label=r"$\textrm{JSD} = 0.5$")
        ax[2,j].plot(a_arr2[idx], acc_arr[idx], 'o', c='k', fillstyle='none', ms=10, label=lbl2[j])

        ax[1,j].set_xlabel(r'$\textrm{JSD}$')
        ax[1,j].set_ylabel('Acceptance Rate')
        ax[1,j].set_yscale('log')
        ax[1,j].set_ylim(10**-7, 10)
        ax[0,j].legend(loc='best', frameon=False)
        ax[1,j].legend(loc='best', frameon=False)
        ax[2,j].legend(loc='best', frameon=False)
        ax[2,j].set_yscale('log')
        ax[2,j].set_ylim(10**-10, 1)

        ax[0,j].set_xlabel(r"$\bar H$")
        ax[0,j].set_ylabel('Probability')
        ax[2,j].set_xlabel(r"$A$")
        ax[2,j].set_ylabel('Acceptance Rate')
    
        ax[0,j].text(0.3, 1.10, eqn[j], transform=ax[0,j].transAxes, fontsize=24)

    for i, a in enumerate(ax[:,0]):
        a.text(-.15, 1.05, string.ascii_uppercase[i], transform=a.transAxes, weight='bold', fontsize=20)

    plt.savefig(os.path.join(FIG_DIR, 'choosing_a.pdf'), bbox_inches='tight')


############################
###     FIG 14   ###########
############################

def choosing_m():
    X, Y = np.load(os.path.join(BASE_DIR, 'Processed/Cleaned/MIN_cost_func/MIN_7_HAR_stats.npy'))

    Xmin = X[Y>0][0]
    Xmax = X[Y>0][-1]
    beta_arr = np.array([10**float(n) for n in np.arange(0, 8, 0.2)])
    fig, ax = plt.subplots(3,2, figsize=(30,30))
    plt.subplots_adjust(wspace=0.30, hspace=0.30)
    col = ['b', 'orange', 'g']
    col = Dark2_8.hex_colors
    target_sel = 0.5
    eqn = {0:r'$C_3 = 1 - (\bar H / A)^m$',
           1: r'$C_4 = 1 / (\bar H + A)^m$'}

    for j in range(2):
        if not j:
            fn = lambda x, b, a, m: np.exp(-b*(1-(x/a)**m))
            m_arr = np.array([4, 1, 0.25])
            A = Xmax
        else:
            fn = lambda x, b, a, m: np.exp(-b/(x+a)**m)
            m_arr = np.array([0.25, 1, 4])
            A = -Xmin
        beta_arr = np.array([10**float(n) for n in np.arange(0, 15, 0.1)])

        for i, m in enumerate(m_arr):
            acc, sel = np.array([biases.evaluate_cost_function(X, Y, fn, [beta, A, m]) for beta in beta_arr]).T
            ax[1,j].plot(sel, acc, label=r"$m={0:6.2f}$".format(m), c=col[i])
            qX = biases.evaluate_cost_function(X, Y, fn, [biases.get_beta_for_acc(sel, beta_arr, target_sel), A, m], qX_only=True)
            acc, sel = biases.evaluate_cost_function(X, Y, fn, [biases.get_beta_for_acc(sel, beta_arr, target_sel), A, m])
            ax[1,j].plot(sel, acc, 'o', c=col[i], fillstyle='none', ms=10)
            ax[0,j].plot(X, qX, c=col[i], label=r"$m={0:6.2f}$".format(m))

        ax[0,j].plot(X, Y, c='k')

        acc_arr = []
        m_arr = np.arange(0.4, 4.2, 0.2)
        idx = np.where(np.abs(m_arr-1)<0.01)[0][0]
        for m in m_arr:
            acc, sel = np.array([biases.evaluate_cost_function(X, Y, fn, [beta, A, m]) for beta in beta_arr]).T
            acc, sel = biases.evaluate_cost_function(X, Y, fn, [biases.get_beta_for_acc(sel, beta_arr, target_sel), A, m])
            acc_arr.append(acc)

        ax[2,j].plot(m_arr, acc_arr, c='k')
#       ax[2,j].plot(m_arr[idx], acc_arr[idx], 'o', c='k', fillstyle='none', ms=10)

        ax[1,j].set_xlabel(r'$\textrm{JSD}$')
        ax[1,j].set_ylabel('Acceptance Rate')
        ax[1,j].set_yscale('log')
        ax[1,j].set_ylim(10**-7, 10)
        ax[0,j].legend(loc='best', frameon=False)
        ax[1,j].legend(loc='best', frameon=False)
        ax[2,j].set_yscale('log')
        ax[2,j].set_ylim(10**-6, 1)
    
        ax[0,j].set_xlabel(r"$\bar H$")
        ax[0,j].set_ylabel('Probability')
        ax[2,j].set_xlabel(r"$m$")
        ax[2,j].set_ylabel('Acceptance Rate')

        ax[0,j].text(0.3, 1.10, eqn[j], transform=ax[0,j].transAxes, fontsize=24)
    
    for i, a in enumerate(ax[:,0]):
        a.text(-.15, 1.05, string.ascii_uppercase[i], transform=a.transAxes, weight='bold', fontsize=20)


    plt.savefig(os.path.join(FIG_DIR, 'choosing_m.pdf'), bbox_inches='tight')


############################
###     FIG 15   ###########
############################

def bias_sensitivity(df2):
    df = pd.read_feather(os.path.join(BASE_DIR,'Processed/methods_check.feather'))
    biases = sorted(df.bias.unique())
    fig, ax = plt.subplots(7,2, sharex=True, sharey=True, figsize=(15,15))
    ax = ax.reshape(ax.size)
    eqn = {'A':r'$C_3 = 1 - (\bar H / A)^{{{0}}}$',
           'B': r'$C_4 = 1 / (\bar H + A)^{{{0}}}$'}
    for i, b in enumerate(biases):
        eqn_type = b.split('_')[3]
        lbl = eqn[eqn_type].format(str(float(b.split('_')[2])))
        ax[i].plot(*df2.loc[(df2.bias=='HAR_10_1')&(df2.n_notes==7)&(df2.min_int==80), ['JSD', 'fr_10']].T.values, 'o', c='k', fillstyle='full', alpha=0.5)
        ax[i].plot(*df.loc[df.bias==b, ['JSD', 'fr_10']].T.values, 'o', c='orange', fillstyle='full', alpha=0.5, label=lbl)
        ax[i].set_yticks(np.arange(0,.5,.2))
#       ax[i].set_xticks(np.arange(6, 12))
#       ax[i].set_xlim(0.1, 1.0)
        ax[i].tick_params(axis='both', which='major', direction='out', length=6, width=2, pad=8)
        handles = [Line2D([],[], lw=0, marker='o', color='orange', label=lbl, ms=10)]
#       ax[i].legend(loc='lower right', handles=handles, frameon=False)
        ax[i].legend(bbox_to_anchor=(0.60, 0.5), handles=handles, frameon=False, handletextpad=0)
    for a in ax[-2:]:
        a.set_xlabel(r"$\textrm{JSD}$")
    for a in ax[range(0, ax.size,2)]:
        a.set_ylabel(r"$f_{\textrm{D}}$")

    ax[0].text(0.3, 1.20, r'$C_3 = 1 - (\bar H / A)^m$', transform=ax[0].transAxes, fontsize=24)
    ax[1].text(0.3, 1.20, r'$C_4 = 1 / (\bar H + A)^m$', transform=ax[1].transAxes, fontsize=24)
        
    fig.savefig(os.path.join(FIG_DIR, 'bias_sensitivity.pdf'), bbox_inches='tight')




###########################
###     TAB 1   ###########
###########################

def database_reference_table(df):
    with open("/home/johnmcbride/Dropbox/phd/LaTEX/Scales/SI_table_04.txt", 'w') as o:
        o.write("\\begin{table}\\centering\n")
        o.write("\\caption{\\label{tab:params}\n  }\n")
        o.write("\\begin{tabular}{l|cccc}\n")
        o.write(" Reference & Theory & Instrument & Recording & Total   \\\\ \n")
        o.write("\\toprule\n")
        for ref in df.Reference.value_counts().keys():
            bibtex = df.loc[df.Reference==ref, 'bibtex'].values[0]
            source = [len(df.loc[(df.Reference==ref)&(df.Source==x)]) for x in ['Theory', 'Instrument', 'Recording']]
            o.write(f"\\cite{{{bibtex}}} & {source[0]} & {source[1]} & {source[2]} & {sum(source)}  \\\\ \n")
        o.write("\\end{tabular}\n\\end{table}")

###########################
###     TAB 3   ###########
###########################

def prominent_harmonic_intervals_table(n1=50, n2=50, att=0.8):
    counts = []
    for n1, n2, att in product([2,11], [11], [1.0, 0.7, 0.4]):
        counts.append(utils.harmonic_series_intervals(n1=n1, n2=n2, att=att))
    with open("/home/johnmcbride/Dropbox/phd/LaTEX/Scales/SI_table_03.txt", 'w') as o:
        o.write("\\begin{table}\\centering\n")
        o.write("\\caption{\\label{tab:params}\n  }\n")
        o.write("\\begin{tabular}{l|cc|cc|cc|cc|cc|cc}\n")
        o.write(" $n_1$ &  \\multicolumn{6}{c|}{1}  &  \\multicolumn{6}{c}{10}  \\\\ \n")
        o.write(" $a$  " + "&  \\multicolumn{2}{c|}{0}  &  \\multicolumn{2}{c|}{0.3}  &  \\multicolumn{2}{c|}{0.6}  "*2 + "\\\\ \n")
        o.write(" & ratio & weight"*6 + "  \\\\ \n")
        o.write("\\toprule\n")
        for i in range(10):
            ratios  = [counts[j][i][0] if len(counts[j]) > i else '' for j in range(6)]
        o.write("\\end{tabular}\n\\end{table}")

###########################
###     TAB 4   ###########
###########################

def write_parameter_values_table(paths):
    labels = ['RAN', 'MIN', 'HAR', 'TRANS', 'FIF']
    biases = ['', '', 'hs_n1_w10', 'distI_2_0', 'Nim5_r0.0_w10']
    bins = ['', ''] + [np.linspace(0, x, 101) for x in [50, 0.1, 0.25]]

    const = pd.read_feather(os.path.join(SRC_DIR, 'Params', 'constants.feather'))

    with open("/home/johnmcbride/Dropbox/phd/LaTEX/Scales/SI_table_05.txt", 'w') as o:
        o.write("\\begin{table}\\centering\n")
        o.write("\\caption{\\label{tab:params}\n  }\n")
        o.write("\\begin{tabular}{lccccccccc}\n")
        o.write("Model  &  $N$  & $I_{\\textrm{min}}$ &  $n$  &  $w$  & $\H_{min}$ &  $\H_{max}$ & $\\beta$ &  $q$  & $\\textrm{JSD}$ \\\\ \n")
        o.write("\\toprule\n")
        for l, b, bn in zip(labels, biases, bins):
            for n in range(4,10):
                df = pd.read_feather(paths[l][n])
                splt = os.path.split(paths[l][n])[1].split('_')
                beta = f"{float(splt[-1].replace('.feather', '')):6.1f}"

                mi = {'RAN':'0'}.get(l, '80')
                w = {'HAR':'20', 'FIF':'20'}.get(l, '')
                nn = {'TRANS':'2'}.get(l, '')
                q = f"{float(len(df) / df.n_att.sum()):7.1e}"

                hmin, hmax = ['', '']
                
                if l == 'HAR':
                    df = utils.update_hss_scores_df(utils.get_all_ints(df))
                    hmin, hmax = [str(x) for x in const.loc[(const.bias==l)&(const.N==int(n))&(const.W==int(int(w)/2))&(const.M==1),  ['Min', 'Max']].values[0]]
                elif l == 'FIF':
                    df = utils.calculate_fifths_bias(df, w=10)
                    beta = f"{round(float(beta)*200./3., 4):4.1f}"
                if b == '':
                    jsd = ''
                else:
                    cost, bn = np.histogram(df[b], bins=bn)
                    X, min_cost = np.load(BASE_DIR + f"/Processed/Cleaned/MIN_cost_func/MIN_{n}_{l}_stats.npy")
                    jsd = f"{jensenshannon(cost, min_cost):4.2f}"

                o.write(' & '.join([l, str(n), mi, nn, w, hmin, hmax, beta, q, jsd]) + " \\\\\n")
        o.write("\\end{tabular}\n\\end{table}")


###########################
###     TAB 5   ###########
###########################

def write_goodness_of_fit_table(df_real, boot_conf):
    labels = ['RAN', 'MIN', 'HAR', 'TRANS', 'FIF']
#   if 0 not in df.N.unique():
#       scores = np.array([[s for s in utils.average_goodness_of_fit(df, X=m)] for m in ['r2', 'RMSD', 'd_RMSD', 'met1']]).T
#       for l, s in zip(labels, scores):
#           df.loc[len(df)] = [l, 0, 0] + list(s)
#   if 1 not in df.N.unique():
#       scores = np.array([[s for s in utils.average_goodness_of_fit(df, X=m, ave='geo')] for m in ['r2', 'RMSD', 'd_RMSD', 'met1']]).T
#       for l, s in zip(labels, scores):
#           df.loc[len(df)] = [l, 1, 0] + list(s)
#   
    S_n = np.array([len(df_real.loc[df_real.n_notes==n]) for n in range(4,10)])
    with open("/home/johnmcbride/Dropbox/phd/LaTEX/Scales/SI_table_01.txt", 'w') as o:
        o.write("Model  &  $N$  &  $S$  &  JSD  &  CI  &  CvM  &  CI  &  $f_D$  &  CI  \\\\ \n")
        for l in labels:
            for i, n in enumerate(range(4,10)):
#               scores = df.loc[(df.N==n)&(df.bias==l), ['r2', 'RMSD', 'd_RMSD', 'met1']].values[0]
#               S = df.loc[(df.N==n)&(df.bias==l), 'S'].values[0]
                JSD = f"{boot_conf[l]['jsd_int'][n]['mean']:4.2f}"
                JSD_CI = "({0}, {1})".format(*[f"{boot_conf[l]['jsd_int'][n][x]:4.2f}" for x in ['lo', 'hi']])
                CVM = f"{boot_conf[l]['cvm_int'][n]['mean']:4.2f}"
                CVM_CI = "({0}, {1})".format(*[f"{boot_conf[l]['cvm_int'][n][x]:4.2f}" for x in ['lo', 'hi']])
                fD = f"{boot_conf[l]['fD'][n]['mean']:4.2f}"
                fD_CI = "({0}, {1})".format(*[f"{boot_conf[l]['fD'][n][x]:4.2f}" for x in ['lo', 'hi']])
                S = S_n[i]
                scores = ' & '.join([JSD, JSD_CI, CVM, CVM_CI, fD, fD_CI])
                if not i:
                    o.write(f"{l:6s} & {n:3d} & {S:3d} & {scores} \\\\\n")
                elif n==9:
                    o.write(f"       & {n:3d} & {S:3d} & {scores} \\vspace{{0.1cm}} \\\\\n")
                else:
                    o.write(f"       & {n:3d} & {S:3d} & {scores} \\\\\n")

            JSD = f"{boot_conf[l]['jsd_int']['mean']['mean']:4.2f}"
            JSD_CI = "({0}, {1})".format(*[f"{boot_conf[l]['jsd_int']['mean'][x]:4.2f}" for x in ['lo', 'hi']])
            CVM = f"{boot_conf[l]['cvm_int']['mean']['mean']:4.2f}"
            CVM_CI = "({0}, {1})".format(*[f"{boot_conf[l]['cvm_int']['mean'][x]:4.2f}" for x in ['lo', 'hi']])
            fD = f"{boot_conf[l]['fD']['mean']['mean']:4.2f}"
            fD_CI = "({0}, {1})".format(*[f"{boot_conf[l]['fD']['mean'][x]:4.2f}" for x in ['lo', 'hi']])
            scores = ' & '.join([JSD, JSD_CI, CVM, CVM_CI, fD, fD_CI])

            o.write(f"       & arithmatic mean &    & {scores} \\vspace{{0.2cm}} \\\\\n")




###########################
###     FIG #   ###########
###########################

#   def attenuation_correlation():
#       fig, ax = plt.subplots()
#       data = np.load(os.path.join(BASE_DIR, 'attenutation_model_corr.npy'))
#       att = np.arange(0.01, 1.01, 0.01)
#       col1 = RdYlGn_11.hex_colors
#       col2 = Paired_12.hex_colors
#       col = [col2[5], col1[4], col1[6], col2[3]]
#       lbls = ['FIF', r'$\text{HAR}^{3}$', r'$\text{HAR}^{2}$', r'$\text{HAR}$']
#       for i, d in enumerate(data):
#           ax.plot(att, d[::-1], '-', c=col[i], label=lbls[i])
#       ax.legend(loc='best', frameon=False, ncol=2)
#       ax.set_xlabel('a')
#       ax.set_ylabel(r"Pearson's $r$")
#       ax.set_yticks(np.arange(0.85, 1.05, 0.05))
#       ax.set_ylim(.83, 1.0)

#       fig.savefig(os.path.join(FIG_DIR, 'attenuation_correlation.pdf'), bbox_inches='tight')




