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
from palettable.colorbrewer.qualitative import Paired_12, Set1_9, Dark2_8, Accent_4
from palettable.colorbrewer.diverging import  RdYlGn_6, RdYlGn_11
from palettable.cmocean.sequential    import  Haline_16, Thermal_10
from scipy.stats import linregress, pearsonr
import seaborn as sns
from shapely.geometry.point import Point

import graphs
import utils

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath} \usepackage{xcolor} \usepackage{soul}']

BASE_DIR = '/home/johnmcbride/projects/Scales/Data_compare/'
DATA_DIR = '/home/johnmcbride/projects/Scales/Data_compare/Data_for_figs/'
REAL_DIR = '/home/johnmcbride/projects/Scales/Data_compare/Processed/Real'

FIG_DIR = '/home/johnmcbride/Dropbox/phd/LaTEX/Scales/Figures'

BIASES = ['none', 'S#1_n1', 'S#1_n2',#'',
          'distI_n1', 'distI_n2', 'distW',#'',
          'distI_n1_S#1_n1', 'distI_n1_S#1_n2', 'distI_n2_S#1_n1', 'distI_n2_S#1_n2',
          'distW_S#1_n1', 'distW_S#1_n2', 'distW_S#2_n2', 'distW_S#2_n3',
          'hs_n1_w05', 'hs_n1_w10', 'hs_n1_w15', 'hs_n1_w20',
          'hs_n2_w05', 'hs_n2_w10', 'hs_n2_w15', 'hs_n2_w20',
          'hs_n3_w05', 'hs_n3_w10', 'hs_n3_w15', 'hs_n3_w20',
          'hs_r3_w05', 'hs_r3_w10', 'hs_r3_w15', 'hs_r3_w20']
          
BIAS_GROUPS = ['none', 'S#1', 'HS',
               'distW', 'distW_S#1', 'distW_S#2',
               'distI', 'distI_S#1']

BIAS_GROUPS = ['none', 'HS',
               'S#1', 'distW',
               'distW_S#1', 'distW_S#2',
               'distI', 'distI_S#1', 'AHS']

groups = ['none'] + ['S#1']*2 + ['distI']*2 + ['distW'] + ['distI_S#1']*4 + \
         ['distW_S#1']*2 + ['distW_S#2']*2 + ['HS']*16
BIAS_KEY = {BIASES[i]:groups[i] for i in range(len(BIASES))}


###########################
###     FIG 1   ###########
###########################

def world_map(df_real):
    df_real = df_real.loc[(df_real.n_notes>3)&(df_real.n_notes<10)].reset_index(drop=True)

    counts = df_real.loc[(df_real.Theory=='N')&(df_real.Country.str.len()>0), 'Country'].value_counts()
    countries = counts.keys()
    co = counts.values

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world['cent_col'] = world.centroid.values

    coord = [world.loc[world.name==c, 'cent_col'].values[0] for c in countries]
    gdf = geopandas.GeoDataFrame(pd.DataFrame(data={'Country':countries, 'count':co, 'coord':coord}), geometry='coord')
    

    Cont = ['Western', 'Middle East', 'South Asia', 'East Asia', 'South East Asia', 'Africa', 'Oceania', 'South America']
    theory = [len(df_real.loc[(df_real.Theory=='Y')&(df_real.Continent==c)]) for c in Cont]
    inst   = [len(df_real.loc[(df_real.Theory=='N')&(df_real.Continent==c)]) for c in Cont]

    cont_coord = [Point(*x) for x in [[17, 48], [38, 35], [79, 24], [110, 32], [107, 12], [18, 8], [150, -20], [-70, -10]]]

    cont_df = geopandas.GeoDataFrame(pd.DataFrame(data={'Cont':Cont, 'count':theory, 'coord':cont_coord}), geometry='coord')

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(2,3, width_ratios=[1.0, 7.0, 1.0], height_ratios=[1,0.6])
    gs.update(wspace=0.1 ,hspace=0.10)
    ax = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,1])]
    col = Paired_12.mpl_colors
    ft1 = 12

    world.plot(ax=ax[0], color=(0.6, 0.6, 0.6), edgecolor=(1.0,1.0,1.0), lw=0.2)
    world.loc[world.name.apply(lambda x: x in countries)].plot(ax=ax[0], color=(0.3, 0.3, 0.3), edgecolor=(1.0,1.0,1.0), lw=0.2)
    gdf.plot(color='r', ax=ax[0], markersize=gdf['count'].values*0.5)
    cont_df.plot(color='g', ax=ax[0], markersize=cont_df['count'].values)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim(-185, 185)
    ax[0].set_ylim(-60, 88)

    width = 0.4
    X = np.arange(len(Cont))
    ax[1].bar(X - width/2, theory, width, label='Theory', color=np.array(col[2])*0.9)
    ax[1].bar(X + width/2, inst, width, label='Measured', color=np.array([col[4]])*0.9)
    for i in range(len(theory)):
        ax[1].annotate(str(theory[i]), (X[i]-abs(len(str(theory[i])))**0.5*width/2, theory[i]+5), fontsize=ft1)
        ax[1].annotate(str(inst[i]), (X[i]+(2-len(str(inst[i]))**0.5)*width/4, inst[i]+5), fontsize=ft1)

    ax[1].set_xticks(X)
    [tick.label.set_fontsize(ft1) for tick in ax[1].xaxis.get_major_ticks()]
    [tick.label.set_fontsize(ft1) for tick in ax[1].yaxis.get_major_ticks()]
    ax[1].set_xticklabels(Cont, rotation=28, fontsize=ft1)
    ax[1].legend(loc='upper right', frameon=False, fontsize=ft1)
    ax[1].set_ylabel('Number of scales', fontsize=ft1+2)
    ax[1].set_ylim(0, 175)

    plt.savefig(os.path.join(FIG_DIR, 'world_map.pdf'), bbox_inches='tight')
    

###########################
###     FIG 2   ###########
###########################

def hss_instructional(diff=20, att='None'):
    if att == 'None':
        att = utils.get_attractors(1, diff=diff)
    fig, ax = plt.subplots(figsize=(12,4))
    ft = 18

    ax = graphs.plot_windows_of_attraction(ax, att[1], np.array(att[3]), diff)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0,100)
    ax.set_xlim(-10,1210)
    ax.set_yticks([])
    ax.set_xticks(range(0,1400,200))
    ax.set_xticklabels(range(0,1400,200), fontsize=ft+2)
    ax.set_xlabel('Interval size / cents', fontsize=ft+4)
    ax.set_ylabel('Harmonicity score', fontsize=ft+4)
    ax.tick_params(axis='x', direction='out', length=5, width=2.0)

    fig.savefig(os.path.join(FIG_DIR, 'hss_score.pdf'), bbox_inches='tight')


###########################
###     FIG 3   ###########
###########################

def plot_all_adj_int_dist_by_model(paths, df_real, X='pair_ints', mix=[], nbin=60, partabase='none'):
    fig, ax = plt.subplots(3,2, figsize=( 8,  8))
    ax = ax.reshape(ax.size)[[0,2,4,1,3,5]]
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    ax2 = fig.add_axes([0.745, 0.295, 0.15, 0.06])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)


    ytick = [np.arange(0, (i+1)*0.005, 0.005) for i in [1, 2, 2, 2, 2, 3]]
    ylims = [0.010, 0.012, 0.016, 0.016, 0.018, 0.025]
    ft1 = 14
    labels = ['DAT', 'RAN', 'MIN', 'TRANS', 'HAR', 'FIF']

    col1 = RdYlGn_6.hex_colors
    col2 = Paired_12.mpl_colors
    cols = [col1[1], col2[1], col2[3], col1[0]]
    lw = np.array([1,1,1,1.0])*1.7

    dI = [9.36, 7.55, 7.27, 6.02, 4.01]
    ax2.bar(range(len(dI)), dI, color=['k'] + cols)
    ax2.set_xticks(range(len(dI)))
    ax2.set_xticklabels(labels[1:], rotation=90, fontsize=ft1)
    ax2.set_yticks([0,4,8])
    ax2.set_yticklabels([0,4,8], fontsize=ft1)
#   ax2.set_xlabel(r"Models")
    ax2.set_ylabel(r"$d_\textrm{I}$", labelpad=0, fontsize=ft1+2)
#   sys.exit()

    for i, n in enumerate(range(4,10)):
        df_r = pd.read_feather(paths['RAN'][n])
        bins = range(0,550,20)
        ax[i].hist(utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n, X]), bins=bins, color=[.6]*3, normed=True, alpha=.7, label='DAT')
        sns.kdeplot(utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n, X]), ax=ax[i], label='DAT', color='k')
        sns.kdeplot(utils.extract_floats_from_string(df_r[X]), ax=ax[i], color='k', linestyle=":", label='RAN')
        n_data = len(df_real.loc[df_real.n_notes==n, X])
        for count, j in enumerate(labels[2:]):
            sns.kdeplot(utils.extract_floats_from_string(pd.read_feather(paths[j][n])[X]), ax=ax[i], label=labels[count+2], color=cols[count], lw=lw[count])
        for a in [ax[i]]:
            if i not in [2,5]:
                a.set_xticklabels([])
            else:
                [tick.label.set_fontsize(ft1) for tick in a.xaxis.get_major_ticks()]
            a.tick_params(axis='x', direction='in', length=5, width=1.5)
            a.set_yticks([])
            a.set_xlim(0, 550)
            a.set_ylim(0, ylims[i])
            a.get_legend().set_visible(False)
            if i < 3:
                a.text(0.68, .78, r"$N={0}$".format(n), transform=a.transAxes, fontsize=ft1+4)
                a.text(0.68, .63, r"$S={0}$".format(n_data), transform=a.transAxes, fontsize=ft1+4)
            else:
                a.text(0.68, .25, r"$N={0}$".format(n), transform=a.transAxes, fontsize=ft1+4)
                a.text(0.68, .10, r"$S={0}$".format(n_data), transform=a.transAxes, fontsize=ft1+4)
    ax[1].set_ylabel('Normalised probability distribution', fontsize=ft1+4)
    ax[5].set_xlabel(r'Adjacent interval size ($I_A$) / cents', fontsize=ft1+4)
    ax[5].xaxis.set_label_coords(-0.05, -0.15)

    h1, l1 = ax[0].get_legend_handles_labels()
    handles = [h1[-1], Line2D([0],[0], color='w')] + h1[:-1]
    labels = ['DAT', ''] + labels
    x_anch = 1.90
    ax[0].legend(handles, labels, bbox_to_anchor=(x_anch, 1.35), frameon=False, ncol=4, fontsize=ft1)

    fig.savefig(os.path.join(FIG_DIR, 'adj_int_distribution.pdf'), bbox_inches='tight')
    


###########################
###     FIG 4   ###########
###########################

def plot_all_scale_by_model(paths, df_real, X='scale', mix=[], nbin=60, partabase='none'):
    fig, ax = plt.subplots(2,2, figsize=(12.5,  6.75))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    cols = np.array(Paired_12.hex_colors)[[11,9,3,7,0]]
    ytick = [np.arange(0, (i+1)*0.001, 0.001) for i in [3, 3, 3, 3, 2, 3]]
    ylims = [0.0044, 0.0043]
    ft1 = 21
    lw = 2

    labels = ['DAT', 'RAN', 'MIN', 'TRANS', 'HAR', 'FIF']
    col1 = RdYlGn_6.hex_colors
    col2 = Paired_12.mpl_colors
    cols = col1[:3] + [col2[3], col2[1], col2[9]]
    cols = [col1[1], col2[1], col2[3], col1[0]]
    lw = np.array([1,1,1,1.0])*2.4

    hlcol = ["orange!25", "blue!25", "green!25", "red!25"]

    bins = np.arange(15, 1200, 30)
    bins = np.linspace(15, 1185, num=40)
    xxx = bins[:-1] + 0.5 * (bins[1] - bins[0])


    for i, n in enumerate([5,7]):
        for j  in range(2):
            ax[i,j].hist(utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n, X]), bins=bins, color=[.6]*3, normed=True, alpha=0.7, label='DAT')
            histD, bins = np.histogram(utils.extract_floats_from_string(df_real.loc[df_real.n_notes==n, X]), bins=bins, normed=True)
        n_data = len(df_real.loc[df_real.n_notes==n, X])
        for count, lbl in enumerate(labels[2:]):
            histM, bins = np.histogram(utils.extract_floats_from_string(pd.read_feather(paths[lbl][n])[X]), bins=bins, normed=True)
            df_hist = pd.DataFrame(data={'bin':xxx, lbl:histM})
            
            SStot = np.sum((histD - np.mean(histD))**2)
            SSres = np.sum((histD - histM)**2)
            r2 = 1 - SSres/SStot

#           print(lbl)
#           print(linregress(histM, histD))

            if count < 2:
                sns.lineplot(x='bin', y=lbl, data=df_hist, label=lbl, color=cols[count], ax=ax[i,0], lw=lw[count])
#               ax[i,0].text(0.65, .65-(count%2)*0.1, r"$R^2={0}$".format(round(r2,2)), transform=ax[i,0].transAxes, fontsize=ft1+4, color=cols[count])

                ax[i,0].text(0.65, .70-(count%2)*0.15, r"$R^2={0}$".format(round(r2,2)), transform=ax[i,0].transAxes, fontsize=ft1+4)
                rect = mpatches.Rectangle((.63,.69-(count%2)*0.15), 0.34, 0.15, facecolor=cols[count], alpha=0.5, transform=ax[i,0].transAxes)
                ax[i,0].add_patch(rect)
            else:
                sns.lineplot(x='bin', y=lbl, data=df_hist, label=lbl, color=cols[count], ax=ax[i,1], lw=lw[count])
#               ax[i,1].text(0.65, .65-(count%2)*0.1, r"$R^2={0}$".format(round(r2,2)), transform=ax[i,1].transAxes, fontsize=ft1+4, color=cols[count])

                ax[i,1].text(0.65, .70-(count%2)*0.15, r"$R^2={0}$".format(round(r2,2)), transform=ax[i,1].transAxes, fontsize=ft1+4)
                rect = mpatches.Rectangle((.63,.69-(count%2)*0.15), 0.34, 0.15, facecolor=cols[count], alpha=0.5, transform=ax[i,1].transAxes)
                ax[i,1].add_patch(rect)
            

        for a in ax[i]:
            a.set_xticks(np.arange(0, 1201, 200))
            if i != 1:
                a.set_xticklabels([])
            else:
                a.set_xticklabels(['0', '', '400', '', '800', '', '1200'])
                [tick.label.set_fontsize(ft1+2) for tick in a.xaxis.get_major_ticks()]
            a.tick_params(axis='x', direction='in', length=5, width=1.5, color='k')
            a.set_yticks([])
            a.set_xlabel('')
            a.set_ylabel('')
            a.set_xlim(-50, 1250)
            a.set_ylim(0, ylims[i])
            a.get_legend().set_visible(False)
            a.text(0.65, .87, r"$N={0}$".format(n), transform=a.transAxes, fontsize=ft1+4)
#           a.text(0.65, .75, r"$S={0}$".format(n_data), transform=a.transAxes, fontsize=ft1+4)
#       ax[i,0].set_yticks(ytick[i])
    ax[1,0].set_xlabel('Notes in scale / cents', fontsize=ft1+6)
    ax[1,0].xaxis.set_label_coords(1.0, -0.14)
    ax[1,0].set_ylabel('Normalised probability distribution', fontsize=ft1+6)
    ax[1,0].yaxis.set_label_coords(-0.05,  1.00)

    h1, l1 = ax[0,0].get_legend_handles_labels()
    h2, l2 = ax[0,1].get_legend_handles_labels()
#   handles = np.array(h1 + h2)[[2,0,1,3,4]]
#   labels = ['DAT', 'constrained', 'HSS', 'compress', '5ths']
    handles = [h1[0],h1[1]] + [h1[2]] + [Line2D([0],[0], color='w')] + h2[:2]
    labels = ['MIN', 'TRANS', 'DAT', '', 'HAR', 'FIF']
    fig.legend(handles, labels, bbox_to_anchor=(0.68, 1.04), frameon=False, ncol=3, fontsize=ft1)
    lgd = fig.legend([], [], bbox_to_anchor=(0.71, 1.04), frameon=False, ncol=5)

    fig.savefig(os.path.join(FIG_DIR, 'scale_distribution.pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')


###########################
###     FIG 5   ###########
###########################

def model_comparison(df1, df2):
    fig, ax = plt.subplots(1,1, figsize=( 5,2.0))
    col1 = RdYlGn_11.hex_colors
    col2 = Paired_12.hex_colors
    ft = 12

    col_s = [col2[5], col1[4], col1[6], col2[3], col2[1], col2[7], 'k']
    groups = np.array(['FIF', 'HAR3', 'HAR2', 'HAR', 'TRANS', 'MIN', 'RAN'])
    lbls = ['FIF', r'$\text{HAR}^{3}$', r'$\text{HAR}^{2}$', r'$\text{HAR}$', 'TRANS', 'MIN', 'RAN']

    for i, bg in enumerate(groups):
        if i in [0,4,5,6]:
            ax.scatter(df1.loc[(df1.bias_group==bg)&(df1.method=='best'), 'met1']*1000., df1.loc[(df1.bias_group==bg)&(df1.method=='best'), 'fr_10'], color=col_s[i],  s=60, edgecolor='k', alpha=0.7, label=lbls[i])
        else:
            ax.scatter([], [], color=col_s[i],  s=60, edgecolor='k', alpha=0.7, label=lbls[i])
    for i, bg in enumerate(groups):
        if i not in [1,2,3]:
            continue
        ax.scatter(df2.loc[(df2.bias_group==bg)&(df1.method=='best'), 'met1']*1000., df2.loc[(df2.bias_group==bg)&(df1.method=='best'), 'fr_10'], color=col_s[i],  s=60, edgecolor='k', alpha=0.7)

    idx = [36, 34, 25, 19]
    ax.plot(np.array(df1.loc[idx, 'met1'])*1000., np.array(df1.loc[idx, 'fr_10']), 's', mec='k', mew=1.5, ms=10, fillstyle='none')
    idx = 13
    ax.plot(np.array(df2.loc[idx, 'met1'])*1000., np.array(df2.loc[idx, 'fr_10']), 's', mec='k', mew=1.5, ms=10, fillstyle='none')

    handles, labels = ax.get_legend_handles_labels()
    handles = [mpatches.Circle([[],[]], fc=col_s[i], ec='k', label=groups[i]) for i in range(7)]
    ax.legend(loc='best', labels=labels[:7], handles=handles[:7], frameon=False, ncol=2, columnspacing=0.5, handletextpad=0.3)
    ax.set_xlim(3.5, 10)
    ax.set_ylim(0.0, 0.48)
    ax.set_ylabel(r'$f_{\textrm{D}}$', fontsize=ft)
    ax.set_xlabel(r'$I_A$ distribution deviation ($d_{\textrm{I}}$)', fontsize=ft)
    ax.set_xticks(np.arange(4, 12, 2))
    ax.set_yticks(np.arange(0, 0.5, 0.1))

    plt.savefig(os.path.join(FIG_DIR, 'model_comparison.pdf'), bbox_inches='tight')



###########################
###     FIG 6   ###########
###########################

def plot_clusters_and_found_scales(paths, df):
    fig, ax = plt.subplots(figsize=(15, 6))
    gs  = gridspec.GridSpec(4,8, width_ratios=[1,1,1,1,.10,1,.10,0.8])
#   gs.update(wspace=0.25) #,hspace=0.20)

    ax = [plt.subplot(gs[int(i/4), i%4]) for i in range(16)]
    ax.append(plt.subplot(gs[:,5]))
    ax.append(plt.subplot(gs[:,7]))

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    theoryA = "TRANS"
    theoryB = "FIF"
    theoryC = "HAR"

    idx1 = list(set([int(i) for n in range(4,10) for ss in pd.read_feather(paths[theoryA][n])["ss_w10"].values for i in ss.split(';') if len(i)]))
    idx2 = list(set([int(i) for n in range(4,10) for ss in pd.read_feather(paths[theoryB][n])["ss_w10"].values for i in ss.split(';') if len(i)]))
    idx3 = list(set([int(i) for n in range(4,10) for ss in pd.read_feather(paths[theoryC][n])["ss_w10"].values for i in ss.split(';') if len(i)]))

    df['found'] = [True if i in idx1+idx2+idx3 else False for i in df.index]
    df['octave'] = df.scale.apply(lambda x: int(x.split(';')[-1]))

    truth_table = [lambda x, idx1, idx2: x not in idx1 and x not in idx2,
    lambda x, idx1, idx2: x in idx1 and x not in idx2,
    lambda x, idx1, idx2: x not in idx1 and x in idx2,
    lambda x, idx1, idx2: x in idx1 and x in idx2,
    lambda x, idx1, idx2: x in idx1 or x in idx2]

    lbls = ['neither', theoryA, theoryB, 'both']
    cat = 'cl_16'
    rot = [45, 45]

    x_labels =  [f'{x:15s}' for x in ['mono_N={4,5}', 'bi_N={4}', 'bi_N={6,7}', 'tri_N=5', 'bi_N=5', 'bi_N={5,6}',
                 'mono_N=7', 'mono_N={6,7}', 'bi_N={7,8,9}', 'bi_N=7',
                 'bi_N={8,9}', 'tetra_N={5,6,7}', 'bi_N={7,8,9}', 'tri_N={6,7}', 'tri_N={7,8}', 'tri_N={6,7}']]
    c_lbls = 'abcdefghijklmnop'

    ft = 12
    width = 0.80

    ###############################
    ### Stacked bar chart

    new_red = [min(1, x) for x in np.array(Paired_12.mpl_colors[5])*1.0]
    new_blu = [min(1, x) for x in np.array(Paired_12.mpl_colors[1])*1.0]

    col = [[0.8]*3, new_blu, new_red, [.5]*3]
    al = [1] + [0.7]*2 + [1]

    tots = {k:float(len(df.loc[(df.n_notes<=9)&(df.n_notes>=4)&(df[cat]==k)])) for k in df[cat].unique()}
    uniq = sorted([x for x in df.loc[df[cat].notnull(),cat].unique()])
    base = np.zeros(len(x_labels))

    idx = [i for i in df.index if truth_table[-1](i, idx1, idx2)]
    parts = {k:float(len(df.loc[(df.n_notes<=9)&(df.n_notes>=4)&(df[cat]==k)&([True if x in idx else False for x in df.index])])) for k in df[cat].unique()}
    fracs = [parts[k] / tots[k] for k in uniq]

    idxsort = np.argsort(fracs)[::-1]
    print(idxsort)
#   idxsort = [9, 4, 10, 6, 8, 3, 11, 5, 0, 15, 1, 14, 7, 13, 2, 12] 

    for i, tt in enumerate(truth_table[:4]):
        idx = [i for i in df.index if tt(i, idx1, idx2)]
        cnts = df.loc[idx, cat].value_counts()
        Y = np.array([cnts[k] if k in cnts.keys() else 0 for k in uniq])

        print(f"{lbls[i]} total: {Y.sum()}")

        X = np.arange(1,len(uniq)+1)[::-1]

        Y = Y[idxsort]

        ax[16].barh(X, -Y, width, left=-base, color=col[i], label=lbls[i], alpha=al[i])
        base = Y + base

#       if not i:
#           sns.catplot(y='cl_16', x='p_80_1200', data=df, ax=ax[17], order=np.array(idxsort)+1, orient='h', kind='box', color='grey')
#           ax[17].set_xscale('log')
#           ax[17].set_xticks([10**-11, 10**-9, 10**-7])
#           ax[17].set_xticklabels([r"$10^{{{0}}}$".format(s) for s in [-11, -9, -7]], fontsize=ft+4)
#           ax[17].set_xlim(5*10**-13, 10**-6)
#           ax[17].set_xlabel(r"$P_{MIN}$", fontsize=ft+4)

    ax[16].set_ylim(0.5, 16.5)
    ax[16].set_yticks([])
    ax[16].set_xticks(np.arange(0,-125,-25))
    ax[16].set_xticklabels(np.arange(0,125,25), fontsize=ft+1)
    ax[16].set_xlabel(f"scales found", fontsize=ft+4)

    ax[16].spines['top'].set_visible(False)
    ax[17].spines['top'].set_visible(False)
    ax[16].spines['left'].set_visible(False)
    ax[17].spines['right'].set_visible(False)

    not_found = {k:len(df.loc[(df.n_notes<=9)&(df.n_notes>=4)&(df[cat]==k)&(df.found==False)]) for k in df[cat].unique()}
    reason1 = {k:len(df.loc[(df.n_notes<=9)&(df.n_notes>=4)&(df[cat]==k)&(df.found==False)&((df.min_int<71)|(np.abs(1200-df.octave)>10))]) for k in df[cat].unique()}
    reason2 = {k:len(df.loc[(df.n_notes<=9)&(df.n_notes>=4)&(df[cat]==k)&(df.found==False)&(df.min_int>=71)&(np.abs(1200-df.octave)<=10)&(df.pMIN<10**-9.95)]) for k in df[cat].unique()}
    reason3 = {k:len(df.loc[(df.n_notes<=9)&(df.n_notes>=4)&(df[cat]==k)&(df.found==False)&(df.min_int>=71)&(np.abs(1200-df.octave)<=10)&(df.pMIN>10**-9.95)&(df.pALL>10**-3.20)]) for k in df[cat].unique()}
    reason4 = {k:len(df.loc[(df.n_notes<=9)&(df.n_notes>=4)&(df[cat]==k)&(df.found==False)&(df.min_int>=71)&(np.abs(1200-df.octave)<=10)&(df.pMIN>10**-9.95)&(df.pALL<10**-3.20)]) for k in df[cat].unique()}
    base = np.zeros(len(x_labels))
    X = np.arange(1,17)[::-1]

    col_r = np.array(Accent_4.hex_colors)[[0,3,1,2]]
    lbl_r = ['i', 'ii', 'iii', 'iv']
    for i, r in enumerate([reason1, reason2, reason3, reason4]):
        Y = np.array([r[k] if k in r.keys() else 0 for k in uniq])[idxsort]
        ax[17].barh(X, Y, width, left=base, color=col_r[i], label=lbl_r[i], edgecolor='k', linewidth=0.1)
        base = Y + base
        print(f"{lbl_r[i]} total: {sum(Y)}")

    ax[17].set_xlabel("scales not found", fontsize=ft+4)
    ax[17].set_xticks(np.arange(0,100,25))
    ax[17].set_xticklabels(np.arange(0,100,25), fontsize=ft+1)
    ax[17].set_ylim(0.5, 16.5)
    ax[17].set_yticks(X)
    ax[17].set_ylabel('')
    ax[17].set_yticklabels(list(c_lbls), fontsize=ft+4, ha='center')
    ax[17].tick_params(axis='y', which='major', pad=15, length=0)

    ###############################
    ### Cluster pair_int distributions

    cl = 16
    bins = 50
    Cont = sorted(df.Continent.unique())
    Cont = ['Western', 'Middle East', 'South Asia', 'East Asia', 'South East Asia', 'Africa', 'Oceania', 'South America']
    cols = Dark2_8.hex_colors
    col_dic = {Cont[i]:cols[i] for i in range(8)}
    lab_uni = sorted([x for x in df[f"cl_{cl:02d}"].unique() if not np.isnan(x)])

    inset = []
    x_c = 0.1837
    x_m = 0.1284
    y_c = 0.1954
    y_m = 0.1949
    axes = [ [x, y, .065, .10] for y in np.arange(0, y_m*3+.01, y_m)[::-1]+y_c for x in np.arange(0, x_m*3+.01, x_m)+x_c]

    show_x = [0,4,8,12]
    show_y = [12,13,14,15]

    bins = range(0,550,20)
    bar_col = np.array(Dark2_8.hex_colors)[[0,1,2,5,7,6,3,4]]

    for i, lbl in enumerate(np.array(lab_uni)[idxsort]):
        ax[i].hist(utils.extract_floats_from_string(df.loc[df[f"cl_{cl:02d}"]==lbl, 'pair_ints']), bins=bins, label='DAT', color=[.6]*3, normed=True, alpha=0.7, edgecolor='k')
#       sns.kdeplot(utils.extract_floats_from_string(df.loc[df[f"cl_{cl:02d}"]==lbl, 'pair_ints']), ax=ax[i], color='k')
#       sns.distplot(df.loc[df[f"cl_{cl:02d}"]==lbl, 'p_80_450'], bins=bins, ax=ax[i])
        n_data = len(df.loc[df[f"cl_{cl:02d}"]==lbl, 'pair_ints'])

        value_cnt = df.loc[df[f"cl_{cl:02d}"]==lbl,'Continent'].value_counts()
        keys = value_cnt.keys()
        freq = [value_cnt[c] if c in keys else 0 for c in Cont]

#       vals = value_cnt.values
#       freq = {keys[j]:float(vals[j])/float(sum(vals)) for j in range(len(vals))}
#       wc.generate_from_frequencies(freq)
        inset.append(fig.add_axes(axes[i]))
        if i==3:
            inset[-1].bar(range(len(Cont)), freq, color=bar_col, label=Cont[i], ec='k', lw=0.2)
        else:
            inset[-1].bar(range(len(Cont)), freq, color=bar_col, ec='k', lw=0.0)
#       inset[-1].imshow(wc)
        plt.setp(inset[-1], xticks=[], yticks=[])

        ax[i].set_xlim(-0.001, 0.008)
        ax[i].set_xlim(30, 550)
        ax[i].set_ylim(0, 0.04)
#       ax[i].set_title(f"N={df.loc[df[f'cl_{cl:02d}']==lbl,'n_notes'].unique()}  sample={sum(vals)}")
#       ax[i].text(0.6, .25, r"$S={0}$".format(n_data), transform=ax[i].transAxes)#size=40)
        ax[i].text(0.03, .80, c_lbls[i], transform=ax[i].transAxes, fontsize=ft+8)
        ax[i].tick_params(axis='x', direction='in', length=7, width=2.5)

#       if i%4 != 0:
        ax[i].set_yticks([])
#       else:
#           ax[i].set_yticks([0, 0.02, 0.04])
        ax[i].set_xticks(range(100,600,100))
        if i < 12:
            ax[i].set_xticklabels([])
        else:
            ax[i].set_xticklabels(range(100,600,100), fontsize=ft+1)
#           ax[i].set_xticklabels(['', '200', '', '400', ''], fontsize=ft+2)
    ax[8].set_ylabel('Normalised probability distribution', fontsize=ft+4)
    ax[8].yaxis.set_label_coords(-0.05, 1.00)
    ax[14].set_xlabel('Adjacent interval size ($I_A$) / cents', fontsize=ft+4)
    ax[14].xaxis.set_label_coords( 0.0, -0.25)

    ax[0].text( -0.15, 1.20, "A", transform=ax[0].transAxes, fontsize=ft+18)
    ax[16].text(-0.05, 1.05, "B", transform=ax[16].transAxes, fontsize=ft+18)
    ax[17].text(-0.05, 1.05, "C", transform=ax[17].transAxes, fontsize=ft+18)

    ### Legends
    handles = [mpatches.Patch(color=bar_col[i], label=Cont[i], ec='k', lw=0.2) for i in range(len(Cont))]
    inset[3].legend(loc='center right', bbox_to_anchor=( 0.70,  1.45), handles=handles, frameon=False, ncol=4, fontsize=ft+2, columnspacing=1.0, handletextpad=0.2)
    ax[16].legend(loc='upper left', bbox_to_anchor=( 0.10, 1.15), fontsize=ft+2, frameon=False, ncol=2, columnspacing=1, handlelength=1., handletextpad=0.2)
    ax[17].legend(loc='upper left', bbox_to_anchor=( 0.10, 1.15), fontsize=ft+2, frameon=False, ncol=2, columnspacing=1, handlelength=1., handletextpad=0.2)

    fig.savefig(os.path.join(FIG_DIR, 'clustering.pdf'), bbox_inches='tight')
#   plt.close()









