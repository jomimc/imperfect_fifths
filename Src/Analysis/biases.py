import re
import os
import sys
import time

from itertools import permutations, combinations, product
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import cumfreq
from sklearn.cluster import DBSCAN
import statsmodels.nonparametric.api as smnp

import utils

TEMP_MIN = 50.
TEMP_MAX = 300.
TEMP_LOW_MARGIN = 0.50
TEMP_HI_MARGIN = 1.50
N_TRIALS = 50

def calculate_energy(ints, base, m, n):
    return np.mean([abs(round(i/base) - i/base)**m * float(round(i/base))**n for i in ints])


def calculate_energy_2(ints, base, m, n):
    return np.mean([(abs(round(i/base) - i/base) * base)**m * float(round(i/base))**n for i in ints])


def template_function(inp, new=False):
#   ints, m, n, TEMP_LOW_MARGIN = inp
    ints, m, n = inp
    temp_min = max(TEMP_MIN, min(ints)*TEMP_LOW_MARGIN)
    temp_max = min(TEMP_MAX, min(ints)*TEMP_HI_MARGIN)
    baseArr = np.linspace(temp_min, temp_max, num=N_TRIALS)
    energies = np.zeros(baseArr.size, dtype=float)
    for i, base in enumerate(baseArr):
        if new:
            energies[i] = calculate_energy_2(ints, base, m, n)
        else:
            energies[i] = calculate_energy(ints, base, m, n)
    if len(np.where(energies==0)[0]) > 1:
        idxMin = np.where(energies==0)[0][-1]
    else:
        idxMin = np.argmin(energies)
#   return baseArr[idxMin], energies[idxMin]
    return energies[idxMin]
    
def min_int_template(inp):
    ints, m, n = inp
    base = min(ints)
    return  calculate_energy(ints, base, m, n)
    
def get_cumulative_frequency(out):
    limits = (min([min(o) for o in out]), max([max(o) for o in out]))
    freq = [cumfreq(o, numbins=1000, defaultreallimits=limits).cumcount / float(len(o)) for o in out]
    return freq

def evaluate_biases(df_list, names):

    pool = mp.Pool(28)
    df = pd.DataFrame(columns=['bias', 'm', 'n', 'min_temp']+names)
    
    for m in range(4):
        for n in range(4):
            print(m, n)
            if (m + n == 0):
                continue
            for min_temp in [0.5, 0.6, 0.7, 0.8]:
                outputs = [np.array(list(pool.imap_unordered(template_function,  [([float(x) for x in pi.split(';')], m, n, min_temp) for pi in df.pair_ints]))) for df in df_list]
                freq = get_cumulative_frequency([o[:,1] for o in outputs])

                if m == 0:
                    bias = 'S#1'
                elif n == 0:
                    bias = 'distI'
                else:
                    bias = 'distI_S#1'
                df.loc[len(df)] = [bias, m, n, min_temp] + freq

    return df

def evaluate_distW(df_list, names):
    df = pd.DataFrame(columns=['log', 'alpha']+names)
    for alpha in [0.1, 1.0]:
        outputs = [[x + alpha for x in df.opt_c] for df in df_list]
        freq = get_cumulative_frequency(outputs)
        df.loc[len(df)] = [True, alpha] + freq

        try:
            outputs = [[np.log10(x + alpha) for x in df.opt_c] for df in df_list]
            freq = get_cumulative_frequency(outputs)
        except:
            freq = [np.zeros(len(o)) for o in outputs]
        df.loc[len(df)] = [False, alpha] + freq
    return df

def plot_distI_performance(bias_df):
    fig, ax = plt.subplots(2,4, sharex=True, sharey=True)
    ax = ax.reshape(ax.size)
    [a.set_xlabel('df_real') for a in ax[4:]]
    ax[0].set_ylabel('df_rand')
    ax[4].set_ylabel('df_constrain')
    ax[0].set_title('min_temp = 0.5')
    ax[3].set_title('min_temp = 0.8')

    for i in range(len(bias_df)):
        m = bias_df.loc[i, 'm']
        n = bias_df.loc[i, 'n']
        min_temp = bias_df.loc[i, 'min_temp']

        j = int(min_temp*10-5)

#       if min_temp != 0.8:
#           continue

        lbl = f"m={bias_df.loc[i, 'm']}, n={bias_df.loc[i, 'n']}"
        if m ==0:
            pt = '-'
        elif n == 0:
            pt = ':'
        else:
            pt = '--'

        if 1:
#           ax[j].plot(bias_df.loc[i, 'real'], bias_df.loc[i, 'rand']     /bias_df.loc[i, 'real'], pt,  label=lbl)
            ax[j].plot(bias_df.loc[i, 'real'], bias_df.loc[i, 'constrain'], pt,  label=lbl)
            ax[j+4].plot(bias_df.loc[i, 'real'], bias_df.loc[i, 'constrain']/bias_df.loc[i, 'real'], pt,  label=lbl)
        else:
            ax[j].plot(bias_df.loc[i, 'real'], bias_df.loc[i, 'rand'], pt,  label=lbl)
            ax[j+4].plot(bias_df.loc[i, 'real'], bias_df.loc[i, 'constrain'], pt,  label=lbl)
    ax[4].legend(loc='lower left')


def plot_distW_performance(bias_df):
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    ax = ax.reshape(ax.size)
    [a.set_xlabel('df_real') for a in ax[4:]]
    ax[0].set_ylabel('df_rand')
    ax[2].set_ylabel('df_constrain')
    ax[0].set_title('min_temp = 0.5')
    ax[1].set_title('min_temp = 0.8')

    for i in range(len(bias_df)):
        alpha = bias_df.loc[i, 'alpha']
        log   = bias_df.loc[i, 'log']

        j = {True:1, False:0}[log]

#       if min_temp != 0.8:
#           continue

        lbl = f"alpha = {alpha}"
        pt = '-'

        if 1:
            ax[j].plot(bias_df.loc[i, 'real'], bias_df.loc[i, 'rand']     /bias_df.loc[i, 'real'], pt,  label=lbl)
#           ax[j].plot(bias_df.loc[i, 'real'], bias_df.loc[i, 'constrain'], pt,  label=lbl)
            ax[j+2].plot(bias_df.loc[i, 'real'], bias_df.loc[i, 'constrain']/bias_df.loc[i, 'real'], pt,  label=lbl)
        else:
            ax[j].plot(bias_df.loc[i, 'real'], bias_df.loc[i, 'rand'], pt,  label=lbl)
            ax[j+2].plot(bias_df.loc[i, 'real'], bias_df.loc[i, 'constrain'], pt,  label=lbl)
    ax[2].legend(loc='lower left')

def evaluate_hs(df_real, df_constrain):
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    ax = ax.reshape(ax.size)
    for i, n in enumerate(range(1,4)):
        for j, w in enumerate([5,10,15,20]):
#           outputs = [1./df_real.loc[df_real.n_notes==7, f"hs_n{n}_w{w:02d}"].values, 1./df_constrain[f"hs_n{n}_w{w:02d}"].values]
            outputs = [df_real.loc[df_real.n_notes==7, f"hs_n{n}_w{w:02d}"].values, df_constrain[f"hs_n{n}_w{w:02d}"].values]
            freq = get_cumulative_frequency(outputs)
            ax[j].plot(freq[0], freq[1], label=f"hs_n{n}_w{w:02d}")
#           if w == 20:
#               ax[3].plot(freq[0], freq[1], label=f"hs_n{n}_w{w:02d}")
#       ax[i].legend(loc='best')
#   ax[3].legend(loc='best')
    for i in range(3,4):
        for j, w in enumerate([5,10,15,20]):
#           outputs = [df_constrain[f"hs_n{n}_w{w:02d}"].values, df_real.loc[df_real.n_notes==7, f"hs_r3_w{w:02d}"].values]
            outputs = [df_real.loc[df_real.n_notes==7, f"hs_r3_w{w:02d}"].values, df_constrain[f"hs_n{n}_w{w:02d}"].values]
            freq = get_cumulative_frequency(outputs)
            ax[j].plot(freq[0], freq[1], label=f"hs_n{n}_w{w:02d}")
    for i in range(4):
        ax[i].legend(loc='best')
            

def evaluate_cost_function_old(df, a=1, b=0, cat='hs_n1_w10', plot=True):
    Xmax = df[cat].max()
    bins = np.linspace(0, Xmax*1.2, 101)
    dx = bins[1] - bins[0]
    X = bins[:-1] + dx/2.
#   print(df[cat].min(), df[cat].max())

    pX, bins = np.histogram(df[cat], bins=bins)
    pX = pX / np.sum(pX)

    return pX, bins[:-1] + 0.5 * (bins[1]-bins[0])

    if plot:
        fig, ax = plt.subplots(2,1)

    functions = [lambda x, a, b: np.exp(-a / (x-b)**2),
                 lambda x, a, b: np.exp(-a * (1-x/b)**2)]
#   functions = [lambda x, a, b: np.exp(-a * (1-x/b)),
#                lambda x, a, b: np.exp(-a * (1-x/b)**5)]

    acc, sel = [], []

    for i, fn in enumerate(functions):
#   qX = pX * np.array([min(1,x) for x in np.exp(-a / (X+b))])
#   qX = pX * np.array([min(1,x) for x in np.exp(-a * (1 - X/b))])
        qX = pX * np.array([min(1, fn(x, a, b)) for x in X])
        qX[np.isnan(qX)]=0
        acceptance = np.sum(qX)

        qX = qX / np.sum(qX)
        selectivity =  utils.calc_jensen_shannon_distance(pX, qX)
        acc.append(acceptance)
        sel.append(selectivity)

    #   print(f"Acceptance = {acceptance}")
    #   print(f"Selectivity = {selectivity}")

    #   plt.plot([selectivity], [acceptance], 'o', color='k')
        if plot:
            ax[i].plot(X, pX)
            ax[i].plot(X, qX)

    return acc, sel

def run_cf_eval(df, cat='hs_n1_w10'):
    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    A = np.arange(1,500,10)
#   for b in np.arange( 30,100,10):
#   for b in [-10, -1, -0.1, -0.01, 0.1, 1, 2, 6]:
    for b in [9.46]:
        evil = np.array([evaluate_cost_function_old(df, cat=cat, a=a, b=b, plot=False) for a in A])
        ax[0].plot(evil[:,1,0], evil[:,0,0], label=b)
        ax[1].plot(evil[:,1,1], evil[:,0,1], label=b)
    for a in ax:
        a.set_yscale('log')
        a.legend(loc='best', frameon=False)
    print(df[cat].min(), df[cat].max())

    
def evaluate_cost_function(X, pX, fn, params, qX_only=False, plot=False):
    qX = pX * np.array([min(1, fn(x, *params)) for x in X])
    qX[np.isnan(qX)]=0
    acc = np.sum(qX)

    qX = qX / np.sum(qX)
    if qX_only:
        return qX
    sel = jensenshannon(pX, qX)

    if plot:
#       fig, ax = plt.subplots()
        plt.plot(X, pX)
        plt.plot(X, qX)

    return acc, sel


def get_beta_for_acc(sel, beta, target):
    for i, s in enumerate(sel):
        try:
            if s <= target <= sel[i+1]:
                return beta[i] + (target - s) * (beta[i+1] - beta[i]) / (sel[i+1] - s)
        except IndexError:
            print("Target acceptance not in range")





