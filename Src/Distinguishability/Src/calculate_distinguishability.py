import os 
import string
import sys

import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import multiprocessing as mp
import numpy as np
import pandas as pd
from palettable.colorbrewer.qualitative import Paired_12
import seaborn as sns
import scipy.stats as stats

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


INT_MIN =  0.
INT_MAX = 250.
D_INT = 5.
ACC = 0.99
INTS = np.arange(INT_MIN, INT_MAX, D_INT)

FIGS_DIR = '/home/johnmcbride/Dropbox/phd/LaTEX/Scales/Figures/'

def gaussian(x, mean, var):
    return np.exp( - (x - mean)**2 / (2. * var)) / (var * 2 * np.pi)**0.5

def integrate_gauss(mean, var, x1, x2, num=1000):
    X = np.linspace(x1, x2, num=num)
    P =  gaussian(X, mean, var)
#   return X, P
    return np.trapz(P, X)

def get_percentage_correct_from_range_of_ints(dI, prod_var, percep_var, ints=INTS):
    correct = []
    for I0 in ints:
        int_cats = np.arange(0, I0*2, dI)
        prob_produced = []
        prob_correct = []
        for I1 in int_cats:
            prod_prob = integrate_gauss(I0, prod_var, I1-dI/2., I1+dI/2.)
            percep_prob = [integrate_gauss(i, percep_var, I1-dI/2., I1+dI/2.) for i in [0, I0, I0*2]]

            prob_produced.append(prod_prob)
            prob_correct.append(percep_prob[1] / sum(percep_prob))

        correct.append(np.sum(np.array(prob_produced) * np.array(prob_correct)) / np.sum(prob_produced))
    return np.array(correct)

def get_percentage_correct_from_range_of_ints_2(dI, prod_var, percep_var, ints=INTS):
    correct = []
    for I0 in ints:
        int_cats = np.arange(0, I0*2, dI)
        prob_produced = []
        prob_correct = []
        for I1 in int_cats:
            prod_prob = integrate_gauss(I0, prod_var, I1-dI/2., I1+dI/2.)
            percep_prob = [integrate_gauss(i, percep_var, I1-dI/2., I1+dI/2.) for i in [0, I0]]

            prob_produced.append(prod_prob)
            prob_correct.append(percep_prob[1] / sum(percep_prob))

        correct.append(np.sum(np.array(prob_produced) * np.array(prob_correct)) / np.sum(prob_produced))
    return np.array(correct)

def get_interval_by_accuracy(ints, correct, acc=ACC):
    try:
        i = np.where(correct > acc)[0][0]
    except:
        i = np.argmin(np.abs(correct - acc))
    if i:
        return ints[i-1] + (ints[i] - ints[i-1]) * (acc - correct[i-1]) / (correct[i] - correct[i-1])
    else:
        return ints[0]

def plot_distinguishability_by_grid_size():
    dI = 5

    dI_arr = [3, 5, 10, 20, 25, 30]

    prod_sdev_arr = np.arange(5., 32.5, 2.5)
    percep_sdev_arr = np.arange(5., 32.5, 2.5)

    fig1, ax1 = plt.subplots(2,3)
    ax1 = ax1.reshape(ax1.size)

    df_list = []

    for i, dI in enumerate(dI_arr):
        xi, yi = np.meshgrid(prod_sdev_arr, percep_sdev_arr)
        prod_in = xi.ravel()
        percep_in = yi.ravel()

        pool = mp.Pool(24)

        correct = pool.starmap(get_percentage_correct_from_range_of_ints, [(dI, prod_in[i]**2, percep_in[i]**2) for i in range(len(prod_in))])
        thresh_list = [get_interval_by_accuracy(INTS, c) for c in correct]
        df_list.append(pd.DataFrame(data={'production':prod_in, 'perception':percep_in, 'threshold':thresh_list, 'dI':[dI]*prod_in.size}))

        sns.heatmap(df_list[i].pivot('production', 'perception', 'threshold'), ax=ax1[i], vmin=50, vmax=180, annot=True)
        ax1[i].invert_yaxis()
        ax1[i].set_title(f"dI = {dI}")
   
#   plt.legend(loc='best')
#   plt.plot(np.arange(50, 550, 50), thresh_int)

    plt.show()

def plot_distinguishability_ranges():
    dI = 5

    min_prod = [5., 10., 30.]
    min_per  = [10., 20., 40.]
    rang = 27.5
    titles = ['expert', 'good_untrained', 'bad_untrained']

    fig, ax = plt.subplots(3)

    for i in range(3):
        prod_sdev_arr = np.arange(min_prod[i], min_prod[i]+rang, 2.5)
        percep_sdev_arr = np.arange(min_per[i], min_per[i]+rang, 2.5)

        xi, yi = np.meshgrid(prod_sdev_arr, percep_sdev_arr)
        prod_in = xi.ravel()
        percep_in = yi.ravel()

        pool = mp.Pool(28)

        correct = pool.starmap(get_percentage_correct_from_range_of_ints, [(dI, prod_in[j]**2, percep_in[j]**2) for j in range(len(prod_in))])
        thresh_list = [get_interval_by_accuracy(INTS, c) for c in correct]

        annot = np.zeros(xi.shape, dtype='<U3')
        np.fill_diagonal(annot, [str(int(x)) for x in np.array(thresh_list).reshape(xi.shape).T.diagonal()])

        df = pd.DataFrame(data={'production':prod_in, 'perception':percep_in, 'threshold':thresh_list, 'dI':[dI]*prod_in.size})

        sns.heatmap(df.pivot('production', 'perception', 'threshold'), ax=ax[i], vmin=50, vmax=180, annot=annot, fmt="s")
        ax[i].invert_yaxis()
        ax[i].set_title(titles[i])
       
    plt.show()

def plot_distinguishability_ranges_one_plot():
    dI = 5

    min_prod = [10., 20., 40.]
    min_per  = [10., 20., 40.]
    rang = 27.5
    titles = ['expert', 'good_untrained', 'bad_untrained']

    fig, ax = plt.subplots()

    prod_sdev_arr = np.arange(5, 57.5, 5)
    percep_sdev_arr = np.arange(5, 57.5, 5)

    xi, yi = np.meshgrid(prod_sdev_arr, percep_sdev_arr)
    prod_in = xi.ravel()
    percep_in = yi.ravel()

    pool = mp.Pool(28)

    correct = pool.starmap(get_percentage_correct_from_range_of_ints, [(dI, prod_in[j]**2, percep_in[j]**2) for j in range(len(prod_in))])
    thresh_list = [get_interval_by_accuracy(INTS, c) for c in correct]

    annot = np.zeros(xi.shape, dtype='<U3')
    np.fill_diagonal(annot, [str(int(x)) for x in np.array(thresh_list).reshape(xi.shape).T.diagonal()])
    np.save('Results/annotations', annot)

    df = pd.DataFrame(data={'production':prod_in, 'perception':percep_in, 'threshold':thresh_list, 'dI':[dI]*prod_in.size})

    df.to_feather(f'Results/three_notes_acc{ACC}.feather')

    xticks = np.arange(5, 55, 5)
    yticks = np.arange(5, 55, 5)
    sns.heatmap(df.pivot('production', 'perception', 'threshold'), ax=ax, vmin=30, vmax=300, annot=annot, fmt="s", xticklabels=xticks, yticklabels=yticks)

    ax.invert_yaxis()

    ax_scale = 5.0
    ax.set_xticks((np.arange(5, 55, 5)-2.5)/ax_scale)
    ax.set_yticks((np.arange(5, 55, 5)-2.5)/ax_scale)

    plt.savefig('Figs/accurate_intervals.png', dpi=1200)
    plt.savefig('Figs/accurate_intervals.pdf', dpi=1200)
#   plt.show()

def plot_distinguishability_two_notes():
    dI = 5

    min_prod = [10., 20., 40.]
    min_per  = [10., 20., 40.]
    rang = 27.5
    titles = ['expert', 'good_untrained', 'bad_untrained']

    fig, ax = plt.subplots()

    prod_sdev_arr = np.arange(2.5, 57.5, 2.5)
    percep_sdev_arr = np.arange(2.5, 57.5, 2.5)

    xi, yi = np.meshgrid(prod_sdev_arr, percep_sdev_arr)
    prod_in = xi.ravel()
    percep_in = yi.ravel()

    pool = mp.Pool(28)

    correct = pool.starmap(get_percentage_correct_from_range_of_ints_2, [(dI, prod_in[j]**2, percep_in[j]**2) for j in range(len(prod_in))])
    thresh_list = [get_interval_by_accuracy(INTS, c) for c in correct]

    annot = np.zeros(xi.shape, dtype='<U3')
    np.fill_diagonal(annot, [str(int(x)) for x in np.array(thresh_list).reshape(xi.shape).T.diagonal()])

    df = pd.DataFrame(data={'production':prod_in, 'perception':percep_in, 'threshold':thresh_list, 'dI':[dI]*prod_in.size})

    xticks = np.arange(5, 55, 5)
    yticks = np.arange(5, 55, 5)
    sns.heatmap(df.pivot('production', 'perception', 'threshold'), ax=ax, vmin=30, vmax=300, annot=annot, fmt="s", xticklabels=xticks, yticklabels=yticks)

    ax.invert_yaxis()

    ax_scale = 2.5
    ax.set_xticks((np.arange(5, 55, 5)-2.5)/ax_scale)
    ax.set_yticks((np.arange(5, 55, 5)-2.5)/ax_scale)

    plt.savefig('Figs/two_notes_accurate_intervals.png', dpi=1200)
    plt.savefig('Figs/two_notes_accurate_intervals.pdf', dpi=1200)
#   plt.show()

def plot_frac_correct():
    fig, ax = plt.subplots()
    dI = 2
    for std in [5, 10, 20, 40]:
        correct = get_percentage_correct_from_range_of_ints(dI, std**2, std**2)
        ax.plot(INTS, correct, label=r"$\sigma = {0}$".format(std))
    ax.legend(loc='best', frameon=False)
    plt.show()

def plot_heatmap():
    fig, ax = plt.subplots()
    df = pd.read_feather(f'Results/three_notes_acc{ACC}.feather')

    annot = np.load('Results/annotations.npy')
    xticks = np.arange(5, 55, 5)
    yticks = np.arange(5, 55, 5)
    sns.heatmap(df.pivot('production', 'perception', 'threshold'), ax=ax, vmin=30, vmax=300, annot=annot, fmt="s", xticklabels=xticks, yticklabels=yticks)

    ax.invert_yaxis()

    ax_scale = 5.0
    ax.set_xticks((np.arange(5, 55, 5)-2.5)/ax_scale)
    ax.set_yticks((np.arange(5, 55, 5)-2.5)/ax_scale)

    plt.savefig('Figs/accurate_intervals.png', dpi=1200)
    plt.savefig('Figs/accurate_intervals.pdf', dpi=1200)

def plot_SI():
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(2,3, width_ratios=[1.0, 1.0, 1.8], height_ratios=[1.0, 1.0])
    gs.update(wspace=0.30 ,hspace=0.40)

    ax = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[:,1]),fig.add_subplot(gs[:,2])]

    std = 20
    X = np.linspace(0, 200, num=1000)
    ax[0].plot(X, stats.norm.pdf(X, 100, std), label=f"Category A", c='k')

    col = ['k'] + list(np.array(Paired_12.mpl_colors)[[1,3,5]])
    cat = [f"Category {s}" for s in 'ABC']
    Y = []
    for i, mu in enumerate([100, 50, 150]):
        Y.append(stats.norm.pdf(X,  mu, std))
#       ax[1].plot(X, stats.norm.pdf(X,  mu, std), label=cat[i], c=col[i])

    Y = np.array(Y)
    ysum = np.sum(Y, axis=0)
    Y = Y/ysum
    for i, mu in enumerate([100, 50, 150]):
        ax[1].plot(X, Y[i], '-', label=cat[i], c=col[i])

    ax[0].set_xlabel("Produced interval")
    ax[1].set_xlabel("Produced interval")
    ax[0].set_ylabel("Probability")
    ax[1].set_ylabel(r"$P_{Cat}$")
    ax[0].set_ylim(0, 0.035)
    ax[1].set_ylim(0, 1.70)
    ax[0].set_yticks([])
    ax[1].set_yticks([0,1])
    for a in ax[:2]:
        a.legend(loc='upper right', frameon=False)


    dI = 2
    for i, std in enumerate([5, 10, 20, 40]):
        correct = get_percentage_correct_from_range_of_ints(dI, std**2, std**2)
        line, = ax[2].plot(INTS, correct, label=r"$\sigma = {0}$".format(std), c='k')
        line.set_dashes([12-i*2-3, 3+i*0])
    ax[2].legend(loc='best', frameon=False)
    ax[2].plot([0,250],[.99]*2, '-', c=col[3], alpha=0.7)
    ax[2].set_xlim(0, 250)

    ax[2].set_xlabel(r'$I_{\textrm{min}}$')
    ax[2].set_ylabel("Fraction correctly perceived")

    df = pd.read_feather(f'Results/three_notes_acc{ACC}.feather')
    annot = np.load('Results/annotations.npy')
    xticks = np.arange(5, 55, 5)
    yticks = np.arange(5, 55, 5)
    sns.heatmap(df.pivot('production', 'perception', 'threshold'), ax=ax[3], vmin=30, vmax=300,
                annot=annot, fmt="s", xticklabels=xticks, yticklabels=yticks, cbar_kws={'label':r'$I_{\textrm{min}}$'})
    ax[3].invert_yaxis()

    ax_scale = 5.0
    ax[3].set_xticks((np.arange(5, 55, 5)-2.5)/ax_scale)
    ax[3].set_yticks((np.arange(5, 55, 5)-2.5)/ax_scale)
    ax[3].set_xlabel(r'$\sigma_{per}$')
    ax[3].set_ylabel(r'$\sigma_{prod}$')

    X = [-0.11, -0.11, -0.27, -0.17]
    Y = [1.05, 1.05, 1.02, 1.02]
    for i, a in enumerate(ax):
        a.text(X[i], Y[i], string.ascii_uppercase[i], transform=a.transAxes, weight='bold', fontsize=16)

    plt.savefig(FIGS_DIR + 'transmission_model.pdf', bbox_inches='tight')


if __name__ == "__main__":

#   plot_distinguishability_ranges()
#   plot_distinguishability_ranges_one_plot()
#   plot_distinguishability_two_notes()

#   plot_frac_correct()
#   plot_heatmap()

    plot_SI()
            
    

