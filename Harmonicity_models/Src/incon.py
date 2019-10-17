from collections import Counter
from itertools import product
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import linregress, pearsonr
from scipy.stats.mstats import zscore
from sklearn import linear_model

PATH_MODELS = "/home/johnmcbride/projects/Scales/Vocal/incon/models.txt"
MODELS = np.array([l.strip('\n').replace(' ','') for l in open(PATH_MODELS, 'r')])


def get_consonance_metric_correlations(f=60):
    incon = importr('incon')
    scores = np.array([[float(incon.incon(f"{f} {f+i}", str(m))[0]) for i in range(1,13)]for m in MODELS])

    r_vals = np.zeros((scores.shape[0], scores.shape[0]), dtype=float)
    conf   = np.zeros((scores.shape[0], scores.shape[0]), dtype=float)
    for i in range(scores.shape[0]):
        for j in range(i, scores.shape[0]):
            r = pearsonr(scores[i], scores[j])
            r_vals[i,j] = r[0]
            r_vals[j,i] = r[0]
            conf[i,j] = r[1]
            conf[j,i] = r[1]

    return scores, r_vals, conf


def reshape_array(arr, idx):
    new_arr = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            new_arr[i,j] = arr[idx[i], idx[j]]
    return new_arr


def cluster_models(r_vals, n=4):
    li = linkage(pdist(np.abs(r_vals)), 'ward')
    return fcluster(li, li[-n,2], criterion='distance')
    

def plot_correlations(r_vals, idx='None'):
    fig, ax = plt.subplots()
    if idx == 'None':
        im = ax.imshow(np.abs(r_vals))
        ax.set_yticklabels(MODELS)
    else:
        im = ax.imshow(reshape_array(np.abs(r_vals), idx))
        ax.set_yticklabels(MODELS[idx])
    fig.colorbar(im)

    ax.invert_yaxis()
    ax.set_yticks(range(len(MODELS)))


def plot_linear_regression(scores, i, j):
    fig, ax = plt.subplots(1,3)
    ax[0].plot(scores[i] / scores[i].max(), 'o')
    ax[1].plot(scores[j] / scores[j].max(), 'o')
    ax[2].plot(scores[i] / scores[i].max(), scores[j] / scores[j].max(), 'o')
    print(linregress(scores[i], scores[j]))


def plot_bar_model_perf(r_vals):
    fig, ax = plt.subplots()
    width = 0.8
    idx = np.argsort(r_vals)
    ax.barh(range(len(r_vals)), r_vals[idx])
    ax.set_yticks(range(len(r_vals)))
    ax.set_yticklabels(MODELS[idx])


